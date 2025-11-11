# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger as logging

# from nemo.collections.llm.gpt.model.base import GPTConfig
# from nemo.lightning.pytorch.callbacks import PEFT
from coom.utils import flops_formulas

# from nemo.utils.hyena_flops_formulas import hyena

__all__ = ["FLOPsMeasurementCallback", "MM_FLOPsMeasurementCallback"]

_model_flops_map = {
    "deepseekv3": flops_formulas.deepseekv3,
    "hf_clip_vit_l": flops_formulas.clip_vit_l,
    "neva_projection": flops_formulas.neva_projection,
    "flux": flops_formulas.flux,
}


class FLOPsMeasurementCallback(Callback):
    """
    Calculate and log FLOPs per second after every ``trainer.log_every_n_steps`` steps.

    Args:
        model_config (GPTConfig): Model parameters.
        data_config (pl.LightningDataModule): Data module being used in the experiment.
        model_name (str): Name of the model being run. The following models are supported:
            gpt3, llama2, llama3, nemotron, mixtral, bert, hyena.


    """

    higher_is_better = True

    def __init__(
        self, data_config: pl.LightningDataModule, model_name: str = "deepseekv3"
    ):
        self.data_cfg = data_config
        self.model = model_name.lower()

        if "deepseek" not in self.model:
            raise ValueError(
                f"This callback is restricted to DeepSeek models only. Got: {self.model}"
            )

        # === Hardcoded DeepSeekV3 Config ===
        # Matches your MLATransformerConfig and setup exactly
        self.hardcoded_config = {
            "gbs": getattr(self.data_cfg, "global_batch_size", 16),
            "enc_seq_len": 4096,
            "hs": 4096,
            "layers": 8,
            "ffn_hs": 14336,
            "attention_heads": 8,
            "moe_router_topk": 1,
            "query_groups": 8,
            "kv_channels": 128,
            "vocab_size": getattr(
                getattr(self.data_cfg, "tokenizer", None), "vocab_size", 50258
            ),
            "moe_layer_freq": 1,
            "moe_shared_expert_intermediate_size": 2048,
            "moe_ffn_hidden_size": 14336,
            "mtp_num_layers": 0,
            "qk_head_dim": 128,
            "qk_pos_emb_head_dim": 64,
            "v_head_dim": 128,
            "q_lora_rank": 1024,
            "kv_lora_rank": 256,
        }
        # Explicit, parameter-wise construction
        self.flops_config = flops_formulas.FLOPSConfig(
            gbs=getattr(self.data_cfg, "global_batch_size", 16),
            enc_seq_len=4096,
            hs=4096,
            layers=8,
            ffn_hs=14336,
            attention_heads=8,
            moe_router_topk=1,
            query_groups=8,
            kv_channels=128,
            vocab_size=getattr(
                getattr(self.data_cfg, "tokenizer", None), "vocab_size", 50258
            ),
            moe_layer_freq=1,
            moe_shared_expert_intermediate_size=2048,
            moe_ffn_hidden_size=14336,
            mtp_num_layers=0,
            q_lora_rank=1024,
            kv_lora_rank=256,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            # Defaults for unused optional fields (satisfy dataclass signature)
            causal_self_attn=True,
            is_hybrid_model=False,
            hybrid_override_pattern=None,
            mamba_state_dim=None,
            mamba_head_dim=None,
            mamba_num_groups=None,
            mamba_num_heads=None,
            window_attn_skip_freq=None,
            window_size=(128, 0),
        )

        # self.flops_config = flops_formulas.FLOPSConfig(**self.hardcoded_config)
        self.avg_train_step_time = 0.0

        logging.info(
            "DeepSeek FLOPsMeasurementCallback initialized with hardcoded config"
        )

    # def on_train_start(self, trainer, pl_module):
    #     """
    #     PyTorch Lightning callback hook. Ensures that user is not using PEFT
    #     as FLOPS callback does not support it.
    #     """
    #     for callback in trainer.callbacks:
    #         if isinstance(callback, PEFT):
    #             raise NotImplementedError(
    #                 "FLOPs measurement not supported for finetuning jobs"
    #             )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        """Compute TFLOPs per GPU every N steps."""
        try:
            self.avg_train_step_time += trainer.progress_bar_metrics[
                "train_step_timing in s"
            ]
        except KeyError:
            print(
                "'train_step_timing in s' not found. "
                "Ensure TimingCallback is active when using DeepSeekFLOPsMeasurementCallback."
            )

        n = getattr(trainer, "global_step", 0)
        if n % trainer.log_every_n_steps == 0 and self.avg_train_step_time > 0:
            avg_time = self.avg_train_step_time / trainer.log_every_n_steps
            tflops_per_gpu, total_flops = self.eval_tflops_per_sec_per_gpu(avg_time)
            self.avg_train_step_time = 0

            pl_module.log(
                "DeepSeek_TFLOPS_per_GPU",
                tflops_per_gpu,
                on_step=True,
                on_epoch=False,
                batch_size=1,
                prog_bar=True,
            )
            pl_module.log("DeepSeek_Total_TFLOPS", total_flops / 1e12)

    def eval_tflops_per_sec_per_gpu(
        self, train_step_time: List | float | int
    ) -> Tuple[float, float]:
        """Evaluate TFLOPs per second per GPU."""
        total_flops, flops_per_gpu = self.eval_model_flops()

        if not isinstance(train_step_time, list):
            train_step_time = [train_step_time]
        step_time_arr = np.array(train_step_time)
        mean_step_time = np.mean(step_time_arr[len(step_time_arr) // 2 :])

        flops_per_sec_per_gpu = flops_per_gpu / (1e12 * mean_step_time)
        return flops_per_sec_per_gpu, total_flops

    def eval_model_flops(self) -> Tuple[float, float]:
        """Compute total and per-GPU FLOPs for DeepSeek."""
        if self.model not in _model_flops_map:
            raise KeyError(
                f"FLOPs measurement is supported only for DeepSeekV3. Got: {self.model}"
            )

        total_flops = _model_flops_map[self.model](self.flops_config)
        num_devices = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        flops_per_gpu = total_flops / num_devices
        return total_flops, flops_per_gpu


class MM_FLOPsMeasurementCallback(Callback):
    """
    Self-contained multimodal FLOPs measurement callback.
    Works without requiring model config objects.
    Hardcoded for:
      - DeepSeekV3 (LLM)
      - HF CLIP ViT-L (Vision Encoder)
      - NEVA Projection (Image-Text Projector)
      - Flux (Optional multimodal fusion)
    """

    higher_is_better = True

    def __init__(self, data_config: pl.LightningDataModule):
        self.data_cfg = data_config
        gbs = getattr(self.data_cfg, "global_batch_size", 16)
        vocab_size = getattr(
            getattr(self.data_cfg, "tokenizer", None), "vocab_size", 50258
        )

        # === Hardcoded Configs ===

        # ---- DeepSeekV3 ----
        deepseek_cfg = flops_formulas.FLOPSConfig(
            gbs=gbs,
            enc_seq_len=4096,
            hs=4096,
            layers=8,
            ffn_hs=14336,
            attention_heads=8,
            moe_router_topk=1,
            query_groups=8,
            vocab_size=vocab_size,
            moe_layer_freq=1,
            moe_shared_expert_intermediate_size=2048,
            moe_ffn_hidden_size=14336,
            # moe_aux_loss_coeff=1e-4,
        )

        # ---- HF CLIP ViT-L ----
        hf_clip_vit_l_cfg = flops_formulas.FLOPSConfig(
            gbs=gbs,
            layers=24,
            hs=1024,
            img_seq_len=257,  # 224x224 / 14x14 + cls token
            img_h=224,
            img_w=224,
            patch_dim=14,
            in_channels=3,
            class_token_len=1,
        )

        # ---- NEVA Projection ----
        neva_projection_cfg = flops_formulas.FLOPSConfig(
            gbs=gbs,
            projector_type="mlp",
            ffn_hs=4096,
            inp_s=1024,
            hs=4096,
            img_seq_len=256,
        )

        # ---- Flux ---- (optional multimodal joint encoder)
        flux_cfg = flops_formulas.FLOPSConfig(
            gbs=gbs,
            layers=[4, 4],
            hs=2048,
            model_channels=128,
            inp_s=4096,
            in_channels=3,
            vec_in_dim=1024,
        )

        self.flops_config_dict = {
            "deepseekv3": deepseek_cfg,
            "hf_clip_vit_l": hf_clip_vit_l_cfg,
            "neva_projection": neva_projection_cfg,
            "flux": flux_cfg,
        }

        self.avg_train_step_time = 0.0
        logging.info("HardcodedMMFLOPsMeasurementCallback initialized successfully")

    # ------------------------------
    # === Core Lightning Hooks ===
    # ------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        try:
            self.avg_train_step_time += trainer.progress_bar_metrics[
                "train_step_timing in s"
            ]
        except KeyError:
            print(
                "'train_step_timing in s' not found. "
                "Ensure TimingCallback is active for accurate FLOPs."
            )

        n = getattr(trainer, "global_step", 0)
        if n % trainer.log_every_n_steps == 0 and self.avg_train_step_time > 0:
            avg_time = self.avg_train_step_time / trainer.log_every_n_steps
            tflops_per_gpu, total_flops = self.eval_tflops_per_sec_per_gpu(avg_time)
            self.avg_train_step_time = 0

            pl_module.log(
                "MM_TFLOPS_per_GPU",
                tflops_per_gpu,
                on_step=True,
                on_epoch=False,
                batch_size=1,
                prog_bar=True,
            )
            pl_module.log("MM_Total_TFLOPS", total_flops / 1e12)

    # ------------------------------
    # === FLOPs Evaluation Logic ===
    # ------------------------------

    def eval_model_flops(self):
        """Compute total and per-GPU FLOPs for all multimodal submodules."""
        total_flops = 0

        for model_name, flops_cfg in self.flops_config_dict.items():
            if model_name not in _model_flops_map:
                logging.warning(
                    f"Unsupported model {model_name} for FLOPs measurement. "
                    f"Supported: {list(_model_flops_map.keys())}"
                )
                continue
            total_flops += _model_flops_map[model_name](flops_cfg)

        num_devices = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        flops_per_gpu = total_flops / num_devices
        return total_flops, flops_per_gpu

    def eval_tflops_per_sec_per_gpu(self, train_step_time):
        """Compute TFLOPs/sec/GPU."""
        total_flops, flops_per_gpu = self.eval_model_flops()
        if not isinstance(train_step_time, list):
            train_step_time = [train_step_time]
        mean_time = np.mean(train_step_time[len(train_step_time) // 2 :])
        tflops_per_gpu = flops_per_gpu / (1e12 * mean_time)
        return tflops_per_gpu, total_flops
