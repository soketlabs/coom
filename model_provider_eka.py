from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import (
    MLATransformerConfig,
    ModelParallelConfig,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from coom.training.initialize import initialize_megatron

try:
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except:
    HAVE_TE = False

from coom.model.transformer import EKAModel


def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    # non MoE
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear
            if use_te
            else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )


def get_transformer_layer_spec():
    mlp = get_mlp_module_spec(
        use_te=False,
        num_experts=None,
        moe_grouped_gemm=None,
    )

    # MLA Spec

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=WrappedTorchNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=WrappedTorchNorm,
                    k_layernorm=WrappedTorchNorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=WrappedTorchNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# def model_provider():

#     transformer_config = MLATransformerConfig(
#         num_layers=2,
#         num_attention_heads=4,
#         hidden_size=12,
#         use_cpu_initialization=True,
#         pipeline_dtype=torch.float32,
#     )
#     model = EKAModel(
#         config=transformer_config,
#         transformer_layer_spec=get_transformer_layer_spec(),
#         vocab_size=270000,
#         max_sequence_length=512,
#     )
#     return model


# def print_model_stats(model):
#     """Prints total vs trainable parameter counts (and approximate memory size)."""
#     total_params = 0
#     trainable_params = 0

#     for name, param in model.named_parameters():
#         num = param.numel()
#         total_params += num
#         if param.requires_grad:
#             trainable_params += num

#     # Convert to readable sizes
#     def format_size(n):
#         if n > 1e9:
#             return f"{n / 1e9:.2f}B"
#         elif n > 1e6:
#             return f"{n / 1e6:.2f}M"
#         else:
#             return f"{n:,}"

#     # Each param uses dtype.numel bytes (e.g. bf16 = 2 bytes)
#     dtype_size = 2 if any(p.dtype == torch.bfloat16 for p in model.parameters()) else 4
#     total_mem_gb = total_params * dtype_size / (1024**3)
#     train_mem_gb = trainable_params * dtype_size / (1024**3)

#     print("=" * 80)
#     print("MODEL PARAMETER STATS")
#     print(
#         f"Total Parameters:     {format_size(total_params)} ({total_mem_gb:.2f} GB if stored)"
#     )
#     print(
#         f"Trainable Parameters: {format_size(trainable_params)} ({train_mem_gb:.2f} GB if stored)"
#     )
#     print(f"Non-trainable:        {format_size(total_params - trainable_params)}")
#     print("=" * 80)


def model_provider():
    """Build Mixtral-style model configuration using Megatron-Core primitives (1-GPU version)."""

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=2,
        moe_grouped_gemm=False,  # Often False for this spec
        qk_layernorm=True,
        multi_latent_attention=True,
        fp8=None,
        moe_use_legacy_grouped_gemm=False,
        # normalization="RMSNorm",
        qk_l2_norm=False,
        use_kitchen=False,
    )
    # ProcessGroupCollection(expt_tp=2, ep=2)

    # ~8B total, ~1B active parameters

    transformer_config = MLATransformerConfig(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,  # EP size is 4 across 2 pipelines stages maybe?
        context_parallel_size=1,
        expert_tensor_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=True,
        num_layers=8,
        num_moe_experts=2,
        num_attention_heads=8,
        hidden_size=4096,
        ffn_hidden_size=2048,
        moe_ffn_hidden_size=14336,
        use_cpu_initialization=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        init_model_with_meta_device=False,
        # === Mixture-of-Experts ===
        moe_router_topk=1,
        moe_layer_freq=2,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_topk_scaling_factor=1.0,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-3,
        moe_router_dtype=torch.float32,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="seq_aux_loss",
        # moe_enable_deepep=True,
        # moe_deepep_num_sms=10,
        moe_aux_loss_coeff=1e-4,
        moe_permute_fusion=True,
        moe_shared_expert_intermediate_size=2048,
        # === Attention / LoRA ===
        q_lora_rank=1024,
        kv_lora_rank=256,
        qk_head_dim=128,
        # kv_channels should be the same as qk_head_dim * num_query_groups if GQA
        kv_channels=8,  # 128, prev now changed according to deepseek
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        rotary_scaling_factor=40,
        mscale=1.0,
        mscale_all_dim=1.0,
        num_query_groups=8,
        # === Positional Embeddings ===
        max_position_embeddings=4096,
        rotary_base=10000,
        qk_layernorm=True,
        add_bias_linear=False,  # Revert to False for shared experts
        moe_grouped_gemm=True,
        bf16=True,
        # moe_token_dispatcher_type="flex",
        multi_latent_attention=True,
        # Activation for Swiglu
        gated_linear_unit=True,
        activation_func=F.silu,
        # bias_activation_fusion=True,
        use_te_activation_func=True,
        # cache_mla_latents=True,  # uses flash attn if available
    )
    model = EKAModel(
        config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=50258,
        max_sequence_length=4096,
    )

    print("Model initialized for single-GPU inference/training.")
    print(model)
    print(model.decoder.layers[0].mlp.experts.local_experts[0].activation_func)
    print(model.decoder.layers[0].mlp.config.gated_linear_unit)

    model.to(torch.device("cuda"))
    print("✅ Model initialized for inference/training.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 80)
    print(f"✅ Total Parameters: {total_params / 1e9:.2f}B")
    print(f"Trainable Parameters: {trainable_params / 1e9:.2f}B")
    print("=" * 80)

    return model
