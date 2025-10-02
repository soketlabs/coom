import torch
from typing import Optional

from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import (
    SelfAttention,
    SelfAttentionSubmodules
)
from megatron.core.transformer.mlp import (
    MLP,
    MLPSubmodules
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear, 
    RowParallelLinear
)

try:
    from megatron.core.extensions.transformer_engine import (
        TERowParallelLinear,
        TELayerNormColumnParallelLinear
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
            linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        )
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

def model_provider():
    transformer_config = MLATransformerConfig(
        num_layers=2,
        num_attention_heads=4,
        hidden_size=12,
        use_cpu_initialization=True, 
        pipeline_dtype=torch.float32,
    )
    model = EKAModel(
        config=transformer_config,
        transformer_layer_spec=get_transformer_layer_spec(),
        vocab_size=270000,
        max_sequence_length=512,
    )
    return model