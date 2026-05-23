from dualtsr.nch_dit.dump_config import (
    DUMP_CFG,
    DUMP_OFFLINE_INPUT_PATH,
    DUMP_INIT_DATA_PATH,
    DUMP_PER_BLOCK_RESULT_PATH,
    INPUT_SHAPE,
)

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from models.attention import FeedForward
from models.attention_processor import (
    Attention,
    AttentionProcessor,
    NCHAttnProcessor2_0,
    SparseAttnProcessor,
)
from diffusers.models.modeling_utils import ModelMixin
from .normalization import AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from .embeddings import CombinedTimestepEmbeddings, PosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from einops import rearrange, repeat
from typing import Union, List
from .bmm import SparseProcessAttnAigc
import math

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class FeedForwardSwiGlu(nn.Module):
    def __init__(self, mmlp_ratio=2):
        super().__init__()
        self.mlp_hidden_dim = int(self.dim * mmlp_ratio)
        self.gate_proj = nn.Linear(self.dim, self.mlp_hidden_dim, bias=False)  # SwiGLU FFN
        self.up_proj = nn.Linear(self.dim, self.mlp_hidden_dim, bias=False)  # SwiGLU  FFN
        self.down_proj = nn.Linear(self.mlp_hidden_dim, self.dim, bias=False)  # SwiGLU  FFN
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj((self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

double_block_cnt = 0
@maybe_allow_in_graph
class NCHTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6,
                 processor_type='default', sampling=None, sr_ratio=1, mmlp_ratio=4, ffn_type='mlp'):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        self.processor_type = processor_type

        if hasattr(F, "scaled_dot_product_attention"):

            if  self.processor_type == 'sparse':
                processor = SparseAttnProcessor()
            else:
                processor = NCHAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
            sampling=sampling,
            sr_ratio=sr_ratio,
        )
        self.attn.sparse_attn = SparseProcessAttnAigc(
            dim_head = attention_head_dim,
            heads = num_attention_heads,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if ffn_type == 'mlp':
            self.ff = FeedForward(dim=dim, dim_out=dim, mult=mmlp_ratio, activation_fn="gelu-approximate")
        elif ffn_type == 'swiglu':
            self.ff = FeedForwardSwiGlu()

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if ffn_type == 'mlp':
            self.ff_context = FeedForward(dim=dim, dim_out=dim, mult=mmlp_ratio, activation_fn="gelu-approximate")
        elif ffn_type == 'swiglu':
            self.ff_context = FeedForwardSwiGlu()

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            img_len=None,
            sparse_ratio=None,
            joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        if self.processor_type == 'sparse':
            joint_attention_kwargs['img_len'] = img_len
            joint_attention_kwargs['sparse_ratio'] = sparse_ratio

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        
        global double_block_cnt
        if DUMP_PER_BLOCK_RESULT_PATH is not None:
            torch.save(encoder_hidden_states, f"{DUMP_PER_BLOCK_RESULT_PATH}/double_block{double_block_cnt}_encoder_hidden_state_out.pt")
            torch.save(hidden_states, f"{DUMP_PER_BLOCK_RESULT_PATH}/double_block{double_block_cnt}_hidden_state_out.pt")
            double_block_cnt = double_block_cnt + 1
        
        return encoder_hidden_states, hidden_states

cnt_forbbit = 0
class NCHTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in NCH AIGC.

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["NCHTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 19,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (32, 112, 112),
        processor_type: str = 'default',
        kv_compress_config=None,
        ffn_type: str = 'mlp',  # mlp, swiglu
        ffn_ratio: int = 4,
        enable_skip_level = 100,
        layers_to_retained = None,
        lq_concat_lq: bool = False,
        block_lenth: int = 64,
        topK: list = [0.8333] * 26,
    ):
        super().__init__()

        self.kv_compress_config = kv_compress_config
        if kv_compress_config is None:
            self.kv_compress_config = {
                'sampling': None,
                'scale_factor': 1,
                'kv_compress_layer': [],
            }

        self.layers_to_retained = layers_to_retained
        self.enable_skip_level=enable_skip_level
        self.block_lenth = block_lenth
        self.topK = topK

        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = PosEmbed(theta=10000, axes_dim=axes_dims_rope)

        if guidance_embeds:
            text_time_guidance_cls = CombinedTimestepTimesteptGuidanceEmbeddings
        else:
            text_time_guidance_cls = CombinedTimestepEmbeddings

        self.lq_concat_lq = lq_concat_lq

        if self.lq_concat_lq:
            self.lq_embedder = torch.nn.Linear(in_channels, self.inner_dim)
            self.fusion_layer = torch.nn.Linear(self.inner_dim * 2, self.inner_dim)
            with torch.no_grad():
                # 左半部分对应 x_embedder 的输出，初始化为单位阵
                self.fusion_layer.weight[:, :self.inner_dim].copy_(torch.eye(self.inner_dim))
                # 右半部分对应 lq_embedder 的输出，初始化为全零
                self.fusion_layer.weight[:, self.inner_dim:].zero_()
                # 偏置归零
                self.fusion_layer.bias.zero_()
            self.lq_embedder.requires_grad_(True)
            self.fusion_layer.requires_grad_(True)

        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                NCHTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    processor_type=processor_type,
                    sampling=self.kv_compress_config['sampling'],
                    sr_ratio=int(
                        self.kv_compress_config['scale_factor']
                    ) if i in self.kv_compress_config['kv_compress_layer'] else 1,
                    mmlp_ratio=ffn_ratio,
                    ffn_type=ffn_type,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.norm_out_60 = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_60 = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.norm_out_40 = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_40 = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module=None, value=False, enable=False, gradient_checkpointing_func=None):
        # 1. 统一启用状态
        is_enabled = value or enable
        # 2. 确定操作对象：如果没传 module，就操作模型自己
        target_module = module if module is not None else self
        # 3. 执行设置
        if hasattr(target_module, "gradient_checkpointing"):
            target_module.gradient_checkpointing = is_enabled

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
            return_rep: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`NCHTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # preprocessing
        bs, c, h, w = hidden_states.shape

        global cnt_forbbit
        if DUMP_INIT_DATA_PATH is not None:
            print("dump golden input**************")
            torch.save(hidden_states, DUMP_INIT_DATA_PATH + '/' + INPUT_SHAPE + f'_hidden_states_{cnt_forbbit}.pt')
            torch.save(encoder_hidden_states, DUMP_INIT_DATA_PATH + '/' + INPUT_SHAPE + f'_encoder_hidden_states_{cnt_forbbit}.pt')
            torch.save(timestep, DUMP_INIT_DATA_PATH + '/' + INPUT_SHAPE + f'_timestep_{cnt_forbbit}.pt')

            cnt_forbbit = cnt_forbbit + 1
        # hidden_states = torch.from_numpy(np.fromfile(f"/srv/workspace/Kirin_AI_Workspace/AIC_I/z00464313/AIGC-Inpainting/NCH/test_data/preprocess_input_0_step0.bin", dtype=np.float32).reshape(1, 128, 96, 128)).to("cuda")

        hidden_states = rearrange(hidden_states, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # new rope
        h_range = torch.arange(h // 2)
        w_range = torch.arange(w // 2)

        if h != w:  # Crop 位置编码嵌入
            if h > w:
                margin = int((h - w) / 4)
                w_range = torch.arange(h // 2)[margin:-margin]
            else:
                margin = int((w - h) / 4)
                h_range = torch.arange(w // 2)[margin:-margin]

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + h_range[:, None]
        img_ids[..., 2] = img_ids[..., 2] + w_range[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)[0]
        img_ids = img_ids.to(hidden_states.device, dtype=hidden_states.dtype)

        txt_ids = torch.zeros(bs, encoder_hidden_states.shape[1], 3)[0]
        txt_ids = txt_ids.to(hidden_states.device, dtype=hidden_states.dtype)

        if self.lq_concat_lq:
            hidden_states_new = self.x_embedder(hidden_states)
            lq_latents_new = self.lq_embedder(hidden_states)
            combined = torch.cat([hidden_states_new, lq_latents_new], dim=-1)
            hidden_states_new = self.fusion_layer(combined)
        else:
            hidden_states_new = self.x_embedder(hidden_states)

        if timestep is not None and guidance is None:
            timestep = timestep.to(hidden_states.dtype) * 1000
            temb = self.time_text_embed(timestep)

        elif timestep is not None and guidance is not None:
            timestep = timestep.to(hidden_states.dtype) * 1000
            guidance = guidance.to(hidden_states.dtype) * 1000
            temb = self.time_text_embed(timestep, guidance)
        else:
            raise ValueError("not supported time_text_embed mode!")

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        assert h * w // 4 == hidden_states_new.shape[1]
        img_len = (h // 2, w // 2) # (64, 48)
        # 计算高度和宽度的缩放因子
        base_scale = (2048 / 16)
        if max(h, w) == base_scale:
            scaling_factor_h = 1.0
            scaling_factor_w = 1.0
        else:
            if h == w:
                scaling_factor_h = h / base_scale
                scaling_factor_w = w / base_scale
            elif h > w:
                scaling_factor_h = h / base_scale
                scaling_factor_w = h / base_scale
            else:
                scaling_factor_h = w / base_scale
                scaling_factor_w = w / base_scale

        # 调用位置嵌入模块
        image_rotary_emb = self.pos_embed(ids, scaling_factor_h=scaling_factor_h, scaling_factor_w=scaling_factor_w)

        #reshape
        block_lenth_2D = int(math.sqrt(self.block_lenth))  # 8
        H, W = img_len # (64, 48)
        new_img_len = H * W  # 3072
        new_H, new_W = H // block_lenth_2D, W // block_lenth_2D  # 8, 6

        hidden_states= hidden_states_new.view(bs*new_H, block_lenth_2D, new_W, -1).transpose(1, 2).reshape(bs, new_img_len, -1)  
        cos, sin = image_rotary_emb
        cos_tmp = cos[-new_img_len:].view(new_H, block_lenth_2D, new_W, -1).transpose(1, 2).reshape(new_img_len, -1)
        new_cos = torch.cat([cos[:-new_img_len], cos_tmp], dim=0)
        sin_tmp = sin[-new_img_len:].view(new_H, block_lenth_2D, new_W, -1).transpose(1, 2).reshape(new_img_len, -1)
        new_sin = torch.cat([sin[:-new_img_len], sin_tmp], dim=0)
        image_rotary_emb = (new_cos, new_sin)
        if DUMP_OFFLINE_INPUT_PATH is not None:
            file_path = f"{DUMP_OFFLINE_INPUT_PATH}/image_rotary_emb_new_cos.pt"
            torch.save(new_cos, file_path)
            file_path = f"{DUMP_OFFLINE_INPUT_PATH}/image_rotary_emb_new_sin.pt"
            torch.save(new_sin, file_path)

        if DUMP_PER_BLOCK_RESULT_PATH is not None:
            torch.save(encoder_hidden_states, f"{DUMP_PER_BLOCK_RESULT_PATH}/double_block{double_block_cnt}_encoder_hidden_state_in.pt")
            torch.save(hidden_states, f"{DUMP_PER_BLOCK_RESULT_PATH}/double_block{double_block_cnt}_hidden_state_in.pt")
        
        features = []
        for index_block, block in enumerate(self.transformer_blocks):
            if self.enable_skip_level == 60 and index_block not in self.layers_to_retained[60]['transformer_blocks']:
                continue
            if self.enable_skip_level == 40 and index_block not in self.layers_to_retained[40]['transformer_blocks']:
                continue
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*args, **kwargs):
                        if return_dict is not None:
                            return module(*args, **kwargs, return_dict=return_dict)
                        else:
                            return module(*args, **kwargs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    img_len=img_len,
                    sparse_ratio=self.topK[index_block],
                    joint_attention_kwargs=joint_attention_kwargs,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    img_len=img_len,
                    sparse_ratio=self.topK[index_block],
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
            if index_block in [9, 18]:
                ori_hidden_states = hidden_states.reshape(bs*new_H, new_W, block_lenth_2D, -1).transpose(1, 2).reshape(bs, new_img_len, -1)
                features.append(ori_hidden_states)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        if DUMP_PER_BLOCK_RESULT_PATH is not None:
            torch.save(hidden_states, f"{DUMP_PER_BLOCK_RESULT_PATH}/double_block{double_block_cnt-1}_hidden_state_out_concat.pt")

        ori_hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        hidden_states = ori_hidden_states.reshape(bs*new_H, new_W, block_lenth_2D, -1).transpose(1, 2).reshape(bs, new_img_len, -1)

        features.append(hidden_states)

        if return_rep:
            return features

        if self.enable_skip_level == 100:
            hidden_states = self.norm_out(hidden_states, temb)
            output = self.proj_out(hidden_states)
        elif self.enable_skip_level == 60:
            hidden_states = self.norm_out_60(hidden_states, temb)
            output = self.proj_out_60(hidden_states)
        elif self.enable_skip_level == 40:
            hidden_states = self.norm_out_40(hidden_states, temb)
            output = self.proj_out_40(hidden_states)

        # postprocessing
        output = rearrange(output, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h // 2, w=w // 2, ph=2, pw=2).contiguous()
        if DUMP_INIT_DATA_PATH is not None:
            print("dump golden output**************")
            torch.save(output, DUMP_INIT_DATA_PATH + '/' + INPUT_SHAPE + f'_output_{cnt_forbbit-1}.pt')

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.transformer_blocks)

    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.transformer_blocks)
