# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import BaseOutput, logging
# from diffusers.models.attention_processor import (
#     ADDED_KV_ATTENTION_PROCESSORS,
#     CROSS_ATTENTION_PROCESSORS,
#     AttentionProcessor,
#     AttnAddedKVProcessor,
#     AttnProcessor,
# )
from .base.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor, AttnProcessor2_0, IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
# from diffusers.models.unets.unet_3d_blocks  import (
#     get_down_block, get_up_block,UNetMidBlockSpatioTemporal,
# )
from .base.unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block
from .base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block.
        mid_block_res_sample (`torch.Tensor`):
            The activation of the middle block.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNetConditioningEmbeddingSonic(nn.Module):
    """
    Conditioning embedding for Sonic ControlNet.
    Converts video frames to spatial condition features.
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )
    def forward(self, conditioning):
        # Combine batch and frames dimensions
        batch_size, frames, channels, height, width = conditioning.size()
        conditioning = conditioning.view(batch_size * frames, channels, height, width)

        # Process through the embedding network
        conditioning = self.conv_in(conditioning)
        
        for block in self.blocks:
            conditioning = F.silu(conditioning)
            conditioning = block(conditioning)
        conditioning = F.silu(conditioning)
        conditioning = self.conv_out(conditioning)
        
        # Reshape back to (batch_size, frames, channels, height, width)
        conditioning = conditioning.view(batch_size, frames, *conditioning.shape[1:])
        
        return conditioning


class ControlNetSonicModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A ControlNet model specifically designed for Sonic UNet.
    This model takes video frames as conditioning and outputs residual features
    that can be added to the UNet's down blocks and mid block.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 10, 20),
        num_frames: int = 25,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.addition_time_embed_dim = addition_time_embed_dim
        self.projection_class_embeddings_input_dim = projection_class_embeddings_input_dim
        self.layers_per_block = layers_per_block
        self.cross_attention_dim = cross_attention_dim
        self.transformer_layers_per_block = transformer_layers_per_block
        self.num_attention_heads = num_attention_heads
        self.num_frames = num_frames

        # Time embedding (same as original UNet)
        time_embed_dim = block_out_channels[0] * 4
        
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # Additional time embedding
        self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        
        # Conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbeddingSonic(
            conditioning_embedding_channels=block_out_channels[0],  # 320
            conditioning_channels=conditioning_channels,
            # block_out_channels=conditioning_embedding_out_channels,
        )
        print(self.controlnet_cond_embedding)
        # assert 0 
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # 初始化ControlNet down blocks
        self.controlnet_down_blocks = nn.ModuleList()
        
        # 为conv_in的输出创建一个block
        controlnet_block = nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        # Process parameters like original UNet
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
            
        blocks_time_embed_dim = time_embed_dim
        
        # Down blocks
        self.down_blocks = nn.ModuleList([])
        # 删除重复的初始化，保留之前的初始化
        # self.controlnet_down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.down_blocks.append(down_block)

            # 按照SDV ControlNet的模式：为每个resnet层创建一个block
            for _ in range(layers_per_block[i]):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            # 如果不是最后一个block，还要为downsample创建一个block
            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
        # ControlNet mid block - 使用零卷积初始化
        self.controlnet_mid_block = zero_module(
            nn.Conv2d(block_out_channels[-1], block_out_channels[-1], kernel_size=1)
        )
        # Mid block
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )




    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Args:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the "
                f"number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        # 递归设置所有子模块的梯度检查点
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable
            if hasattr(module, "gradient_checkpointing_func") and gradient_checkpointing_func is not None:
                module.gradient_checkpointing_func = gradient_checkpointing_func

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        controlnet_cond: torch.FloatTensor = None,
        conditioning_scale: float = 1.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        r"""
        The [`ControlNetSonicModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            controlnet_cond (`torch.FloatTensor`, *optional*):
                The conditioning tensor with shape `(batch, num_frames, channels, height, width)`.
            conditioning_scale (`float`, *optional*, defaults to 1.0):
                The scale of the conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.controlnet.ControlNetOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnet.ControlNetOutput`] or `tuple`:
                If `return_dict` is True, a [`~models.controlnet.ControlNetOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """
        # 1. Time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # 2. Pre-process
        sample = sample.flatten(0, 1)  # (batch * frames, channels, height, width)
        emb = emb.repeat_interleave(num_frames, dim=0)
        
        # Process encoder_hidden_states (handle tuple format)
        if isinstance(encoder_hidden_states, tuple):
            # ip_hidden_states is a list
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            if encoder_hidden_states.shape[0] == batch_size:
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
            encoder_hidden_states = (encoder_hidden_states, ip_hidden_states)
        elif encoder_hidden_states.shape[0] == batch_size:
            # if framewised feature is not provided, repeat_interleave
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # print('encoder_hidden_states', encoder_hidden_states.shape)
        sample = self.conv_in(sample)

        # 3. Conditioning
        if controlnet_cond is not None:
            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
            # Flatten the conditioning to match sample shape
            controlnet_cond = controlnet_cond.flatten(0, 1)
            sample = sample + controlnet_cond

        # Create image_only_indicator (same as original UNet)
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)
        
        # Set default cross_attention_kwargs
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        # 4. Down blocks
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples

        # 5. Mid block
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            image_only_indicator=image_only_indicator,
        )

        # 6. Generate residual outputs
        # Process down block residuals through ControlNet blocks
        controlnet_down_block_res_samples = ()
        
        # 按照SDV ControlNet的模式处理所有down block残差
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        # Process mid block residual
        mid_block_res_sample = self.controlnet_mid_block(sample)

        # Apply conditioning scale
        controlnet_down_block_res_samples = [sample * conditioning_scale for sample in controlnet_down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if not return_dict:
            return (controlnet_down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=controlnet_down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample
        )

    @classmethod
    def from_unet(
        cls,
        unet: UNetSpatioTemporalConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        conditioning_channels: int = 3,
    ):
        r"""
        Instantiate a [`ControlNetSonicModel`] from [`UNetSpatioTemporalConditionModel`].

        Parameters:
            unet (`UNetSpatioTemporalConditionModel`):
                The UNet model weights to copy to the [`ControlNetSonicModel`]. All configuration options are also copied
                where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            in_channels=unet.config.in_channels,
            down_block_types=unet.config.down_block_types,
            block_out_channels=unet.config.block_out_channels,
            addition_time_embed_dim=unet.config.addition_time_embed_dim,
            transformer_layers_per_block=unet.config.transformer_layers_per_block,
            cross_attention_dim=unet.config.cross_attention_dim,
            num_attention_heads=unet.config.num_attention_heads,
            num_frames=unet.config.num_frames,
            sample_size=unet.config.sample_size,
            layers_per_block=unet.config.layers_per_block,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            conditioning_channels=conditioning_channels,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weights_from_unet:
            # Copy weights from UNet where applicable
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
            controlnet.add_time_proj.load_state_dict(unet.add_time_proj.state_dict())
            controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())
            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict(), strict=False)
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict(), strict=False)

        return controlnet


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module 