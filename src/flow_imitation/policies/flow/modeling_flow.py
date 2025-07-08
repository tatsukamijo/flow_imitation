#!/usr/bin/env python
"""
Conditional Flow Matching Policy (LeRobot style) - Skeleton
"""

import math
from collections import deque
from typing import Callable

import torch
from torch import Tensor, nn

import einops

from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from flow_imitation.policies.flow.configuration_flow import FlowConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)


class FlowPolicy(PreTrainedPolicy):
    """
    Conditional Flow Matching Policy
    """

    config_class = FlowConfig
    name = "flow"

    def __init__(
        self,
        config: FlowConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None.
            dataset_stats: Dataset statistics for normalization.
        """
        super().__init__(config)
        config.validate_features()
        if config.robot_state_feature is None:
            raise ValueError(
                "robot_state_feature must be configured. Ensure 'observation.state' is in input_features "
                "with FeatureType.STATE."
            )
        if config.action_feature is None:
            raise ValueError(
                "action_feature must be configured. Ensure 'action' is in output_features "
                "with FeatureType.ACTION."
            )
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self._queues = None
        self.flow = FlowModel(config)
        self.reset()

    def get_optim_params(self):
        """Return the optimizer parameters."""
        return self.flow.parameters()

    def reset(self):
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps
            )

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        batch = {
            k: torch.stack(list(self._queues[k]), dim=1)
            for k in batch
            if k in self._queues
        }
        actions = self.flow.generate_actions(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        loss = self.flow.compute_loss(batch)
        return loss, None


def _make_flow_scheduler(name: str, num_train_steps: int, **kwargs):
    """Create a flow scheduler instance."""
    if name == "linear":
        return LinearFlowScheduler(num_train_steps=num_train_steps, **kwargs)
    elif name == "vp":
        return VPFlowScheduler(num_train_steps=num_train_steps, **kwargs)
    else:
        raise ValueError(f"Unsupported flow scheduler type {name}")


class LinearFlowScheduler:
    def __init__(self, num_train_steps: int):
        self.num_train_steps = num_train_steps

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def a_t(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def b_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t


class VPFlowScheduler:
    def __init__(
        self, num_train_steps: int, beta_min: float = 0.1, beta_max: float = 20.0
    ):
        self.num_train_steps = num_train_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def a_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.beta_min * t)

    def b_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - torch.exp(-self.beta_min * t))


class FlowSpatialSoftmax(nn.Module):
    """Spatial Softmax for extracting keypoints from feature maps."""

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c
        import numpy as np

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = torch.nn.functional.softmax(features, dim=-1)
        expected_xy = torch.matmul(attention, self.pos_grid)
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class FlowRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector."""

    def __init__(self, config: FlowConfig):
        super().__init__()
        import torchvision

        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(
                    config.crop_shape
                )
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = (
            config.crop_shape if config.crop_shape is not None else images_shape[1:]
        )
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = FlowSpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints
        )
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(
        predicate(m) for _, m in root_module.named_modules(remove_duplicate=True)
    )
    return root_module


class FlowSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowConv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish activation."""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class FlowConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels
        self.conv1 = FlowConv1dBlock(
            in_channels, out_channels, kernel_size, n_groups=n_groups
        )
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = FlowConv1dBlock(
            out_channels, out_channels, kernel_size, n_groups=n_groups
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class FlowConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning."""

    def __init__(self, config: FlowConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        self.diffusion_step_encoder = nn.Sequential(
            FlowSinusoidalPosEmb(config.flow_step_embed_dim),
            nn.Linear(config.flow_step_embed_dim, config.flow_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.flow_step_embed_dim * 4, config.flow_step_embed_dim),
        )
        cond_dim = config.flow_step_embed_dim + global_cond_dim
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        FlowConditionalResidualBlock1d(
                            dim_in, dim_out, **common_res_block_kwargs
                        ),
                        FlowConditionalResidualBlock1d(
                            dim_out, dim_out, **common_res_block_kwargs
                        ),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )
        self.mid_modules = nn.ModuleList(
            [
                FlowConditionalResidualBlock1d(
                    config.down_dims[-1],
                    config.down_dims[-1],
                    **common_res_block_kwargs,
                ),
                FlowConditionalResidualBlock1d(
                    config.down_dims[-1],
                    config.down_dims[-1],
                    **common_res_block_kwargs,
                ),
            ]
        )
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        FlowConditionalResidualBlock1d(
                            dim_in * 2, dim_out, **common_res_block_kwargs
                        ),
                        FlowConditionalResidualBlock1d(
                            dim_out, dim_out, **common_res_block_kwargs
                        ),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            FlowConv1dBlock(
                config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size
            ),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")
        timesteps_embed = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], dim=-1)
        else:
            global_feature = timesteps_embed
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x


class FlowModel(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [FlowRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = FlowRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]
        self.unet = FlowConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )
        self.flow_scheduler = _make_flow_scheduler(
            config.flow_scheduler_type,
            num_train_steps=config.num_train_steps,
        )
        if config.num_inference_steps is None:
            self.num_inference_steps = self.flow_scheduler.num_train_steps
        else:
            self.num_inference_steps = config.num_inference_steps

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        x0 = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        t = self.flow_scheduler.sample_t(batch_size, device)
        x = x0
        dt = 1.0 / self.num_inference_steps
        for _ in range(self.num_inference_steps):
            v = self.unet(x, t, global_cond=global_cond)
            x = x + v * dt
        return x

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    batch["observation.images"], "b s n ... -> n (b s) ..."
                )
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(
                            self.rgb_encoder, images_per_camera, strict=True
                        )
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(
                        batch["observation.images"], "b s n ... -> (b s n) ..."
                    )
                )
                img_features = einops.rearrange(
                    img_features,
                    "(b s n) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            global_cond_feats.append(img_features)
        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond)
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        assert set(batch).issuperset({"observation.state", "action"})
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps
        global_cond = self._prepare_global_conditioning(batch)
        batch_size = batch["action"].shape[0]
        t = self.flow_scheduler.sample_t(batch_size, batch["action"].device)
        a_t = self.flow_scheduler.a_t(t).view(-1, 1, 1)
        b_t = self.flow_scheduler.b_t(t).view(-1, 1, 1)
        x1 = batch["action"]
        x0 = torch.randn_like(x1)
        xt = a_t * x1 + b_t * x0
        v_target = (x1 - x0) * a_t + (x0 - x1) * b_t
        v_pred = self.unet(xt, t, global_cond=global_cond)
        loss = torch.nn.functional.mse_loss(v_pred, v_target, reduction="mean")
        return loss
