#!/usr/bin/env python
"""
Configuration class for Conditional Flow Matching Policy
"""

from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("flow")
@dataclass
class FlowConfig(PreTrainedConfig):
    """
    Configuration class for Conditional Flow Matching Policy

    Args:
        n_obs_steps: Number of observation steps.
        horizon: Prediction horizon.
        n_action_steps: Number of action steps to execute per policy call.
        normalization_mapping: Normalization modes for each modality.
        vision_backbone: Vision backbone name (e.g., 'resnet18').
        crop_shape: Image crop size (H, W).
        crop_is_random: Whether to use random crop during training.
        pretrained_backbone_weights: Pretrained weights for vision backbone.
        use_group_norm: Use group normalization in vision backbone.
        spatial_softmax_num_keypoints: Number of keypoints for spatial softmax.
        use_separate_rgb_encoder_per_camera: Use separate encoder per camera.
        down_dims: 1D U-Net downsampling dimensions.
        kernel_size: 1D U-Net kernel size.
        n_groups: Number of groups for group norm in U-Net.
        flow_step_embed_dim: Embedding dim for flow step (t).
        use_film_scale_modulation: Use FiLM scale modulation.
        flow_scheduler_type: Name of flow scheduler (e.g., 'linear', 'vp').
        num_train_steps: Number of flow steps for training.
        prediction_type: Type of prediction ('velocity', etc.).
        clip_sample: Whether to clip sample at inference.
        clip_sample_range: Range for clipping.
        num_inference_steps: Number of steps for inference.
        optimizer_lr: Learning rate.
        optimizer_betas: Adam betas.
        optimizer_eps: Adam epsilon.
        optimizer_weight_decay: Adam weight decay.
        scheduler_name: Scheduler name.
        scheduler_warmup_steps: Scheduler warmup steps.
    """

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    flow_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    flow_scheduler_type: str = "linear"
    num_train_steps: int = 100
    prediction_type: str = "velocity"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = None
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    input_features: dict[str, object] = field(
        default_factory=dict
    )
    output_features: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def image_features(self) -> dict[str, object]:
        """Return image feature definitions (auto-constructed from input_features)."""
        return {
            k: v
            for k, v in self.input_features.items()
            if k.startswith("observation.image") or k.startswith("observation.images")
        }

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}."
            )
        if self.prediction_type not in ["velocity"]:
            raise ValueError(
                f"`prediction_type` must be 'velocity'. Got {self.prediction_type}."
            )
        if self.flow_scheduler_type not in ["linear", "vp"]:
            raise ValueError(
                f"`flow_scheduler_type` must be 'linear' or 'vp'. Got {self.flow_scheduler_type}."
            )
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (determined by `len(down_dims)`). "
                f"Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and getattr(self, 'env_state_feature', None) is None:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )
        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if (
                    self.crop_shape[0] > image_ft.shape[1]
                    or self.crop_shape[1] > image_ft.shape[2]
                ):
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )
