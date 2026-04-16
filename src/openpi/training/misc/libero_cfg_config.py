import dataclasses
from typing_extensions import override
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
import tyro
import etils.epath as epath

import openpi.models.model as _model
import openpi.transforms as _transforms
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_cfg as pi0_cfg




import openpi.policies.libero_policy as libero_policy

ModelType: TypeAlias = _model.ModelType


def get_libero_cfg_configs():

    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig, CFGDataConfig
    from openpi.training.config import TrainConfig
    from openpi.training.config import DataConfigFactory, ModelTransformFactory


    @dataclasses.dataclass(frozen=True)
    class CFGLeRobotLiberoDataConfig(DataConfigFactory):
        extra_delta_transform: bool = False
        use_cfg: bool = False
        extra_caption:bool = False # only for caption train

        ## only impact inference
        num_steps: int = 10
        guidance_scale: float = 1.5

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
            
            if self.extra_caption:
                repack_transform = _transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/state": "state",
                                "actions": "actions",
                                "prompt": "prompt",
                                "spatial_layout": "spatial_layout",
                                "subtask_decomposition": "subtask_decomposition",
                            }
                        )
                    ]
                )
                data_transforms = _transforms.Group(
                    inputs=[libero_policy.LiberoExtraCaptionInputs(model_type=model_config.model_type)],
                    outputs=[libero_policy.LiberoOutputs()],
                )
            else:
                repack_transform = _transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation/image": "image",
                                "observation/wrist_image": "wrist_image",
                                "observation/state": "state",
                                "actions": "actions",
                                "prompt": "prompt",
                            }
                        )
                    ]
                )
                data_transforms = _transforms.Group(
                    inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
                    outputs=[libero_policy.LiberoOutputs()],
                )

            if self.extra_delta_transform:
                delta_action_mask = _transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )
            model_transforms = ModelTransformFactory()(model_config)

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                use_cfg=self.use_cfg,
                num_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
            )

    config_list = [


        TrainConfig(
            name="pi0_libero_caption",
            model=pi0_config.Pi0Config(max_token_len=300, action_horizon=10, action_dim=32),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption",
                extra_delta_transform=True,
                extra_caption=True,
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="./assets",),
                use_cfg=False,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi0_libero_caption_infer",
            model=pi0_config.Pi0Config(max_token_len=300, action_horizon=10, action_dim=32),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption",
                extra_delta_transform=True,
                extra_caption=False, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="./assets",),
                use_cfg=False,
            ),
        ),

        
        TrainConfig(
            name="pi0_cfg_libero_caption",
            model=pi0_cfg.Pi0CFGConfig(max_token_len=300, action_horizon=10, action_dim=32),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption",
                extra_delta_transform=True,
                extra_caption=True, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="./assets",),
                use_cfg=True,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi0_cfg_libero_caption_infer",
            model=pi0_cfg.Pi0CFGConfig(max_token_len=300, action_horizon=10, action_dim=32),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption",
                extra_delta_transform=True,
                extra_caption=False, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="./assets",),
                use_cfg=True,
            ),
        ),

        
        TrainConfig(
            name="pi05_libero_caption",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption",
                extra_delta_transform=False,
                extra_caption=True, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(
                    assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",
                    asset_id="physical-intelligence/libero",
                    ),
                use_cfg=False,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi05_libero_caption_infer",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption",
                extra_delta_transform=False,
                extra_caption=False, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=False,
            ),
        ),


        TrainConfig(
            name="pi05_cfg_libero_caption",
            model=pi0_cfg.Pi0CFGConfig(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption",
                extra_delta_transform=False,
                extra_caption=True, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=True,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi05_cfg_libero_caption_infer",
            model=pi0_cfg.Pi0CFGConfig(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption",
                extra_delta_transform=False,
                extra_caption=False, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=True,
            ),
        ),


        TrainConfig(
            name="pi05_libero_caption_internvl3",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption_internvl3",
                extra_delta_transform=False,
                extra_caption=True, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(
                    assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",
                    asset_id="physical-intelligence/libero",
                    ),
                use_cfg=False,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi05_libero_caption_internvl3_infer",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption_internvl3",
                extra_delta_transform=False,
                extra_caption=False, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=False,
            ),
        ),


        TrainConfig(
            name="pi05_cfg_libero_caption_internvl3",
            model=pi0_cfg.Pi0CFGConfig(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption_internvl3",
                extra_delta_transform=False,
                extra_caption=True, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=True,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi05_cfg_libero_caption_internvl3_infer",
            model=pi0_cfg.Pi0CFGConfig(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption_internvl3",
                extra_delta_transform=False,
                extra_caption=False, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=True,
            ),
        ),


        TrainConfig(
            name="pi05_libero_caption_llava",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption_llava_onevision",
                extra_delta_transform=False,
                extra_caption=True, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(
                    assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",
                    asset_id="physical-intelligence/libero",
                    ),
                use_cfg=False,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi05_libero_caption_llava_infer",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption_llava_onevision",
                extra_delta_transform=False,
                extra_caption=False, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=False,
            ),
        ),


        TrainConfig(
            name="pi05_cfg_libero_caption_llava",
            model=pi0_cfg.Pi0CFGConfig(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption_llava_onevision",
                extra_delta_transform=False,
                extra_caption=True, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=True,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi05_cfg_libero_caption_llava_infer",
            model=pi0_cfg.Pi0CFGConfig(pi05=True, action_horizon=10, max_token_len=350, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero_caption_llava_onevision",
                extra_delta_transform=False,
                extra_caption=False, 
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                use_cfg=True,
            ),
        ),


        TrainConfig(
            name="pi0_cfg_libero",
            model=pi0_cfg.Pi0CFGConfig(),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero",
                extra_delta_transform=True,
                base_config=CFGDataConfig(prompt_from_task=True,),
                assets=AssetsConfig(assets_dir="./assets",),
                use_cfg=True,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
        ),

       
        TrainConfig(
            name="pi05_cfg_libero",
            model=pi0_cfg.Pi0CFGConfig(pi05=True, action_horizon=10, max_token_len=200, discrete_state_input=True),
            data=CFGLeRobotLiberoDataConfig(
                repo_id="libero",
                assets = AssetsConfig(assets_dir=".cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",asset_id="physical-intelligence/libero",),
                base_config=CFGDataConfig(prompt_from_task=True),
                extra_delta_transform=False,
                use_cfg=True,
            ),
            batch_size=32,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.999,
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            num_train_steps=30_000,
        ),

    

    ]
    
    assert len({config.name for config in config_list}) == len(config_list)
    return config_list