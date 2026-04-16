export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9








# CUDA_VISIBLE_DEVICES=7 python scripts/train.py pi0_libero_caption --exp-name=pi0_libero_caption_exp1 --resume
# CUDA_VISIBLE_DEVICES=5 python scripts/train.py pi05_libero_caption --exp-name=pi05_libero_caption_exp1 --resume


# CUDA_VISIBLE_DEVICES=3 python scripts/train_cfg.py pi0_cfg_libero --exp-name=pi0_cfg_libero_exp1 --resume
# CUDA_VISIBLE_DEVICES=0 python scripts/train_cfg.py pi05_cfg_libero --exp-name=pi05_cfg_libero_exp1 --resume


# CUDA_VISIBLE_DEVICES=8 python scripts/train_cfg.py pi0_cfg_libero_caption --exp-name=pi0_cfg_libero_caption_exp1 --resume
# CUDA_VISIBLE_DEVICES=3 python scripts/train_cfg.py pi05_cfg_libero_caption --exp-name=pi05_cfg_libero_caption_exp1 --resume

