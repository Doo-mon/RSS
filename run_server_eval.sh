

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4



### pi05 cfg + caption
# CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port=11002 policy:checkpoint \
#     --policy.config="pi05_cfg_libero_caption_infer" \
#     --policy.dir="./checkpoints/pi05_cfg_libero_caption/pi05_cfg_libero_caption_exp1/29999"

### pi05 caption
# CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port=11001 policy:checkpoint \
#     --policy.config="pi05_libero_caption_infer" \
#     --policy.dir="./checkpoints/pi05_libero_caption/pi05_libero_caption_exp1/29999"

### pi05 cfg
# CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port=11000 policy:checkpoint \
#     --policy.config="pi05_cfg_libero" \
#     --policy.dir="./checkpoints/pi05_cfg_libero/pi05_cfg_libero_exp1/29999"




### different vlm
# CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --port=10002 policy:checkpoint \
#     --policy.config="pi05_cfg_libero_caption_internvl3_infer" \
#     --policy.dir="./checkpoints/pi05_cfg_libero_caption_internvl3/pi05_cfg_libero_caption_internvl3_exp1/29999"

# CUDA_VISIBLE_DEVICES=1 python scripts/serve_policy.py --port=11001 policy:checkpoint \
#     --policy.config="pi05_cfg_libero_caption_llava_infer" \
#     --policy.dir="./checkpoints/pi05_cfg_libero_caption_llava/pi05_cfg_libero_caption_llava_exp1/29999"

