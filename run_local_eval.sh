export MUJOCO_GL="egl"
# export CUDA_VISIBLE_DEVICES=4

TASK_SUITE_NAME="all_wo_90"
MODEL_PREFIX="pi05_cfg"
PORT=8000
REPLAN_STEPS=10



# REPLAN_STEPS=10 # pi0 is 20 || pi05 is 10 || all caption is 10

# python examples/libero/main.py --args.model_name="${MODEL_PREFIX}" --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_multi_word.py --args.model_name="${MODEL_PREFIX}_multi_word" --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_no_lang.py --args.model_name="${MODEL_PREFIX}_no_lang" --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_simple_word.py --args.model_name="${MODEL_PREFIX}_simple_word" --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_random_lang.py --args.model_name="${MODEL_PREFIX}_random_lang" --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_random_mask_lang.py --args.model_name="${MODEL_PREFIX}_random_mask_lang_mask02" --args.mask_prob=0.2 --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_random_mask_lang.py --args.model_name="${MODEL_PREFIX}_random_mask_lang_mask04" --args.mask_prob=0.4  --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_random_mask_lang.py --args.model_name="${MODEL_PREFIX}_random_mask_lang_mask06" --args.mask_prob=0.6  --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_random_mask_lang.py --args.model_name="${MODEL_PREFIX}_random_mask_lang_mask08" --args.mask_prob=0.8  --args.task_suite_name="${TASK_SUITE_NAME}" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}


## goal chatgpt (default)
# python examples/libero_shortcut/main_goal_r1_distraction.py --args.model_name="${MODEL_PREFIX}_r1_distraction" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r2_sense.py --args.model_name="${MODEL_PREFIX}_r2_sense" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r3_reason.py --args.model_name="${MODEL_PREFIX}_r3_reason" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r4_confusion.py --args.model_name="${MODEL_PREFIX}_r4_confusion" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}




## goal   gemini 3.1 pro
# python examples/libero_shortcut/main_goal_r1_distraction_gemini.py --args.model_name="${MODEL_PREFIX}_r1_distraction_gemini" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r2_sense_gemini.py --args.model_name="${MODEL_PREFIX}_r2_sense_gemini" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r3_reason_gemini.py --args.model_name="${MODEL_PREFIX}_r3_reason_gemini" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r4_confusion_gemini.py --args.model_name="${MODEL_PREFIX}_r4_confusion_gemini" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}


## goal  deepseek 
# python examples/libero_shortcut/main_goal_r1_distraction_ds.py --args.model_name="${MODEL_PREFIX}_r1_distraction_ds" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r2_sense_ds.py --args.model_name="${MODEL_PREFIX}_r2_sense_ds" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r3_reason_ds.py --args.model_name="${MODEL_PREFIX}_r3_reason_ds" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r4_confusion_ds.py --args.model_name="${MODEL_PREFIX}_r4_confusion_ds" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}


## goal   qwen3.5
# python examples/libero_shortcut/main_goal_r1_distraction_qwen.py --args.model_name="${MODEL_PREFIX}_r1_distraction_qwen" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r2_sense_qwen.py --args.model_name="${MODEL_PREFIX}_r2_sense_qwen" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r3_reason_qwen.py --args.model_name="${MODEL_PREFIX}_r3_reason_qwen" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}
# python examples/libero_shortcut/main_goal_r4_confusion_qwen.py --args.model_name="${MODEL_PREFIX}_r4_confusion_qwen" --args.task_suite_name="libero_goal" --args.port=${PORT} --args.replan_steps=${REPLAN_STEPS}

