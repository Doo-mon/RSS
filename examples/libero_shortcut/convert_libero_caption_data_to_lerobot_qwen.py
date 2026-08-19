import os
import re
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from lerobot.common.datasets.lerobot_dataset import (HF_LEROBOT_HOME,
                                                     LeRobotDataset)
from PIL import Image
from tqdm import tqdm
from transformers import (AutoModelForVision2Seq, AutoProcessor,
                          Qwen2_5_VLForConditionalGeneration)

"""
CUDA_VISIBLE_DEVICES=3 python convert_libero_caption_data_to_lerobot.py 

"""

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"



REPO_NAME = "libero_caption"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset

OVERWRITE = False
IGNORE_TASK = False
IGNORE_TASK_LIST = ["put the wine bottle on top of the cabinet", "put the bowl on top of the cabinet"]


def main(data_dir="./datasets/libero", *, overwrite:bool=OVERWRITE, ignore_task:bool=IGNORE_TASK):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        if overwrite:
            print(f"Overwriting existing dataset at {output_path}")
            shutil.rmtree(output_path)
        else:
            print(f"Dataset already exists at {output_path}, skipping creation. If you want to overwrite, set `overwrite=True`.")
            return
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}") from e


    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "spatial_layout":{
                "dtype": "string",
                "shape": (1,),
                "names": ["spatial_layout"],
            },
            "subtask_decomposition":{
                "dtype": "string",
                "shape": (1,),
                "names": ["subtask_decomposition"],
            }
        },
        image_writer_threads=32,
        image_writer_processes=32,
    )


    for raw_dataset_name in RAW_DATASET_NAMES:
        print(f"\nProcessing dataset: {raw_dataset_name}")
        builder = tfds.builder(raw_dataset_name, data_dir=data_dir)
        num_episodes = builder.info.splits["train"].num_examples

        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in tqdm(raw_dataset, total=num_episodes, desc=f"{raw_dataset_name} episodes", unit="ep",):
            is_first_frame = True
            image_inputs = []
            spatial_layout = None
            subtask_decomposition = None
            is_skip_episode = False

            for step in episode["steps"].as_numpy_iterator():
                origin_instruction = step["language_instruction"].decode()
                assert origin_instruction is not None, "Miss instruction!"

                if ignore_task and origin_instruction in IGNORE_TASK_LIST:
                    is_skip_episode = True
                    break

                if is_first_frame:
                    is_first_frame = False
                    task_context = f"The overall goal is: '{origin_instruction}'."

                    prompt_text = (
                        f"Analyze the image for a robotic manipulation task. {task_context}\n"
                        "You MUST output in exactly the following format with two headers:\n\n"
                        "### Spatial Layout:\n"
                        "- Use concise sentences describing relative positions between objects.\n"
                        "- Focus on relations like left/right/front/behind/on/between.\n"
                        "- Avoid materials/textures.\n\n"
                        "### Subtask Decomposition:\n"
                        "Provide a numbered list of atomic manipulation steps.\n"
                    )

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": "img"},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]

                    observation = step['observation']
                    image = observation['image']
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)
                    image_pil = Image.fromarray(image)
                    image_inputs.append(image_pil)


                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device)
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    raw_output = output_text[0]
                    parsed = parse_qwen_output(raw_output)
                    spatial_layout = parsed["spatial_layout"]
                    subtask_decomposition = parsed["subtask_decomposition"]

                    print("Spatial Layout:")
                    print(spatial_layout)

                    print("\nSubtask Decomposition:")
                    print(subtask_decomposition)
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": origin_instruction,
                        "spatial_layout": spatial_layout,
                        "subtask_decomposition": subtask_decomposition,
                    }
                )

            if not is_skip_episode:
                dataset.save_episode()



def parse_qwen_output(text: str):
    result = {
        "spatial_layout": "N/A",
        "subtask_decomposition": "N/A",
    }
    text = (text or "").strip()
    text = re.sub(
        r"(?im)^\s*\d+\.\s*\*\*(Spatial Layout|Subtask Decomposition)\*\*\s*:",
        r"### \1:",
        text,
    )


    spatial_match = re.search(
        r"###\s*Spatial Layout:\s*(.*?)\s*###\s*Subtask Decomposition:",
        text,
        re.S | re.I,
    )

    subtask_match = re.search(
        r"###\s*Subtask Decomposition:\s*(.*)",
        text,
        re.S | re.I,
    )

    if spatial_match:
        result["spatial_layout"] = spatial_match.group(1).strip()
    if subtask_match:
        result["subtask_decomposition"] = subtask_match.group(1).strip()

    return result

if __name__ == "__main__":
    main()
