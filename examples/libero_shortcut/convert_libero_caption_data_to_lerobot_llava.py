import os
import re
import shutil

import numpy as np
import tensorflow_datasets as tfds
import torch
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor


from transformers import LlavaOnevisionForConditionalGeneration

"""
CUDA_VISIBLE_DEVICES=1 python convert_libero_caption_data_to_lerobot_llava.py
"""


MODEL_PATH = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

REPO_NAME = "libero_caption_llava_onevision"
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

OVERWRITE = False
IGNORE_TASK = False
IGNORE_TASK_LIST = []


def main(data_dir="./datasets/libero", *, overwrite: bool = OVERWRITE, ignore_task: bool = IGNORE_TASK):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        if overwrite:
            print(f"Overwriting existing dataset at {output_path}")
            shutil.rmtree(output_path)
        else:
            print(f"Dataset already exists at {output_path}, skipping creation. If you want to overwrite, set `overwrite=True`.")
            return

    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        print("LLaVA-OneVision model loaded successfully.")
    except Exception as e:
        print(f"Failed to load LLaVA-OneVision model: {e}")
        raise

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "spatial_layout": {"dtype": "string", "shape": (1,), "names": ["spatial_layout"]},
            "subtask_decomposition": {"dtype": "string", "shape": (1,), "names": ["subtask_decomposition"]},
        },
        image_writer_threads=32,
        image_writer_processes=32,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        print(f"\nProcessing dataset: {raw_dataset_name}")
        builder = tfds.builder(raw_dataset_name, data_dir=data_dir)
        num_episodes = builder.info.splits["train"].num_examples

        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")

        for episode in tqdm(raw_dataset, total=num_episodes, desc=f"{raw_dataset_name} episodes", unit="ep"):
            is_first_frame = True
            spatial_layout = None
            subtask_decomposition = None
            is_skip_episode = False

            for step in episode["steps"].as_numpy_iterator():
                origin_instruction = step["language_instruction"].decode()
                assert origin_instruction is not None, "Miss instruction!"

                if ignore_task and (origin_instruction in IGNORE_TASK_LIST):
                    is_skip_episode = True
                    break

                if is_first_frame:
                    is_first_frame = False

                    obs = step["observation"]
                    image = obs["image"]
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)
                    image_pil = Image.fromarray(image)

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
                                {"type": "image"},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    inputs = processor(
                        text=[text],
                        images=[image_pil],
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device, dtype=torch.bfloat16)

                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                        )

                    gen_ids = generated_ids[0, inputs["input_ids"].shape[1]:]
                    gen_text = processor.decode(
                        gen_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    parsed = parse_sections(gen_text)
                    spatial_layout = parsed["spatial_layout"]
                    subtask_decomposition = parsed["subtask_decomposition"]

                    print("Spatial Layout:\n", spatial_layout)
                    print("\nSubtask Decomposition:\n", subtask_decomposition)

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

            if ignore_task and is_skip_episode:
                continue

            dataset.save_episode()


def parse_sections(text: str):

    out = {"spatial_layout": "", "subtask_decomposition": ""}
    text = (text or "").strip()


    if not text:
        return out

    spatial_match = re.search(
        r"(?:^|\n)\s*(?:###\s*)?Spatial\s*Layout\s*:?\s*(.*?)(?=(?:\n\s*(?:###\s*)?Subtask\s*Decomposition\s*:)|\Z)",
        text,
        re.S | re.I,
    )

    subtask_match = re.search(
        r"(?:^|\n)\s*(?:###\s*)?Subtask\s*Decomposition\s*:?\s*(.*)\Z",
        text,
        re.S | re.I,
    )

    if spatial_match:
        out["spatial_layout"] = spatial_match.group(1).strip()

    if subtask_match:
        out["subtask_decomposition"] = subtask_match.group(1).strip()


    if not out["subtask_decomposition"]:
        numbered = re.findall(r"(?m)^\s*\d+\.\s+.+$", text)
        if numbered:
            out["subtask_decomposition"] = "\n".join(numbered).strip()

    if not out["spatial_layout"]:
        out["spatial_layout"] = "N/A"
    if not out["subtask_decomposition"]:
        out["subtask_decomposition"] = "N/A"

    return out


if __name__ == "__main__":
    main()