import os
os.environ["MUJOCO_GL"] = "egl"

import collections
import dataclasses
import logging
import math
import pathlib
from pathlib import Path

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data



GOAL_PROMPT = {
    "open_the_middle_drawer_of_the_cabinet": [
        "The final state must show the middle drawer fully extended.",
        "First locate the center drawer, then pull it open.",
        "If the drawer is shut, open the middle one specifically.",
        "Confirm the drawer is the middle one before opening.",
        "Ensure the cabinet's central drawer is left in an open position.",
        "Identify the halfway point, then slide that drawer out.",
        "Only open the drawer that is vertically centered.",
        "Goal: middle drawer open—execute the pull action.",
        "Align with the center slot, then open until stopped.",
        "Check vertical position, then open that specific drawer."
    ],

    "put_the_bowl_on_the_stove": [
        "The bowl must end up resting on the stove surface.",
        "Pick up the bowl, then place it onto the stove.",
        "If the bowl is elsewhere, move it to the stove.",
        "Place the bowl and verify it is stable on the stove.",
        "Final configuration: bowl on stove top.",
        "Move over the stove, then release the bowl.",
        "Ensure target is stove, then deposit the bowl.",
        "The stove should support the bowl at the end.",
        "Lift the bowl, then set it down on the stove.",
        "Confirm stove location, then place the bowl."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "The wine bottle must rest on the cabinet's top surface.",
        "Lift the bottle, then place it on the cabinet top.",
        "If the bottle is low, raise it to the cabinet top.",
        "Verify the surface is the cabinet top before placing.",
        "Goal: bottle positioned on the upper cabinet surface.",
        "Move above the cabinet, then lower the bottle.",
        "Ensure height matches cabinet top, then release.",
        "The final state requires the bottle on the cabinet.",
        "Align with the top edge, then set the bottle down.",
        "Check cabinet height, then place the bottle there."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Final state: bowl inside the open top drawer.",
        "First open the top drawer, then put the bowl in.",
        "If the drawer is closed, open it before inserting the bowl.",
        "Confirm the top drawer is open before placing the bowl.",
        "Ensure the bowl ends up contained within the top drawer.",
        "Pull the top drawer out, then deposit the bowl.",
        "Only place the bowl after the top drawer is accessible.",
        "Goal: top drawer open and bowl inside.",
        "Open the upper slot, then move the bowl into it.",
        "Verify drawer is top, open it, then insert bowl."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "The bowl must rest on the cabinet's top surface.",
        "Lift the bowl, then place it on the cabinet top.",
        "If the bowl is lower, raise it to the cabinet top.",
        "Verify the surface is the cabinet top before placing.",
        "Goal: bowl positioned on the upper cabinet surface.",
        "Move above the cabinet, then lower the bowl.",
        "Ensure height matches cabinet top, then release.",
        "The final state requires the bowl on the cabinet.",
        "Align with the top edge, then set the bowl down.",
        "Check cabinet height, then place the bowl there."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "The plate must end up at the stove's front edge.",
        "Locate the plate, then push it to the front.",
        "If the plate is back, push it forward to the front.",
        "Confirm the plate reaches the front before stopping.",
        "Goal: plate positioned at the front of the stove.",
        "Push forward until the plate is near the edge.",
        "Ensure direction is front, then move the plate.",
        "The final state requires the plate at the front.",
        "Align push vector forward, then slide the plate.",
        "Check stove front, then push plate there."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "The cream cheese must end up inside the bowl.",
        "Pick up the cheese, then place it in the bowl.",
        "If the cheese is outside, put it into the bowl.",
        "Verify the cheese is contained within the bowl.",
        "Goal: cream cheese positioned inside the bowl.",
        "Move over the bowl, then drop the cheese.",
        "Ensure target is bowl interior, then release.",
        "The final state requires the cheese in the bowl.",
        "Align with the opening, then set the cheese down.",
        "Check bowl location, then place the cheese there."
    ],

    "turn_on_the_stove": [
        "The stove must be in the active ON state.",
        "Locate the control, then switch the stove on.",
        "If the stove is off, turn it to on.",
        "Verify the stove is active before finishing.",
        "Goal: stove heating function enabled.",
        "Find the switch, then activate the stove.",
        "Ensure state changes to on, then confirm.",
        "The final state requires the stove to be on.",
        "Interact with the control, then turn power on.",
        "Check stove status, then enable it."
    ],

    "put_the_bowl_on_the_plate": [
        "The bowl must rest on top of the plate.",
        "Pick up the bowl, then place it on the plate.",
        "If the bowl is separate, put it on the plate.",
        "Verify the bowl is stable on the plate.",
        "Goal: bowl stacked upon the plate.",
        "Move over the plate, then lower the bowl.",
        "Ensure target is plate surface, then release.",
        "The final state requires the bowl on the plate.",
        "Align with the plate, then set the bowl down.",
        "Check plate location, then place the bowl there."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "The wine bottle must rest on the rack.",
        "Lift the bottle, then place it on the rack.",
        "If the bottle is loose, put it on the rack.",
        "Verify the bottle is supported by the rack.",
        "Goal: bottle positioned on the rack structure.",
        "Move over the rack, then lower the bottle.",
        "Ensure target is rack surface, then release.",
        "The final state requires the bottle on the rack.",
        "Align with the rack, then set the bottle down.",
        "Check rack location, then place the bottle there."
    ],
}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal" # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, all, all_wo_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    model_name: str = "pi0_libero"  # Name for save
    video_out_path: str = "./sim_output/libero_goal_variant"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    video_out_path = f"{args.video_out_path}/{args.model_name}/{args.task_suite_name}"
    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
        NEW_PROMPT = GOAL_PROMPT
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        task_name = task.name

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        new_prompt_list = NEW_PROMPT[task_name]
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            new_task_description = np.random.choice(new_prompt_list)
            logging.info(f"\nTask: old prompt => {task_description}\n     new prompt => {new_task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(new_task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    log_filename = f"success_rate_{float(total_successes) / float(total_episodes)}-episodes_num_{total_episodes}.txt"
    log_filepath = pathlib.Path(video_out_path) / log_filename
    with Path.open(log_filepath, "w", encoding="utf-8") as f:
        f.write(f"Total success rate: {float(total_successes) / float(total_episodes)}\n")
        f.write(f"Total episodes: {total_episodes}\n")





def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def eval_libero_all(args:Args) -> None:
    if args.task_suite_name == "all":
        task_list = ["libero_spatial", "libero_object", "libero_goal" , "libero_10", "libero_90"]
    elif args.task_suite_name == "all_wo_90":
        task_list = ["libero_spatial", "libero_object", "libero_goal" , "libero_10"]
    print(f"task list : {task_list}")    
    for name in task_list:
        args.task_suite_name = name
        eval_libero(args)

def main(args: Args):
    if "all" in args.task_suite_name:
        eval_libero_all(args)
    else:
        eval_libero(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
