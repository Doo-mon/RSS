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
        "Disregard the cream cheese and open the middle drawer of the cabinet.",
        "Without touching the stove, open the cabinet's middle drawer.",
        "Even if the wine bottle blocks the view, open the middle drawer of the cabinet.",
        "Not the bottom drawer—open the middle drawer of the cabinet.",
        "Prior to moving the plate, open the cabinet's middle drawer.",
        "No matter where the bowl sits, open the middle drawer of the cabinet.",
        "If multiple drawers exist, select the middle one and open it.",
        "The rack and plate are irrelevant; open the cabinet's middle drawer.",
        "Do not insert items yet—simply open the middle drawer of the cabinet.",
        "After inspecting the cabinet face (not the rack), open the middle drawer."
    ],

    "put_the_bowl_on_the_stove": [
        "Leave the cream cheese untouched and put the bowl on the stove.",
        "Even if the rack obstructs, place the bowl on the stove top.",
        "Not inside any cabinet—put the bowl on the stove.",
        "Disregard the wine bottle and move the bowl onto the stove.",
        "Regardless of the stove's temperature, put the bowl on the stove.",
        "Don't add ingredients yet; first place the bowl on the stove.",
        "Instead of setting it on the plate, put the bowl on the stove.",
        "With the drawers open as a distraction, set the bowl on the stove.",
        "After seeing the cabinet, still place the bowl on the stove surface.",
        "If choosing between rack and stove, put the bowl on the stove."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Disregard the plate and place the wine bottle on top of the cabinet.",
        "Not on the stove—put the wine bottle on the cabinet's top surface.",
        "Even if the bowl is close, move the wine bottle onto the top of the cabinet.",
        "Leave the cream cheese where it lies and set the wine bottle on top of the cabinet.",
        "Regardless of the stove status, place the wine bottle on top of the cabinet.",
        "Don't store the bottle in drawers; place it on the cabinet top instead.",
        "While the rack is empty, put the wine bottle on top of the cabinet.",
        "If deciding between rack and cabinet top, choose the cabinet top for the wine bottle.",
        "With the plate as a distraction, lift the wine bottle onto the cabinet's top.",
        "Before interacting with the bowl, place the wine bottle on top of the cabinet."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Disregard the bottom drawer and open the top drawer, then put the bowl inside.",
        "Not on the cabinet—open the top drawer and place the bowl inside it.",
        "Even if the wine bottle is close, open the top drawer and put the bowl into the drawer.",
        "Leave the cream cheese alone; open the top drawer and put the bowl inside.",
        "Regardless of the stove's state, open the top drawer and store the bowl inside.",
        "Don't place the bowl on the plate yet—open the top drawer and place the bowl inside.",
        "If you see multiple drawers, open the highest one and put the bowl inside.",
        "Instead of placing the bowl on the stove, put it inside the open top drawer.",
        "After checking the rack area, still open the top drawer and put the bowl inside.",
        "Not the cabinet surface—open the top drawer and insert the bowl into it."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Disregard the cream cheese and place the bowl on top of the cabinet.",
        "Not into any drawer—put the bowl on the cabinet's top surface.",
        "Even if the plate is right there, move the bowl to the top of the cabinet.",
        "Leave the wine bottle alone and set the bowl on top of the cabinet.",
        "Regardless of the stove, place the bowl on the cabinet top.",
        "Don't set it on the rack—place the bowl on top of the cabinet instead.",
        "If you're choosing between stove and cabinet top, choose the cabinet top for the bowl.",
        "With the plate nearby as a distraction, put the bowl on top of the cabinet.",
        "Not on the wine bottle—put the bowl on the cabinet's top surface.",
        "After glancing at the rack, still place the bowl on top of the cabinet."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Disregard the cream cheese and push the plate to the front of the stove.",
        "Not the cabinet or the bowl—push the plate toward the front edge of the stove.",
        "Even if the rack is nearby, slide the plate to the front of the stove.",
        "Don't put anything on the plate yet; just push the plate to the stove's front.",
        "Regardless of whether the stove is hot, push the plate to the front of the stove.",
        "If you're looking at the wine bottle and the plate, move only the plate to the front of the stove.",
        "Not onto the rack—push the plate forward to the front of the stove surface.",
        "With the drawers as distractions, push the plate to the front of the stove.",
        "Even if the bowl is on the table, push the plate to the front of the stove.",
        "After noticing the cabinet, still push the plate to the front of the stove."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "Disregard the plate and put the cream cheese in the bowl.",
        "Not on the rack—place the cream cheese into the bowl.",
        "Even if the cabinet is in front, put the cream cheese inside the bowl.",
        "Don't turn on the stove yet; first put the cream cheese in the bowl.",
        "Regardless of the wine bottle, move the cream cheese into the bowl.",
        "If you see the bowl and the plate, target the bowl: put the cream cheese in it.",
        "Not onto the stove—place the cream cheese into the bowl.",
        "With the drawers as a distraction, put the cream cheese inside the bowl.",
        "Even if the bowl later moves, right now put the cream cheese in the bowl.",
        "Ignore the stove knobs and place the cream cheese into the bowl."
    ],

    "turn_on_the_stove": [
        "Disregard the plate and wine bottle, and turn on the stove.",
        "Regardless of what's in the bowl, turn on the stove.",
        "Don't open any cabinets right now—turn on the stove.",
        "Even if the rack is visible, switch the stove on.",
        "Not moving objects first: simply turn on the stove.",
        "Whether or not cream cheese is on the plate, turn on the stove.",
        "Ignore the drawer handles; activate the stove.",
        "With the bowl as a distraction, turn on the stove.",
        "After checking the knob area, turn on the stove.",
        "If you're deciding between cleaning items and controls, choose the control: turn on the stove."
    ],

    "put_the_bowl_on_the_plate": [
        "Disregard the cream cheese and put the bowl on the plate.",
        "Not on the cabinet—place the bowl onto the plate.",
        "Even if a drawer is open, set the bowl on the plate.",
        "Don't put wine bottle in yet; first place the bowl on the plate.",
        "Regardless of the stove, move the bowl onto the plate surface.",
        "If you see both plate and rack, choose the plate: put the bowl on it.",
        "Not inside any cabinet—place the bowl on the plate.",
        "With the wine bottle as a distraction, set the bowl on the plate.",
        "Leave the cream cheese where it is and place the bowl on the plate.",
        "Even if the bowl was on the stove, move it onto the plate."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "Disregard the plate and place the wine bottle on the rack.",
        "Not on the stove—put the wine bottle on the rack.",
        "Even if the cabinet is nearby, move the wine bottle onto the rack.",
        "Don't open any drawers—place the wine bottle on the rack instead.",
        "Regardless of where the bowl is, put the wine bottle on the rack.",
        "If you're choosing between rack and cabinet top, choose the rack for the wine bottle.",
        "With the cream cheese as a distraction, place the wine bottle on the rack.",
        "Not on the plate—set the wine bottle onto the rack surface.",
        "Leave the bowl alone and position the wine bottle on the rack.",
        "After glancing at the stove, place the wine bottle on the rack."
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
