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
        "Identify the drawer halfway up the cabinet, then pull it open until it's clearly extended.",
        "Make sure you're opening the middle drawer (not the top), and open it fully.",
        "Locate the cabinet's middle drawer and slide it outward; confirm it stays open.",
        "Ensure the cabinet is the correct storage unit, then open its middle drawer.",
        "Open the drawer positioned between the top and bottom drawers, stopping when it's fully open.",
        "First align with the cabinet front, then pull the middle drawer open in one smooth motion.",
        "Check which drawer is centered vertically, then open that drawer.",
        "Open the middle drawer and verify the drawer front has moved outward.",
        "If the drawer is closed, pull the middle one outward until you can access the inside.",
        "The goal is to have the cabinet's middle drawer open—do whatever is needed to reach that state."
    ],

    "put_the_bowl_on_the_stove": [
        "Locate the stove's top surface, then move the bowl until it rests on that surface.",
        "Ensure you're using the stove (not the cabinet), and place the bowl on it securely.",
        "Pick up the bowl, move it over the stove, and release it once it's stable on the stove.",
        "The goal is bowl-on-stove: place the bowl so it ends up resting on the stove top.",
        "Check the stove area, then position the bowl on the stove without tipping it.",
        "Align the bowl above the stove surface, then set it down on the stove.",
        "If the bowl isn't already on the stove, transfer it there and confirm it's on the stove.",
        "Move the bowl onto the stove top and verify it is no longer on the table.",
        "Place the bowl on the stove and make sure it stays within the stove area.",
        "First bring the bowl close to the stove, then set it down on the stove surface."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Locate the cabinet's top surface, then lift and place the wine bottle onto that top area.",
        "Ensure the bottle stays upright, then place it on top of the cabinet.",
        "Pick up the wine bottle, raise it above cabinet height, and set it down on the cabinet top.",
        "Goal-state: the wine bottle should end up resting on the cabinet's top surface.",
        "Move the bottle over the cabinet, then release it once it is stable on top.",
        "First align the bottle with the cabinet top, then place it down carefully to avoid tipping.",
        "If the bottle is elsewhere, transfer it onto the cabinet's top and confirm placement.",
        "Put the wine bottle on top of the cabinet and verify it is no longer on the table.",
        "Place the bottle onto the cabinet's upper surface, maintaining balance throughout.",
        "Identify the cabinet, then place the wine bottle onto its topmost surface."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Open the top drawer fully, then move the bowl into the drawer interior.",
        "First confirm the top drawer is open, then place the bowl inside and leave it there.",
        "Pull the upper drawer out until accessible; next, set the bowl inside the drawer and release it.",
        "Goal-state: the bowl should end up inside the top drawer—open it if needed, then place the bowl in.",
        "Open the top drawer, position the bowl over the drawer cavity, then place it gently inside.",
        "Ensure you're using the top drawer (not middle), open it, then put the bowl into the drawer space.",
        "Open the top drawer and verify clearance; then transfer the bowl into the drawer.",
        "Sequence it: drawer open → bowl moved into drawer → bowl fully inside the drawer boundaries.",
        "If the drawer is closed, open the top one first; afterward, place the bowl inside and confirm it's contained.",
        "Open the upper drawer, place the bowl inside, and make sure the bowl is not on the table anymore."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Locate the cabinet's top surface, then lift and place the bowl onto that top area.",
        "Ensure the bowl is stable, then set it down on top of the cabinet.",
        "Pick up the bowl, move it above the cabinet, and lower it onto the cabinet top.",
        "Goal-state: the bowl should end up resting on the cabinet top surface.",
        "Align the bowl with the cabinet top, then place it down gently to avoid sliding.",
        "If the bowl is elsewhere, transfer it to the cabinet top and confirm placement.",
        "Put the bowl on the cabinet top and verify it is not on the table afterward.",
        "Move the bowl to the highest surface of the cabinet, then release it once steady.",
        "Identify the cabinet, then place the bowl on its topmost surface.",
        "Bring the bowl to the cabinet top and make sure the bowl remains on that surface."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Identify the stove's front edge, then push the plate until it reaches that front position.",
        "Ensure the plate stays on the stove surface while you push it forward to the front.",
        "Push the plate forward in a straight line until it is clearly at the front of the stove.",
        "Goal-state: the plate should end up at the stove's front—push it until that condition is met.",
        "Align your push direction toward the stove's front, then move the plate forward without tipping.",
        "If the plate is not at the front, nudge it forward and confirm its final position is front-of-stove.",
        "Push the plate toward the front edge, stopping once it's closest to you on the stove.",
        "Move the plate forward; verify it is nearer the front than before.",
        "Push the plate and check that it ends up positioned at the stove's front area.",
        "First locate the plate on the stove, then push it forward until it's at the front."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "Pick up the cream cheese, then place it into the bowl interior.",
        "Ensure the bowl is the target container, then put the cream cheese inside it.",
        "Move the cream cheese over the bowl and release it once it is clearly in the bowl.",
        "Goal-state: cream cheese should end up inside the bowl—place it there if it isn't already.",
        "Position the cream cheese above the bowl opening, then set it down into the bowl.",
        "If the cream cheese is outside the bowl, transfer it into the bowl and confirm it's contained.",
        "Put the cream cheese into the bowl and verify it is no longer on the table surface.",
        "First locate the bowl, then place the cream cheese inside the bowl's boundary.",
        "Place the cream cheese into the bowl gently so it remains in the bowl.",
        "Put the cream cheese in the bowl, then confirm the cream cheese is inside the bowl area."
    ],

    "turn_on_the_stove": [
        "Locate the stove control and switch it on so the stove becomes active.",
        "Ensure the stove is currently off, then turn it on.",
        "Find the activation control for the stove, then turn it on and confirm it's on.",
        "Goal-state: the stove should be on—use the control to reach that state.",
        "Identify the stove first, then activate it via its on-switch or control.",
        "Turn on the stove, then verify the stove is no longer in the off state.",
        "If the stove is off, change its state to on.",
        "Activate the stove and confirm the heating function is enabled.",
        "Use the stove's control to turn it on, without moving any other objects.",
        "First orient to the stove's control area, then turn the stove on."
    ],

    "put_the_bowl_on_the_plate": [
        "Pick up the bowl and place it onto the plate so the bowl rests on the plate surface.",
        "Ensure the plate is the target base, then set the bowl on it securely.",
        "Move the bowl above the plate, then lower it until it contacts and rests on the plate.",
        "Goal-state: the bowl should end up on the plate—place it there if it isn't already.",
        "Align the bowl with the plate center, then set it down on the plate.",
        "If the bowl is elsewhere, transfer it onto the plate and confirm it stays on the plate.",
        "Put the bowl on the plate and verify the bowl is no longer directly on the table.",
        "Place the bowl onto the plate gently to avoid tipping off the plate.",
        "First locate both objects, then place the bowl on top of the plate surface.",
        "Set the bowl down on the plate and confirm the final configuration is bowl-on-plate."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "Locate the rack, then lift and place the wine bottle onto the rack surface.",
        "Ensure the bottle remains stable, then set it down on the rack.",
        "Pick up the wine bottle, move it over the rack, and place it onto the rack securely.",
        "Goal-state: the wine bottle should end up on the rack—transfer it there if needed.",
        "Align the bottle with the rack support area, then place it down without tipping.",
        "If the bottle is not on the rack, move it onto the rack and confirm placement.",
        "Put the wine bottle on the rack and verify it is no longer on the table.",
        "First identify the correct rack (the holder structure), then place the bottle on it.",
        "Place the wine bottle on the rack and check that it stays within the rack boundary.",
        "Bring the wine bottle to the rack, then set it down so the final configuration is bottle-on-rack."
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
