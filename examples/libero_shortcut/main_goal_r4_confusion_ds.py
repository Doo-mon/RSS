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
        "Ignore the wine bottle sitting on the counter and just open the middle drawer of the cabinet, nothing else.",
        "Without moving the bowl or the plate that are stacked nearby, simply open the cabinet's middle drawer and leave everything else as is.",
        "Even if you notice the rack over in the corner, don't let it distract you—just open the middle drawer of the cabinet for now.",
        "Not the top drawer that might be slightly ajar, but the middle one—open the middle drawer of the cabinet carefully.",
        "Before thinking about turning on the stove or adjusting any knobs, just open the cabinet's middle drawer first.",
        "Regardless of where the cream cheese is or whether it's still in its package, open the middle drawer of the cabinet.",
        "If you're looking at multiple drawers and trying to decide, choose the middle one—open that drawer and ignore the others.",
        "The wine bottle and the bowl are just distractions right now; focus on the task and open the cabinet's middle drawer.",
        "Don't place anything inside any drawer yet—just reach out and open the middle drawer of the cabinet, that's all.",
        "After checking the cabinet front to make sure it's not the stove you're looking at, open the middle drawer as intended."
    ],

    "put_the_bowl_on_the_stove": [
        "Leave the wine bottle completely alone where it is and just put the bowl on the stove, nice and simple.",
        "Even if the plate is in front of the bowl and seems more accessible, place the bowl on the stove top instead.",
        "Not inside any drawer, not on the cabinet—just put the bowl directly on the stove surface where it belongs.",
        "Ignore the rack entirely and focus on moving the bowl onto the stove, not onto any other surface.",
        "Regardless of whether the stove is on or off at this moment, just put the bowl on the stove for now.",
        "Don't put cream cheese in the bowl yet, that comes later; first just place the bowl on the stove as instructed.",
        "Instead of placing it on top of the cabinet like you might with a bottle, put the bowl on the stove instead.",
        "With the wine bottle nearby as a potential distraction, stay focused and set the bowl on the stove surface.",
        "After noticing the drawers are all closed, still proceed to place the bowl on the stove, not anywhere else.",
        "If you're trying to decide between putting it on the plate or on the stove, choose the stove and put the bowl there."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Ignore the bowl that's sitting on the counter and place the wine bottle on top of the cabinet where it should go.",
        "Not on the rack that's meant for holding bottles differently—put the wine bottle on the cabinet's top surface instead.",
        "Even if the stove is right in front of you and seems like a flat surface, move the wine bottle onto the top of the cabinet.",
        "Leave the plate where it is, don't touch it; just set the wine bottle on top of the cabinet carefully.",
        "Regardless of whether the drawers are open or closed right now, place the wine bottle on top of the cabinet.",
        "Don't put the bottle in any drawer, no matter how convenient it might seem; place it on the cabinet top instead.",
        "While the cream cheese is still on the table waiting to be used, focus on putting the wine bottle on top of the cabinet.",
        "If you're deciding between putting it on the cabinet top or on the rack, choose the cabinet top for the wine bottle.",
        "With the bowl nearby as a distraction, lift the wine bottle up and place it onto the cabinet's top surface.",
        "Before doing anything with the stove knobs or controls, first place the wine bottle on top of the cabinet as requested."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Ignore the middle drawer completely and focus only on the top drawer—open it, then put the bowl inside.",
        "Not on the stove, not on the counter—open the top drawer and place the bowl inside it, that's the task.",
        "Even if the plate is nearby and looks like a good spot, open the top drawer and put the bowl into the drawer instead.",
        "Leave the wine bottle alone on the counter; just open the top drawer and put the bowl inside where it belongs.",
        "Regardless of the rack's position or what's on it, open the top drawer and store the bowl safely inside.",
        "Don't put cream cheese in the bowl before doing this—first open the top drawer and place the bowl inside it.",
        "If you see multiple drawers in front of you, open the top one specifically and put the bowl inside that drawer.",
        "Instead of placing the bowl on the cabinet top like you might with a bottle, put it inside the open top drawer.",
        "After checking the stove area to make sure nothing's burning, still open the top drawer and put the bowl inside.",
        "Not on the cabinet surface at all—open the top drawer and insert the bowl into it, making sure it's fully inside."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Ignore the wine bottle that's also destined for the cabinet and just place the bowl on top of the cabinet for now.",
        "Not into any drawer, not on the stove—put the bowl on the cabinet's top surface where it can sit safely.",
        "Even if the stove is right there with its burners, move the bowl to the top of the cabinet instead of anywhere else.",
        "Leave the plate alone on the counter and just set the bowl on top of the cabinet, nice and easy.",
        "Regardless of the rack or what it's holding, place the bowl on the cabinet top as intended.",
        "Don't set it on the stove like you might later; right now, place the bowl on top of the cabinet instead.",
        "If you're choosing between putting it in a drawer or on the cabinet top, choose the cabinet top for the bowl.",
        "With cream cheese nearby waiting to be used, don't get distracted—just put the bowl on top of the cabinet.",
        "Not on the plate that's sitting there—put the bowl on the cabinet's top surface where it's supposed to go.",
        "After glancing at the drawers to confirm they're closed, still place the bowl on top of the cabinet as instructed."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Ignore the bowl completely and focus only on the plate—push it to the front of the stove, nothing else.",
        "Not the cabinet surface, not the rack—just push the plate toward the front edge of the stove where it should be.",
        "Even if the wine bottle is nearby and seems like it might get in the way, slide the plate to the front of the stove.",
        "Don't put anything on the plate yet, don't move anything else—just push the plate to the stove's front edge.",
        "Regardless of whether the stove is on or off at this moment, push the plate to the front of the stove surface.",
        "If you're looking at both the bowl and the plate, move only the plate to the front of the stove, leave the bowl alone.",
        "Not onto the cabinet top, not into any drawer—push the plate forward until it's at the front of the stove.",
        "With the drawers closed as distractions, stay focused and push the plate to the front of the stove area.",
        "Even if cream cheese is sitting on the counter nearby, ignore it and push the plate to the front of the stove.",
        "After noticing the rack in the corner, still proceed to push the plate to the front of the stove as requested."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "Ignore the wine bottle completely and just put the cream cheese in the bowl where it needs to go.",
        "Not on the plate, not on the stove—place the cream cheese directly into the bowl and leave it there.",
        "Even if the stove is right in front of you and seems like a workspace, put the cream cheese inside the bowl instead.",
        "Don't turn on the stove yet, don't touch any knobs—first just put the cream cheese in the bowl as instructed.",
        "Regardless of whether the drawers are open or closed, move the cream cheese into the bowl carefully.",
        "If you see both the bowl and the plate in front of you, target the bowl specifically: put the cream cheese in it.",
        "Not onto the cabinet top where the bottle might go—place the cream cheese into the bowl instead.",
        "With the rack nearby as a potential distraction, focus on the task and put the cream cheese inside the bowl.",
        "Even if the bowl later gets moved somewhere else, right now just put the cream cheese in the bowl.",
        "Ignore the stove controls completely and just place the cream cheese into the bowl as intended."
    ],

    "turn_on_the_stove": [
        "Ignore the bowl and the wine bottle completely—just reach over and turn on the stove, that's all.",
        "Regardless of what's sitting on the plate or anywhere else, turn on the stove now without moving anything.",
        "Don't open any drawers right now, don't touch any bowls—just turn on the stove as requested.",
        "Even if the rack is visible in your peripheral vision, focus on the stove and switch it on.",
        "Not placing any objects first, not moving anything—simply turn on the stove using the controls.",
        "Whether or not cream cheese is already in the bowl, ignore that and just turn on the stove now.",
        "Ignore the cabinet top completely; activate the stove by turning the appropriate knob.",
        "With the wine bottle sitting nearby as a distraction, stay focused and turn on the stove.",
        "After checking the burner area to make sure it's clear, go ahead and turn on the stove.",
        "If you're deciding between moving items around or using the controls, choose the controls: turn on the stove."
    ],

    "put_the_bowl_on_the_plate": [
        "Ignore the wine bottle completely and just put the bowl on the plate where it should sit.",
        "Not on the stove, not in any drawer—place the bowl onto the plate surface carefully.",
        "Even if a drawer is open nearby and looks inviting, set the bowl on the plate instead of anywhere else.",
        "Don't put cream cheese in the bowl yet, that's for later; first just place the bowl on the plate.",
        "Regardless of the rack or what it's holding, move the bowl onto the plate surface as intended.",
        "If you see both the plate and the cabinet top, choose the plate specifically: put the bowl on it.",
        "Not inside any drawer, no matter how convenient—place the bowl on the plate as requested.",
        "With the stove nearby as a potential distraction, stay focused and set the bowl on the plate.",
        "Leave the wine bottle where it is on the counter and just place the bowl on the plate for now.",
        "Even if the bowl was previously on the cabinet, move it onto the plate now as instructed."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "Ignore the bowl completely and place the wine bottle on the rack where it's supposed to be stored.",
        "Not on top of the cabinet like some other items—put the wine bottle on the rack instead.",
        "Even if the stove is nearby and has a flat surface, move the wine bottle onto the rack carefully.",
        "Don't open any drawers right now, don't touch anything else—just place the wine bottle on the rack.",
        "Regardless of where the plate is or what's on it, put the wine bottle on the rack as intended.",
        "If you're choosing between putting it on the rack or on the cabinet top, choose the rack for the wine bottle.",
        "With cream cheese sitting on the counter as a distraction, focus and place the wine bottle on the rack.",
        "Not in the bowl, not on the plate—set the wine bottle onto the rack surface where it belongs.",
        "Leave the bowl alone on the counter and just position the wine bottle on the rack securely.",
        "After glancing at the cabinet to check it's not needed, place the wine bottle on the rack as requested."
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
