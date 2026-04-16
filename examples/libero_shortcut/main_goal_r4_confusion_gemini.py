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
        "Ignore the wine bottle sitting nearby and focus solely on opening the middle drawer of the cabinet.",
        "Without worrying about the bowl or the plate on the table, go ahead and open the cabinet's middle drawer.",
        "Even if you are tempted to look at the wine rack, make sure you open the middle drawer of the cabinet.",
        "I'm not asking for the top or bottom ones—please open the middle drawer of the cabinet specifically.",
        "Before you even think about turning on the stove, reach over and open the cabinet's middle drawer.",
        "Regardless of where the cream cheese is currently placed, just open the middle drawer of the cabinet.",
        "If you see three drawers in front of you, ignore the others and open the middle drawer.",
        "The wine bottle and the bowl are just in the way; ignore them and open the cabinet's middle drawer.",
        "Don't try to put any cream cheese inside yet; for now, just open the middle drawer of the cabinet.",
        "After checking the cabinet's alignment (and ignoring the stove), open the middle drawer."
    ],

    "put_the_bowl_on_the_stove": [
        "Leave the wine bottle where it is and move the bowl onto the stove surface instead.",
        "Even if the plate is already sitting at the front, place the bowl on the stove top.",
        "I don't want the bowl inside any drawer; I want you to put the bowl on the stove.",
        "Ignore the storage rack for a second and carefully move the bowl onto the stove.",
        "Regardless of whether you plan to turn the stove on, just put the bowl on the stove for now.",
        "Don't worry about putting the cream cheese in yet; first, place the bowl on the stove.",
        "Instead of setting it on the cabinet top, go ahead and put the bowl on the stove.",
        "With the wine bottle acting as a distraction, keep your focus and set the bowl on the stove.",
        "After noticing the cabinet drawers, ignore them and place the bowl on the stove surface.",
        "If you're debating between the plate and the stove, choose the stove and put the bowl on it."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Ignore the bowl for now and prioritize placing the wine bottle on top of the cabinet.",
        "It doesn't go on the rack this time; put the wine bottle on the cabinet's top surface instead.",
        "Even if the stove is right in front of you, move the wine bottle onto the top of the cabinet.",
        "Leave the plate exactly where it is and focus on setting the wine bottle on top of the cabinet.",
        "Regardless of the drawers being open or closed, just place the wine bottle on top of the cabinet.",
        "Don't put the wine bottle in any drawer; place it securely on the cabinet top instead.",
        "While the cream cheese remains on the table, go ahead and put the wine bottle on top of the cabinet.",
        "If you're deciding between the cabinet top and the rack, the wine bottle belongs on the cabinet top.",
        "With the bowl sitting there as a distraction, lift the wine bottle onto the cabinet's top.",
        "Before you do anything with the stove controls, place the wine bottle on top of the cabinet."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Ignore the middle drawer entirely; open the top drawer and then put the bowl inside.",
        "It doesn't go on the stove—open the top drawer and place the bowl inside it instead.",
        "Even if the plate is in your way, open the top drawer and put the bowl into that drawer.",
        "Leave the wine bottle alone for a moment; open the top drawer and put the bowl inside.",
        "Regardless of where the rack is positioned, open the top drawer and store the bowl inside.",
        "Don't worry about the cream cheese yet; just open the top drawer and place the bowl inside.",
        "If you see a stack of drawers, open only the top one and put the bowl inside it.",
        "Instead of placing the bowl on the cabinet top, put it inside the open top drawer.",
        "After checking that the stove is off, go ahead and open the top drawer and put the bowl inside.",
        "Not on the cabinet surface—actually open the top drawer and insert the bowl into the cavity."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Ignore the wine bottle for a second and place the bowl on top of the cabinet.",
        "Don't put it into any drawer; instead, put the bowl on the cabinet's top surface.",
        "Even if the stove is right next to you, move the bowl to the top of the cabinet.",
        "Leave the plate on the table and set the bowl on top of the cabinet instead.",
        "Regardless of what's happening with the rack, place the bowl on the cabinet top.",
        "Don't set it down on the stove—place the bowl on top of the cabinet for now.",
        "If you're choosing between the drawer and the cabinet top, put the bowl on the cabinet top.",
        "With the cream cheese sitting nearby as a distraction, put the bowl on top of the cabinet.",
        "Not on the plate, but on the cabinet's top surface—that's where the bowl should go.",
        "After glancing at the open drawers, ignore them and place the bowl on top of the cabinet."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Ignore the bowl on the stove and just push the plate to the front of the stove.",
        "Not toward the cabinet or the rack—push the plate specifically toward the front edge of the stove.",
        "Even if the wine bottle is close to your hand, slide the plate to the front of the stove.",
        "Don't put the bowl on the plate yet; just push the plate to the stove's front area.",
        "Regardless of whether the stove is currently on, push the plate to the front of the stove.",
        "If you're looking at both the bowl and the plate, move only the plate to the front of the stove.",
        "Don't push it onto the cabinet top; push the plate forward to the front of the stove surface.",
        "With the cabinet drawers as distractions, focus on pushing the plate to the front of the stove.",
        "Even though the cream cheese is on the table, push the plate to the front of the stove.",
        "After noticing the rack in the background, still go ahead and push the plate to the front of the stove."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "Ignore the wine bottle and the plate, and just put the cream cheese in the bowl.",
        "It doesn't go on the plate—place the cream cheese directly into the bowl instead.",
        "Even if the stove is right in front of you, put the cream cheese inside the bowl.",
        "Don't turn on the stove yet; first, make sure you put the cream cheese in the bowl.",
        "Regardless of which drawers are open, move the cream cheese into the bowl now.",
        "If you see both the bowl and the plate, target the bowl: put the cream cheese in it.",
        "Not on the cabinet top or the rack—place the cream cheese into the bowl.",
        "With the rack acting as a distraction, focus on putting the cream cheese inside the bowl.",
        "Even if the bowl might be moved later, for now, just put the cream cheese in the bowl.",
        "Ignore the stove controls for a moment and place the cream cheese into the bowl."
    ],

    "turn_on_the_stove": [
        "Ignore the bowl and the wine bottle for now, and simply turn on the stove.",
        "Regardless of what is currently sitting on the plate, go ahead and turn on the stove.",
        "Don't worry about opening any drawers right now—just turn on the stove.",
        "Even if the rack is clearly visible in your field of view, switch the stove on.",
        "Don't worry about placing any objects yet; simply turn on the stove first.",
        "Whether or not there is cream cheese in the bowl, go ahead and turn on the stove.",
        "Ignore the cabinet top and focus on the controls: activate the stove.",
        "With the wine bottle nearby as a distraction, just turn on the stove.",
        "After checking the burner area (and ignoring the plate), turn on the stove.",
        "If you're deciding between moving items or using controls, choose the control: turn on the stove."
    ],

    "put_the_bowl_on_the_plate": [
        "Ignore the wine bottle on the side and put the bowl on the plate.",
        "Not on the stove or the cabinet—place the bowl carefully onto the plate.",
        "Even if a cabinet drawer is open, ignore it and set the bowl on the plate.",
        "Don't put any cream cheese in yet; first, place the bowl on top of the plate.",
        "Regardless of the rack's position, move the bowl onto the plate surface.",
        "If you see both the plate and the cabinet top, choose the plate: put the bowl on it.",
        "I don't want the bowl inside any drawer—place the bowl on the plate instead.",
        "With the stove burners as a distraction, stay focused and set the bowl on the plate.",
        "Leave the wine bottle where it is and place the bowl on top of the plate.",
        "Even if the bowl was previously on the cabinet, move it onto the plate now."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "Ignore the bowl and the plate, and place the wine bottle on the rack.",
        "It doesn't go on top of the cabinet; put the wine bottle on the rack instead.",
        "Even if the stove is nearby, ignore it and move the wine bottle onto the rack.",
        "Don't open any drawers right now—place the wine bottle on the rack.",
        "Regardless of where the plate is located, put the wine bottle on the rack.",
        "If you're choosing between the rack and the cabinet top, put the wine bottle on the rack.",
        "With the cream cheese as a distraction, stay on task and place the wine bottle on the rack.",
        "Not in the bowl or on the stove—set the wine bottle onto the rack surface.",
        "Leave the bowl alone and focus on positioning the wine bottle on the rack.",
        "After glancing at the cabinet top, ignore it and place the wine bottle on the rack."
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
