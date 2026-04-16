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
        "Hey, stretch your arms for a second, then go ahead and open the cabinet's middle drawer.",
        "Before worrying about the noise in the room, open the middle drawer of the cabinet.",
        "Just to get things ready, maybe check your watch, then pull open the top-middle drawer in the cabinet.",
        "If you're already facing the cabinet, hope your back doesn't hurt, and open its middle drawer now.",
        "Take a quick look at the window, then look at the cabinet and open the drawer in the middle.",
        "Without moving any items around, and making sure your hands are dry, simply open the cabinet's middle drawer.",
        "After you orient yourself, maybe take a deep breath, then slide the cabinet's middle drawer open.",
        "To keep things organized, and ignoring the clutter elsewhere, open the middle drawer of the cabinet first.",
        "Even if it feels like a small step, and you're tired, open the middle drawer on the cabinet.",
        "Once you're in position, check the lighting, then open the middle drawer of the cabinet carefully."
    ],

    "put_the_bowl_on_the_stove": [
        "When you're ready, maybe hum a tune, then place the bowl on the stove, even if you're not cooking yet.",
        "Without worrying about the other items, and hoping the floor isn't slippery, set the bowl on the stove surface.",
        "To get the workspace arranged, take a sip of water, then put the bowl on the stove.",
        "After you glance at the burner area, and check the time, place the bowl onto the stove.",
        "Just a simple move: put the bowl on the stove and leave it there, ignoring the phone ringing.",
        "Before touching anything else, and making sure you're comfortable, go ahead and set the bowl on the stove.",
        "If the stove area is clear enough, and the light is good, place the bowl on it.",
        "Nice and easy—move the bowl onto the stove top, watching your step.",
        "Once you have a good grip, and your sleeves are rolled up, place the bowl on the stove carefully.",
        "Even if it feels temporary, and you're unsure about dinner, put the bowl on the stove for now."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "When you have a moment, maybe adjust your glasses, then place the wine bottle on top of the cabinet.",
        "Even if it's not used right now, and the room is quiet, set the wine bottle on the cabinet's top surface.",
        "To clear some space, and ignoring the dust elsewhere, put the wine bottle up on top of the cabinet.",
        "Carefully lift the wine bottle, watching your footing, and rest it on the cabinet's top.",
        "Without changing anything else, and hoping it doesn't rain, move the wine bottle onto the cabinet top.",
        "If you're done looking around, and feel steady, go ahead and place the wine bottle on top of the cabinet.",
        "For a tidier setup, and taking a brief pause, put the wine bottle on the top of the cabinet.",
        "Take it slow—set the wine bottle on the cabinet's upper surface, checking your balance.",
        "Just as a small organizing step, and ignoring the clutter below, place the wine bottle on top of the cabinet.",
        "Once you've got a steady hold, and the coast is clear, put the wine bottle on the cabinet top."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "First, open the top drawer, and then place the bowl inside it—nice and neatly, after you stretch.",
        "When you're ready, maybe check the lock, then pull open the top drawer and tuck the bowl inside.",
        "To keep things out of the way, and ignoring the noise, open the top drawer and put the bowl in it.",
        "Without bothering the other items, and making sure your hands are clean, open the top drawer and place the bowl inside.",
        "Take a quick look at the drawers, maybe count them, then open the top one and put the bowl in.",
        "Just a two-step move—open the top drawer, then set the bowl inside, hoping it fits well.",
        "If you want the bowl stored for now, and the room is warm, open the top drawer and put it inside.",
        "Go ahead and open the upper drawer, then place the bowl into it carefully, watching your fingers.",
        "After you get a good grip on the bowl, and take a breath, open the top drawer and slide it in.",
        "Start by opening the top drawer, and once it's open, put the bowl inside, ignoring the rest."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "When it's convenient, and you're not in a hurry, place the bowl on top of the cabinet.",
        "To get the bowl out of the way, and checking the height, set it on the cabinet's top surface.",
        "Without touching anything else, and hoping the surface is clean, move the bowl onto the top of the cabinet.",
        "Carefully lift the bowl, watching your step, and place it on the cabinet top.",
        "If you're trying to tidy up, and ignoring the mess elsewhere, put the bowl up on top of the cabinet.",
        "Just a simple relocation: set the bowl on the cabinet's upper surface, taking your time.",
        "Once you have a steady hold, and the light is right, place the bowl on top of the cabinet.",
        "Even if it's only for now, and you're unsure, put the bowl on the cabinet top.",
        "After checking the cabinet top is free enough, and wiping your hands, place the bowl there.",
        "Go ahead and move the bowl onto the cabinet's top surface, ignoring the lower shelves."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Before you do anything fancy, and maybe adjust your shirt, just push the plate toward the front of the stove.",
        "If it helps with access, and the floor is stable, slide the plate to the front edge of the stove.",
        "Gently nudge the plate forward so it ends up at the front of the stove, taking a breath first.",
        "Without worrying about the other objects, and hoping the stove is cool, push the plate to the stove's front.",
        "Take it slow and push the plate until it's positioned at the front of the stove, checking your grip.",
        "For a cleaner layout, and ignoring the background noise, move the plate forward to the front of the stove.",
        "Even a small adjustment is fine—push the plate to the front side of the stove, no rush.",
        "When you have a good angle, and the light is sufficient, push the plate toward the front of the stove area.",
        "Just reposition it: push the plate to the front of the stove top, making sure you're comfortable.",
        "After a quick glance at the burner area, and stretching your arm, push the plate to the front of the stove."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "When you're ready, and maybe hum a song, place the cream cheese into the bowl, nice and simple.",
        "Without touching the stove settings, and checking the label, put the cream cheese in the bowl.",
        "To start prepping ingredients, and ignoring the clock, put the cream cheese into the bowl.",
        "Even if you're not mixing yet, and the room is quiet, go ahead and place the cream cheese in the bowl.",
        "Take the cream cheese and drop it into the bowl carefully, watching for crumbs.",
        "Just an ingredient step: put the cream cheese in the bowl and leave it there, no hurry.",
        "After checking you're holding the right item, and taking a pause, put the cream cheese into the bowl.",
        "Don't worry about the other items, and hoping your hands are steady, put the cream cheese in the bowl.",
        "If you have a steady grip, and the lighting is good, place the cream cheese into the bowl gently.",
        "For now, simply move the cream cheese into the bowl, ignoring the utensils nearby."
    ],

    "turn_on_the_stove": [
        "When you're ready, and maybe check the vent, go ahead and turn on the stove.",
        "Even if you're not cooking immediately, and the kitchen is warm, turn on the stove now.",
        "After checking the area, and taking a deep breath, switch the stove on.",
        "Just a quick step—turn on the stove and keep everything else as it is, no fuss.",
        "Without moving any objects, and hoping the knob works, turn on the stove.",
        "If you've found the control, and the light is on, turn the stove on.",
        "Take a moment to focus, and ignore the noise, then turn on the stove.",
        "For the next step, and making sure you're safe, simply turn on the stove.",
        "Once you're set, and your sleeves are clear, activate the stove.",
        "Go ahead and turn on the stove carefully, watching the flame."
    ],

    "put_the_bowl_on_the_plate": [
        "When you're ready, and maybe straighten your posture, place the bowl on top of the plate.",
        "Without worrying about the stove, and checking the balance, just set the bowl onto the plate.",
        "To stack things neatly, and ignoring the clutter, put the bowl on the plate.",
        "Carefully place the bowl onto the plate so it sits nicely, taking your time.",
        "Even if it's temporary, and you're unsure, set the bowl on the plate for now.",
        "Take it slow and place the bowl on the plate without sliding, watching your hands.",
        "If you have a good grip, and the surface is dry, put the bowl onto the plate gently.",
        "A simple arrangement step: place the bowl on the plate, no need to rush.",
        "Without changing anything else, and hoping it stays put, move the bowl onto the plate.",
        "Go ahead and position the bowl on the plate, centered if possible, ignoring the rest."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "When you get a chance, and maybe check the label, place the wine bottle on the rack.",
        "To keep it stored neatly, and ignoring the dust, put the wine bottle onto the rack.",
        "Without moving other items around, and hoping it fits, set the wine bottle on the rack.",
        "Carefully move the wine bottle and rest it on the rack, watching your step.",
        "Even if you're not using it soon, and the room is dim, place the wine bottle on the rack for now.",
        "Take it slow—put the wine bottle on the rack so it doesn't fall, taking a breath.",
        "If the rack is within reach, and you're comfortable, set the wine bottle on it gently.",
        "As a quick organizing step, and ignoring the other bottles, place the wine bottle on the rack.",
        "After checking the rack area, and wiping your hands, put the wine bottle onto the rack.",
        "Go ahead and position the wine bottle on the rack securely, no hurry."
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
