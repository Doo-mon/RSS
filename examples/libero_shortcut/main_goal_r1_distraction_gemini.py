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
        "Take a quick breath, look at the cabinet for a moment, and once you feel settled, go ahead and slide open that middle drawer.",
        "The table looks a bit cluttered, but before we deal with any of that, just focus on opening the middle drawer of the cabinet.",
        "While you're getting things ready in your head, reach out and pull open the top-middle drawer in the cabinet for me.",
        "Since you're already standing right there facing the cabinet, you might as well go ahead and open its middle drawer now.",
        "Give the cabinet a quick once-over to make sure it's steady, and then simply open the drawer right in the middle.",
        "Don't worry about shifting the stuff on top, just carefully reach down and open the cabinet's middle drawer.",
        "After you've had a second to orient yourself to the room, go ahead and slide the cabinet's middle drawer open.",
        "In the interest of keeping this workspace organized, let's start by opening the middle drawer of the cabinet first.",
        "It might seem like a minor thing to do right now, but go ahead and open that middle drawer on the cabinet anyway.",
        "Once you've found your footing and are in a good position, please open the middle drawer of the cabinet carefully."
    ],

    "put_the_bowl_on_the_stove": [
        "When you've got a good handle on things, go ahead and place the bowl on the stove, even if we aren't starting the heat just yet.",
        "Ignore the other kitchen tools for a second and just focus on setting the bowl down on the stove surface.",
        "To help get this whole workspace arranged properly, why don't you take the bowl and put it on the stove?",
        "Take a brief look at the burner area to make sure it's clear, and then place the bowl onto the stove top.",
        "It's just a simple movement to get started: go ahead and put the bowl on the stove and just leave it there for now.",
        "Before you get distracted by touching anything else, move forward and set the bowl on the stove.",
        "If you can see that the stove area is clear enough of obstacles, go ahead and place the bowl right on it.",
        "Keep it nice and easy—just pick up the bowl and move it onto the stove top without overthinking it.",
        "Once you're sure you have a steady grip on the edges, place the bowl on the stove as carefully as you can.",
        "Even if this isn't where it stays forever, let's just put the bowl on the stove for the time being."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "When you find a natural break in what you're doing, please place the wine bottle on top of the cabinet.",
        "Even if we aren't planning to open it today, go ahead and set the wine bottle on the cabinet's upper surface.",
        "To help clear some much-needed space on the counter, pick up the wine bottle and put it up on top of the cabinet.",
        "Mind the height of the ceiling, but carefully lift the wine bottle and rest it securely on the cabinet's top.",
        "Without worrying about moving the other decor, just move the wine bottle onto the cabinet top and let it be.",
        "If you've finished looking around the kitchen, go ahead and place the wine bottle on top of the cabinet.",
        "To make the whole room look a bit more tidy, let's put the wine bottle on the very top of the cabinet.",
        "There's no rush, so take it slow and set the wine bottle on the cabinet's upper surface where it's safe.",
        "Think of it as a small step toward organizing the place—just place the wine bottle on top of the cabinet.",
        "Once you're confident that you've got a steady hold on the glass, put the wine bottle on the cabinet top."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "First, take a look at the handles, open the top drawer, and then place the bowl inside it as neatly as you can.",
        "When you're ready to clear the counter, pull open the top drawer and tuck the bowl safely inside.",
        "To keep the bowl from getting bumped, go ahead and open the top drawer and put the bowl in there.",
        "Without disturbing the items in the other drawers, just open the top drawer and place the bowl inside.",
        "Take a quick glance at the drawer labels if there are any, then open the top one and put the bowl in.",
        "Let's do this as a smooth two-step move—first open the top drawer, and then set the bowl inside.",
        "If you think the bowl is better off stored for now, go ahead and open the top drawer and put it inside.",
        "Go ahead and reach for the upper drawer, open it wide, and then place the bowl into it very carefully.",
        "After you've adjusted your grip on the bowl so it won't slip, open the top drawer and slide it right in.",
        "Start by finding the handle to the top drawer, open it, and once it's fully open, put the bowl inside."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Whenever it feels convenient for you, please take the bowl and place it on top of the cabinet.",
        "To help get the bowl out of your way while you work, go ahead and set it on the cabinet's top surface.",
        "Don't worry about adjusting the other items up there; just move the bowl onto the top of the cabinet.",
        "Watch your reach as you carefully lift the bowl and place it on the cabinet top.",
        "If you're in the mood to tidy up the kitchen area, let's put the bowl up on top of the cabinet.",
        "It’s just a simple relocation of the item: go ahead and set the bowl on the cabinet's upper surface.",
        "Once you feel like you have a steady hold on the bowl, place it on top of the cabinet securely.",
        "Even if this is just a temporary spot for it, put the bowl on the cabinet top for the moment.",
        "After you've double-checked that the cabinet top has enough room, go ahead and place the bowl there.",
        "Go ahead and stretch a bit if you need to, then move the bowl onto the cabinet's top surface."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Before we get into anything more complicated, just push the plate toward the front edge of the stove.",
        "If it makes it easier for you to reach later, slide the plate to the very front of the stove area.",
        "With a gentle touch, nudge the plate forward so it ends up sitting right at the front of the stove.",
        "Don't worry about the other objects nearby, just carefully push the plate to the stove's front.",
        "Take your time with the movement and push the plate until it's positioned at the front of the stove.",
        "To make the layout feel a bit more accessible, move the plate forward to the front of the stove.",
        "Even a tiny adjustment is perfectly fine—just push the plate toward the front side of the stove.",
        "When you've found a good angle for your hand, push the plate toward the front of the stove area.",
        "Let's just reposition it for a second: push the plate all the way to the front of the stove top.",
        "After taking a quick look at where the burners are, push the plate to the front of the stove."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "When you're feeling ready to prep, go ahead and place the cream cheese into the bowl, nice and simple.",
        "Without worrying about the stove settings right now, just put the cream cheese in the bowl.",
        "To get a head start on preparing the ingredients, go ahead and put the cream cheese into the bowl.",
        "Even if you aren't ready to start mixing yet, go ahead and place the cream cheese in the bowl.",
        "Pick up the cream cheese package and drop it into the bowl as carefully as you can.",
        "This is just a basic ingredient step: put the cream cheese in the bowl and just leave it there.",
        "After you've confirmed you're holding the right item, put the cream cheese into the bowl.",
        "Try not to let the other items distract you—just put the cream cheese in the bowl for now.",
        "If you feel like you have a steady grip on the package, place the cream cheese into the bowl gently.",
        "For the time being, let's simply move the cream cheese into the bowl and get it ready."
    ],

    "turn_on_the_stove": [
        "When you've got your bearings in the kitchen, go ahead and turn on the stove for me.",
        "Even if you aren't planning to cook something this very second, let's turn on the stove now.",
        "After you've done a quick safety check of the area, go ahead and switch the stove on.",
        "It's just a quick, simple step—turn on the stove and keep everything else exactly as it is.",
        "Without moving any of the objects on the counter, just reach over and turn on the stove.",
        "If you've located the correct control knob, go ahead and turn the stove on now.",
        "Take a brief moment to focus on what you're doing, and then simply turn on the stove.",
        "As the next logical step in the process, go ahead and turn on the stove.",
        "Once you're all set and ready to proceed, go ahead and activate the stove.",
        "Carefully reach out toward the controls and go ahead and turn on the stove."
    ],

    "put_the_bowl_on_the_plate": [
        "When you're ready to stack things, go ahead and place the bowl on top of the plate.",
        "Without worrying about what's happening with the stove, just set the bowl onto the plate.",
        "To help stack the dishes more neatly, go ahead and put the bowl on the plate.",
        "Carefully balance the bowl and place it onto the plate so it sits nice and flat.",
        "Even if this is just a temporary arrangement, set the bowl on the plate for now.",
        "Take a slow, steady breath and place the bowl on the plate without letting it slide.",
        "If you feel you have a firm grip on the bowl, put it onto the plate very gently.",
        "Think of this as a simple organizing step: just place the bowl on the plate.",
        "Without changing the position of anything else, go ahead and move the bowl onto the plate.",
        "Go ahead and position the bowl on the plate, trying to center it if you can."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "When you get a free moment in your routine, place the wine bottle on the rack.",
        "To make sure it's stored safely and neatly, go ahead and put the wine bottle onto the rack.",
        "Without shifting the other bottles around, just set the wine bottle on the rack.",
        "Take care with the glass as you move the wine bottle and rest it on the rack.",
        "Even if you aren't planning to open it soon, go ahead and place the wine bottle on the rack.",
        "Take it nice and slow—put the wine bottle on the rack so it's stable and won't fall.",
        "If the rack is easily within your reach, go ahead and set the wine bottle on it gently.",
        "As a quick way to help organize the space, just place the wine bottle on the rack.",
        "After you've checked that the rack area has an open spot, put the wine bottle onto it.",
        "Go ahead and find a good position for the wine bottle on the rack and set it there securely."
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
