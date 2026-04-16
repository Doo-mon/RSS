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
        "Before touching any other part of the cabinet, locate the drawer that's positioned exactly halfway up, then pull it outward until it's clearly extended and stays open.",
        "Make absolutely sure you're opening the middle drawer and not the top or bottom one—once you've confirmed that, go ahead and open it fully.",
        "First visually scan the cabinet to identify the middle drawer based on its vertical position, then slide it outward until it's fully accessible.",
        "Ensure that the cabinet you're facing is the correct storage unit before proceeding; after that, open its middle drawer completely.",
        "The drawer that sits between the top and bottom compartments is the target—open that specific one and stop only when it's fully open.",
        "Start by aligning yourself with the cabinet front so you have a clear view, then pull the middle drawer open in one smooth, continuous motion.",
        "Check which drawer is centered vertically relative to the cabinet's height, and once you've identified it correctly, open that drawer.",
        "Open the middle drawer and then verify that its front panel has moved outward, confirming the drawer is indeed open.",
        "If the middle drawer is currently closed, grasp its handle and pull it outward until you can easily access the interior space.",
        "The goal is simply to have the cabinet's middle drawer open—do whatever is necessary to reach that state, regardless of what else is around."
    ],

    "put_the_bowl_on_the_stove": [
        "First locate the stove's top surface among the kitchen appliances, then move the bowl until it rests securely on that surface.",
        "Ensure you're using the stove as the target surface and not the cabinet or counter—once confirmed, place the bowl on it securely.",
        "Pick up the bowl carefully, move it over the stove area, and release it only once it's stable and fully resting on the stove.",
        "The desired final outcome is bowl-on-stove: place the bowl so it ends up resting on the stove top, regardless of how you get there.",
        "Check the stove area to make sure it's clear enough, then position the bowl on the stove without tipping it or causing it to slide.",
        "Align the bowl directly above the stove surface, then lower it down and set it on the stove, ensuring it doesn't fall off.",
        "If the bowl isn't already sitting on the stove, transfer it there now and confirm that it's fully supported by the stove surface.",
        "Move the bowl onto the stove top and verify afterward that it is no longer on the table or counter where it was before.",
        "Place the bowl on the stove and make sure it stays entirely within the stove's top area, not hanging over the edge.",
        "First bring the bowl close to the stove's edge, then carefully set it down on the stove surface, ensuring it's stable."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Locate the cabinet's top surface—it's the highest horizontal plane—then lift the wine bottle and place it onto that top area.",
        "Ensure the bottle remains upright throughout the movement to avoid spills, then place it on top of the cabinet securely.",
        "Pick up the wine bottle, raise it above cabinet height, and gently set it down on the cabinet top, checking that it's stable.",
        "The goal-state is straightforward: the wine bottle should end up resting on the cabinet's top surface—achieve that outcome.",
        "Move the bottle over the cabinet, align it with a safe spot on top, then release it once it is stable and not wobbling.",
        "First align the bottle with the cabinet top to gauge placement, then place it down carefully to avoid tipping or rolling off.",
        "If the bottle is currently elsewhere on the counter, transfer it onto the cabinet's top and confirm its placement afterward.",
        "Put the wine bottle on top of the cabinet and verify that it is no longer sitting on the table or any lower surface.",
        "Place the bottle onto the cabinet's upper surface, maintaining its balance throughout the movement until it's settled.",
        "Identify the cabinet first, then place the wine bottle onto its topmost surface, ensuring it's fully on top and not hanging off."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Open the top drawer fully so its interior is completely accessible, then move the bowl into the drawer and leave it there.",
        "First confirm that the top drawer is indeed open and stable, then take the bowl and place it inside, releasing it gently.",
        "Pull the upper drawer out until you can easily reach inside; next, set the bowl into the drawer cavity and let go.",
        "The goal-state is simple: the bowl should end up inside the top drawer—open it if it's closed, then place the bowl in.",
        "Open the top drawer first, position the bowl directly over the drawer opening, then place it gently inside without forcing it.",
        "Ensure you're using the top drawer specifically and not the middle one—open it, then put the bowl into the drawer space.",
        "Open the top drawer and verify there's enough clearance inside; then transfer the bowl into the drawer, making sure it fits.",
        "Follow this sequence: drawer open first → then bowl moved into drawer → finally bowl fully inside the drawer boundaries.",
        "If the drawer is currently closed, open the top one first; afterward, place the bowl inside and confirm it's fully contained.",
        "Open the upper drawer, place the bowl inside it, and make sure the bowl is no longer resting on the table or counter."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Locate the cabinet's top surface—it's the flat area on top—then lift the bowl and place it onto that elevated area.",
        "Ensure the bowl is stable in your hands, then set it down carefully on top of the cabinet so it doesn't slide off.",
        "Pick up the bowl, move it above the cabinet height, and lower it gently onto the cabinet top, ensuring it's centered.",
        "The goal-state is clear: the bowl should end up resting on the cabinet top surface—achieve that by any safe means.",
        "Align the bowl with the cabinet top to find a good spot, then place it down gently to avoid sliding or tipping.",
        "If the bowl is currently elsewhere on the counter, transfer it to the cabinet top and confirm it's properly placed.",
        "Put the bowl on the cabinet top and verify afterward that it is not on the table or any other surface.",
        "Move the bowl to the highest surface of the cabinet, then release it only once it's steady and fully supported.",
        "Identify the cabinet first, then place the bowl on its topmost surface, ensuring it's completely on top.",
        "Bring the bowl to the cabinet top and make sure it remains on that surface without any part hanging off."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "First identify where the stove's front edge is located, then push the plate until it reaches that forward position.",
        "Ensure the plate stays entirely on the stove surface while you push it forward—stop when it's clearly at the front.",
        "Push the plate forward in a straight, controlled line until it is positioned right at the front of the stove.",
        "The goal-state is simple: the plate should end up at the stove's front—push it until that condition is fully met.",
        "Align your pushing direction directly toward the stove's front edge, then move the plate forward without letting it tip.",
        "If the plate is not currently at the front, nudge it forward gently and confirm its final position is front-of-stove.",
        "Push the plate toward the front edge of the stove, stopping once it's as close to you as possible on the stove surface.",
        "Move the plate forward along the stove top; afterward, verify that it is nearer to the front than it was before.",
        "Push the plate and then check that it ends up positioned exactly at the stove's front area, not somewhere in the middle.",
        "First locate where the plate currently sits on the stove, then push it forward until it's clearly at the front edge."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "Pick up the cream cheese from wherever it is, then place it carefully into the bowl's interior cavity.",
        "Ensure the bowl is the intended target container, then put the cream cheese inside it, making sure it's fully contained.",
        "Move the cream cheese directly over the bowl opening and release it only once it's clearly inside the bowl.",
        "The goal-state is straightforward: cream cheese should end up inside the bowl—place it there if it isn't already.",
        "Position the cream cheese above the bowl's rim, then set it down into the bowl, ensuring it lands inside.",
        "If the cream cheese is currently outside the bowl, transfer it into the bowl now and confirm it's fully contained.",
        "Put the cream cheese into the bowl and verify afterward that it is no longer sitting on the table or counter surface.",
        "First locate the bowl on the counter, then place the cream cheese inside the bowl's boundary, not on the edge.",
        "Place the cream cheese into the bowl gently so it remains inside the bowl and doesn't bounce out.",
        "Put the cream cheese in the bowl, then confirm that the cream cheese is entirely within the bowl's interior area."
    ],

    "turn_on_the_stove": [
        "Locate the stove's control knob or switch, then turn it to the 'on' position so the stove becomes active.",
        "Ensure the stove is currently in the off state, then proceed to turn it on using the appropriate control.",
        "Find the activation control for the stove—usually a knob or button—then turn it on and confirm it's working.",
        "The goal-state is simple: the stove should be on—use whatever control is available to reach that state.",
        "First identify which stove you're dealing with, then activate it via its on-switch or control mechanism.",
        "Turn on the stove, then verify afterward that the stove is no longer in the off state (e.g., by checking for heat/light).",
        "If the stove is currently off, change its state to on by operating the appropriate control.",
        "Activate the stove and confirm that the heating function is enabled and ready for use.",
        "Use the stove's control interface to turn it on, without moving any other objects or disturbing the setup.",
        "First orient yourself to the stove's control area, then turn the stove on by rotating or pressing the correct control."
    ],

    "put_the_bowl_on_the_plate": [
        "Pick up the bowl carefully and place it onto the plate so that the bowl rests fully on the plate's surface.",
        "Ensure the plate is the intended base for the bowl, then set the bowl on it securely without wobbling.",
        "Move the bowl directly above the plate, then lower it until it contacts and rests entirely on the plate surface.",
        "The goal-state is simple: the bowl should end up on the plate—place it there if it isn't already in that position.",
        "Align the bowl with the plate's center for balance, then set it down gently on the plate, checking stability.",
        "If the bowl is currently elsewhere on the counter, transfer it onto the plate and confirm it stays on the plate.",
        "Put the bowl on the plate and verify afterward that the bowl is no longer directly on the table or counter.",
        "Place the bowl onto the plate gently to avoid tipping it off the plate or causing it to slide.",
        "First locate both objects—the bowl and the plate—then place the bowl on top of the plate surface carefully.",
        "Set the bowl down on the plate and confirm that the final configuration is bowl-on-plate, with no part hanging off."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "Locate the rack—it's the slotted or barred holder—then lift the wine bottle and place it onto the rack surface.",
        "Ensure the bottle remains stable and upright, then set it down on the rack so it's securely supported.",
        "Pick up the wine bottle, move it directly over the rack, and place it onto the rack securely without tipping.",
        "The goal-state is clear: the wine bottle should end up on the rack—transfer it there if it isn't already.",
        "Align the bottle with the rack's support area to find a stable spot, then place it down without letting it fall.",
        "If the bottle is not currently on the rack, move it onto the rack now and confirm it's properly placed.",
        "Put the wine bottle on the rack and verify afterward that it is no longer sitting on the table or counter.",
        "First identify the correct rack (the holder structure meant for bottles), then place the bottle on it securely.",
        "Place the wine bottle on the rack and check that it stays entirely within the rack's boundaries, not hanging off.",
        "Bring the wine bottle to the rack, then set it down so the final configuration is bottle-on-rack, stable and secure."
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
