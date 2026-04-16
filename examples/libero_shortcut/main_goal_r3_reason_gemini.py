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
        "First, identify which drawer is vertically centered, then apply outward pressure until the interior is accessible.",
        "The end state must be the middle drawer in an open position; execute the necessary pull to achieve this.",
        "Check if the middle drawer is closed; if so, slide it out until it reaches its maximum extension.",
        "Ensure your hand is on the correct handle—the one between the top and bottom—then pull it toward you.",
        "Verify the cabinet's identity, then perform a sequence to leave the middle drawer fully open.",
        "Scan the drawer stack, select the center one, and confirm it is no longer flush with the cabinet face after your action.",
        "If you want to see what's inside the middle section, you'll need to pull that specific drawer outward now.",
        "The objective is a state change for the middle drawer from 'shut' to 'open'; please facilitate this.",
        "Locate the midpoint of the cabinet's height and ensure the drawer at that level is moved to an open state.",
        "Confirm the top and bottom drawers remain closed while you transition the middle one to the open position."
    ],

    "put_the_bowl_on_the_stove": [
        "The bowl needs to transition from its current spot to the stove surface; ensure it is stable once placed.",
        "Verify the stove top is clear, then move the bowl until its base is in full contact with the stove.",
        "If the bowl is not yet on the cooking surface, pick it up and deposit it there now.",
        "Achieve the following configuration: bowl resting on stove, table surface empty of this specific bowl.",
        "First, orient toward the heat-generating appliance, then carefully lower the bowl onto its flat top.",
        "Check that you have a firm grip, then transfer the bowl to the stove area and release it.",
        "The final condition for success is that the bowl must be supported entirely by the stove top.",
        "Identify the stove as the target platform, then execute the movement to place the bowl there.",
        "Ensure the bowl is no longer in your hand and is instead resting securely on the stove surface.",
        "Before finishing, confirm that the bowl has been successfully relocated to the stove area."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Lift the wine bottle to a height exceeding the cabinet, then lower it until it rests on the topmost surface.",
        "The goal is for the bottle to be on the cabinet's roof; check the clearance and place it there.",
        "If the wine bottle is currently on the table, perform the steps to move it to the highest point of the cabinet.",
        "Maintain the bottle's vertical orientation while you transition it to the top of the storage unit.",
        "Confirm the cabinet top is the final resting place, then release the bottle once stability is reached.",
        "First, verify the bottle is sealed or steady, then proceed to set it on the upper face of the cabinet.",
        "The wine bottle should end up in a 'placed' state on the cabinet's highest horizontal panel.",
        "Observe the cabinet's height, then lift and position the bottle on that upper boundary.",
        "Ensure the bottle does not tip; move it to the top of the cabinet and verify its balance.",
        "Relocate the beverage container such that the cabinet top becomes its new supporting surface."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Execute a two-part sequence: first, transition the top drawer to 'open', then place the bowl in the cavity.",
        "If the top drawer is shut, pull it out; once the space is revealed, deposit the bowl inside.",
        "The final state requires the top drawer to be open and the bowl to be contained within its boundaries.",
        "Confirm you are at the highest drawer level, open it, and then ensure the bowl is moved into that space.",
        "Check for internal clearance in the top drawer, then open it and set the bowl inside securely.",
        "First, make the top drawer's interior accessible; second, transfer the bowl from the table to that interior.",
        "Start by sliding the uppermost drawer outward; only after it is open should you put the bowl in it.",
        "The objective is bowl-inside-top-drawer; perform the necessary opening and placement actions.",
        "Identify the top handle, pull to expose the drawer space, and then place the bowl inside the unit.",
        "Verify the bowl is no longer visible from the top-down view once the drawer is partially closed or filled."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Analyze the cabinet's height, lift the bowl, and ensure it ends up resting on the topmost surface.",
        "The bowl's final coordinates must be on the upper exterior of the cabinet; move it there now.",
        "First, stabilize the bowl in your hand, then relocate it to the highest flat plane of the storage unit.",
        "If the bowl is resting on the table, pick it up and place it on the top of the cabinet.",
        "Confirm the bowl is balanced on the cabinet's roof before you consider the task complete.",
        "Identify the cabinet as the destination and achieve a 'bowl-on-top' final configuration.",
        "Ensure the bowl is no longer at its starting position and is instead on the cabinet's upper surface.",
        "Move the bowl upward until it is above the cabinet, then set it down on the top panel.",
        "Verify that the bowl is securely supported by the cabinet's top surface and release your grip.",
        "The goal is a successful transfer of the bowl to the highest horizontal surface of the cabinet unit."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "First, locate the plate on the stove, then apply a forward force until it reaches the front edge.",
        "The plate must end up closer to the user than it currently is; push it toward the stove's front boundary.",
        "If the plate is at the back or center, slide it forward until the front-of-stove condition is met.",
        "Maintain contact between the plate and the stove surface while you move it to the leading edge.",
        "Confirm the plate's final position is at the front of the heating appliance before stopping.",
        "Execute a steady push to relocate the plate to the forward-most part of the stove top.",
        "The objective is to minimize the distance between the plate and the front edge of the stove.",
        "Identify the 'front' zone of the stove and ensure the plate is moved into that specific area.",
        "Slide the plate along the stove's surface until it cannot go further forward without falling.",
        "Check the plate's current position; if it's not at the front, push it there immediately."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "The cream cheese needs to be contained by the bowl; move it into the bowl's interior cavity.",
        "First, locate the spreadable dairy item, then ensure its final position is inside the bowl.",
        "If the cream cheese is on the table, pick it up and place it within the bowl's boundaries.",
        "Verify the target vessel is the bowl, then deposit the cream cheese into it and release.",
        "Achieve the state where the cream cheese is fully supported by the inner surface of the bowl.",
        "Check that the bowl is ready to receive items, then put the cream cheese inside it.",
        "The goal state is 'cream cheese inside bowl'; perform the transfer to reach this state.",
        "Move the cream cheese over the bowl opening and lower it until it is inside the container.",
        "Confirm the cream cheese has been relocated from the table to the interior of the bowl.",
        "Ensure the cream cheese is securely placed within the bowl before concluding the action."
    ],

    "turn_on_the_stove": [
        "Identify the power control for the stove and transition it from the 'off' state to the 'on' state.",
        "If the stove is currently inactive, manipulate the controls until the heating function is enabled.",
        "The final required state is an active stove; use the dial or switch to achieve this.",
        "First, find the activation mechanism, then confirm the stove is on after you interact with it.",
        "Check if the stove is off; if so, trigger the 'on' setting to begin the heating process.",
        "Locate the stove's operational interface and ensure the device is switched to its active mode.",
        "The objective is for the stove to be powered on; perform the necessary control adjustment.",
        "Verify the stove is ready for use, then toggle the 'on' switch to change its status.",
        "Locate the specific knob or button that controls the stove's power and activate it now.",
        "Transition the appliance to an operational state by turning the stove on."
    ],

    "put_the_bowl_on_the_plate": [
        "The bowl's base must come into contact with the plate's top surface; place it there now.",
        "Identify the plate as the foundation, then set the bowl down so it is supported by the plate.",
        "If the bowl and plate are separate, move the bowl until it is stacked on top of the plate.",
        "Ensure the bowl is centered on the plate to maintain balance in the final configuration.",
        "The goal state is a vertical stack: plate on the bottom, bowl on the top.",
        "First, pick up the bowl, then align it with the plate and lower it into position.",
        "Confirm the plate is resting on the table before you place the bowl onto its surface.",
        "Relocate the bowl so that it is no longer touching the table, but is touching the plate instead.",
        "Verify the stability of the bowl once it has been placed onto the plate's surface.",
        "Perform the necessary movements to ensure the bowl is resting entirely on the plate."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "First, identify the storage rack, then place the wine bottle onto its support structure.",
        "The wine bottle must transition to the rack; ensure it is properly slotted or seated.",
        "If the bottle is on the table, relocate it to the rack and confirm it is held securely.",
        "Achieve the final state where the wine bottle's weight is supported by the rack's bars or frame.",
        "Check the rack for an available spot, then move the bottle into that specific position.",
        "The goal is 'bottle-on-rack'; transfer the beverage container to achieve this arrangement.",
        "Verify the rack is stable, then carefully place the wine bottle onto the designated area.",
        "Ensure the wine bottle is no longer on the flat table surface and is instead on the rack.",
        "Locate the rack's boundaries and place the bottle within them so it won't slip.",
        "Move the wine bottle onto the rack and verify its final resting state is secure."
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
