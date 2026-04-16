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
        "Extract the intermediate sliding storage container from the multi-shelf furniture assembly.",
        "Pull outward the central containment bin located within the vertical household unit.",
        "Gain access to the mid-level retractable box inside the structured storage piece.",
        "Open the compartment situated halfway down the vertical storage structure for items.",
        "Slide out the middle containment unit embedded within the standing storage fixture.",
        "Reveal the interior space of the central drawer-like section in the storage cabinet.",
        "Draw open the storage compartment situated between the upper and lower sections.",
        "Unlock the mid-height sliding storage space within the large furniture piece.",
        "Manipulate the central retrieval box to an open state within the storage framework.",
        "Expose the middle-level storage cavity by pulling it outward from the cabinet body."
    ],

    "put_the_bowl_on_the_stove": [
        "Position the rounded open-top vessel on the appliance designed specifically for heating food.",
        "Rest the curved ingredient holder on the thermal surface used for cooking processes.",
        "Set the deep dish used for mixing onto the heat-producing kitchen device structure.",
        "Place the hollow food container onto the burner-equipped platform used for warming.",
        "Locate the rounded serving vessel on the area meant for applying thermal energy.",
        "Put the concave kitchenware item on the stove's designated heating zone surface.",
        "Transfer the vessel designed for holding liquids to the cooking appliance top plane.",
        "Situate the open-mouthed container on the heat-generating station for food prep.",
        "Place the food preparation bowl onto the thermal energy source surface area.",
        "Deposit the rounded mixing tool onto the cooking heater's flat top section."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Set the cylindrical glass vessel for alcohol on the uppermost shelf of the storage unit.",
        "Place the sealed liquid container on the highest horizontal plane of the cupboard.",
        "Rest the narrow-necked bottle on the top panel of the storage furniture.",
        "Position the fermented drink holder on the upper surface of the cabinet structure.",
        "Put the tall glass container onto the roof-like surface of the storage box.",
        "Move the beverage vessel to the highest point of the storage enclosure.",
        "Place the wine container on the top flat area of the household storage unit.",
        "Set the long-shaped drink bottle on the uppermost section of the cabinet.",
        "Locate the glass liquid holder on the top exterior surface of the storage piece.",
        "Transfer the bottled beverage to the highest platform of the storage furniture."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Open the highest sliding bin and insert the rounded food vessel into it.",
        "Pull out the upper compartment and place the concave container within.",
        "Access the top storage drawer and deposit the mixing bowl inside.",
        "Slide open the uppermost retrieval space and put the ingredient holder in.",
        "Open the top-level sliding box and store the curved dish within its cavity.",
        "Expose the highest pull-out section and place the open-top vessel inside.",
        "Unlock the upper storage compartment and insert the rounded container.",
        "Draw out the top drawer-like structure and put the food bowl within.",
        "Open the highest containment slot and place the concave kitchenware inside.",
        "Reveal the top sliding storage area and deposit the mixing vessel into it."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Place the rounded food vessel on the highest flat surface of the storage unit.",
        "Set the concave container on the top panel of the cabinet structure.",
        "Rest the mixing bowl on the uppermost horizontal plane of the furniture.",
        "Put the open-top dish on the roof of the storage enclosure.",
        "Position the ingredient holder on the top exterior surface of the cabinet.",
        "Transfer the curved container to the highest point of the storage box.",
        "Place the bowl-shaped object on the upper flat area of the storage unit.",
        "Set the rounded kitchenware on the top section of the storage furniture.",
        "Locate the concave vessel on the highest platform of the cabinet.",
        "Move the food container to the top surface of the storage structure."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Slide the flat dish toward the front edge of the heating appliance.",
        "Push the circular food holder to the forward side of the stove surface.",
        "Move the shallow plate to the front boundary of the cooking platform.",
        "Nudge the flat serving object toward the user-facing edge of the heater.",
        "Advance the disk-shaped dish to the front region of the heat source.",
        "Shift the flat food surface to the forward limit of the cooking station.",
        "Propel the plate-like object toward the front of the thermal appliance.",
        "Slide the serving disk to the front zone of the stove's top area.",
        "Push the flat culinary tool toward the front edge of the heating unit.",
        "Move the shallow dish to the forwardmost part of the cooking surface."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "Insert the soft dairy spread into the rounded mixing vessel.",
        "Place the creamy cheese product into the concave food container.",
        "Put the spreadable dairy item into the open-top bowl structure.",
        "Deposit the soft cheese substance into the curved ingredient holder.",
        "Transfer the creamy dairy block into the vessel designed for mixing.",
        "Move the soft food product into the concave kitchenware item.",
        "Place the cheese spread into the rounded container for ingredients.",
        "Insert the dairy product into the open-mouthed food vessel.",
        "Put the soft spreadable item into the bowl-shaped receptacle.",
        "Deposit the creamy cheese into the concave holding container."
    ],

    "turn_on_the_stove": [
        "Switch the cooking heater to the active power state.",
        "Enable the thermal energy source for food preparation.",
        "Activate the stove's heating mechanism.",
        "Turn on the appliance used for cooking with heat.",
        "Initiate the operation of the cooking surface heater.",
        "Power on the device responsible for thermal cooking.",
        "Engage the heating function of the kitchen stove.",
        "Set the cooking platform to an on state.",
        "Start the heat generation process on the stove.",
        "Activate the thermal appliance for cooking purposes."
    ],

    "put_the_bowl_on_the_plate": [
        "Place the concave vessel on top of the flat food dish.",
        "Set the rounded container onto the shallow serving plate.",
        "Rest the mixing bowl on the flat circular dish.",
        "Put the open-top vessel onto the flat food holder.",
        "Position the curved container atop the flat serving surface.",
        "Transfer the bowl-shaped object onto the flat plate structure.",
        "Place the ingredient holder on the flat dish base.",
        "Set the rounded kitchenware on the flat food platform.",
        "Locate the concave vessel on the flat serving disk.",
        "Move the food container onto the flat plate object."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "Place the glass beverage container on the slatted support structure.",
        "Set the tall bottle on the rack designed for holding items.",
        "Rest the wine vessel on the organized storage stand.",
        "Position the liquid container on the multi-bar holder.",
        "Put the drink bottle onto the framework used for storage.",
        "Transfer the glass vessel to the rack surface.",
        "Place the sealed beverage container on the supporting rails.",
        "Set the tall drink holder on the storage rack structure.",
        "Move the wine bottle onto the elevated holding frame.",
        "Position the liquid vessel on the rack designed for bottles."
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
