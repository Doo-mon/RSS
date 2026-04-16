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
        "Access the mid-level pull-out compartment of the vertical multi-tier storage structure.",
        "Slide open the centrally located storage bay within the tall furniture unit used for stowing household items.",
        "Draw out the intermediate retractable section of the rectangular storage enclosure.",
        "Unseal the sliding cavity positioned halfway between the highest and lowest sections of the storage unit.",
        "Expose the central horizontal retrieval tray of the furniture intended for organizing supplies.",
        "Pull toward you the middle-height containment unit of the upright storage assembly.",
        "Open the intermediate sliding tray embedded in the multi-layered wooden organizer.",
        "Reveal the interior of the centrally positioned storage slot in the tall boxy fixture.",
        "Extend the mid-height sliding chamber of the storage housing to make its contents visible.",
        "Activate the opening of the centrally placed storage bay in the standing furniture structure."
    ],

    "put_the_bowl_on_the_stove": [
        "Place the concave vessel meant for holding ingredients onto the surface where thermal energy is generated for cooking.",
        "Set the rounded, hollow container for serving food onto the appliance top equipped with heating elements.",
        "Move the open-topped hemispherical receptacle onto the flat platform used to apply heat to cookware.",
        "Position the curved food-holding unit on the area designated for high-temperature meal preparation.",
        "Transfer the ingredient-holding vessel onto the primary heating zone of the kitchen cooking apparatus.",
        "Rest the container with a deep interior cavity onto the horizontal plane where burners provide heat.",
        "Deposit the mixing receptacle onto the upper surface of the heat-providing kitchen station.",
        "Place the bowl-shaped food holder on the top surface of the device used for boiling and frying.",
        "Set the hollowed-out serving dish on the designated thermal region of the cooking appliance.",
        "Move the concave ceramic or plastic container onto the appliance surface that produces heat."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Place the tall, narrow-necked glass container for fermented liquids onto the highest flat surface of the storage unit.",
        "Set the sealed glass cylinder typically holding grape-based drinks on the uppermost plane of the tall furniture.",
        "Move the elongated beverage vessel to the ceiling-facing exterior surface of the storage structure.",
        "Rest the upright glass container for liquids on the top-most horizontal panel of the large storage enclosure.",
        "Put the long-necked liquid storage object onto the highest level of the multi-tier furniture unit.",
        "Position the sealed beverage container on the uppermost exterior platform of the rectangular storage body.",
        "Place the tall glass bottle onto the highest horizontal boundary of the standing storage fixture.",
        "Transfer the narrow-necked liquid vessel to the top surface of the boxy storage tower.",
        "Set the glass vessel for stored drinks on the highest reachable surface of the storage cabinet.",
        "Move the elongated container of spirits to the very top face of the vertical storage assembly."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Unseal the highest pull-out compartment and deposit the concave food container into its interior void.",
        "Slide open the uppermost storage bay and place the rounded ingredient-holding vessel inside that space.",
        "Open the highest sliding chamber and move the hollowed-out food receptacle into the retractable cavity.",
        "Expose the top-most pull-out section and store the curved container for ingredients within it.",
        "Pull out the highest storage tray of the unit and place the concave serving vessel inside.",
        "Access the interior of the top-most sliding compartment and transfer the rounded food holder into it.",
        "Extend the highest retractable drawer and position the concave vessel for mixing inside.",
        "Open the uppermost containment bay and place the bowl-shaped food holder into the storage slot.",
        "Slide the top storage drawer outward and set the hemispherical food container within the compartment.",
        "Open the highest available pull-out bay and move the hollow food vessel into that interior space."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Place the concave vessel used for holding food onto the highest horizontal surface of the storage structure.",
        "Set the open-topped hemispherical container on the topmost plane of the tall furniture unit.",
        "Move the rounded vessel for mixing or serving onto the upper face of the standing storage fixture.",
        "Rest the curved food-holding receptacle on the highest flat panel of the storage enclosure.",
        "Put the ingredient-holding container on the top exterior surface of the large rectangular storage body.",
        "Transfer the bowl-shaped food vessel to the uppermost platform of the multi-tier storage unit.",
        "Position the concave container on the highest reachable flat surface of the furniture fixture.",
        "Place the hollowed-out dish for food onto the top plane of the standing storage cabinet.",
        "Set the rounded serving receptacle on the uppermost horizontal boundary of the storage assembly.",
        "Move the concave mixing unit onto the highest exterior surface of the tall storage structure."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Propel the flat, shallow dish for supporting food toward the forward edge of the heat-producing cooking platform.",
        "Slide the thin, circular food-serving surface closer to the front boundary of the cooking appliance.",
        "Move the flat, low-profile vessel for meals toward the forward-most side of the heating surface.",
        "Nudge the shallow food-bearing surface toward the front region of the apparatus used for thermal cooking.",
        "Shift the flat serving disk forward to the leading edge of the active cooking area.",
        "Push the flat, round food-holding platform toward the front side of the burner-equipped surface.",
        "Advance the shallow meal-supporting surface to the front portion of the cooking appliance's top.",
        "Move the flat ceramic or glass food surface toward the edge closest to the user on the cooking heater.",
        "Slide the low-profile food-carrying object forward until it occupies the front zone of the heating appliance.",
        "Propel the flat dish for holding portions toward the forward boundary of the stove's top surface."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "Place the soft, spreadable dairy product into the concave container intended to hold ingredients.",
        "Transfer the creamy cultured milk spread into the open-topped vessel used for mixing.",
        "Move the block of spreadable cheese into the rounded receptacle with an interior cavity.",
        "Deposit the soft dairy substance into the curved food holder.",
        "Put the packaged spreadable cheese product into the ingredient-holding receptacle.",
        "Place the creamy dairy item into the concave vessel used to contain food components.",
        "Insert the soft, white spreadable dairy block into the bowl-shaped container.",
        "Move the creamy cheese spread into the container designed to keep solids together.",
        "Deposit the soft spread typically used on bagels into the mixing receptacle.",
        "Transfer the spreadable dairy product into the concave-shaped food vessel."
    ],

    "turn_on_the_stove": [
        "Activate the heat-generating cooking apparatus so it enters its functional state.",
        "Switch the cooking heat source from an idle state to an active energy-producing state.",
        "Enable the burner-equipped surface to begin generating thermal energy for cooking.",
        "Turn the power control of the cooking appliance to its operational setting.",
        "Initiate the heating function of the device used for food preparation on the countertop.",
        "Engage the mechanism that provides heat on the upper surface of the cooking station.",
        "Power up the appliance responsible for producing high temperatures for cookware.",
        "Set the cooking heat provider to its active, heat-emitting state.",
        "Toggle the heating elements of the primary cooking surface to the 'on' position.",
        "Activate the thermal output of the appliance used to heat pots and pans."
    ],

    "put_the_bowl_on_the_plate": [
        "Place the concave food container onto the flat circular dish that serves as a base.",
        "Set the rounded receptacle with an interior cavity on top of the flat-surfaced serving disk.",
        "Move the bowl-shaped vessel onto the shallow, flat food-supporting surface.",
        "Rest the open-topped container on the flat disk used for presenting meal portions.",
        "Put the ingredient-holding vessel onto the flat ceramic surface that functions as a base.",
        "Position the concave container atop the flat, low-profile dish used for meals.",
        "Transfer the mixing receptacle onto the flat, horizontal food-support surface.",
        "Place the curved container on the flat dish so it is supported by the flat surface below.",
        "Set the bowl-like vessel on the flat serving surface used to carry food items.",
        "Move the food receptacle onto the flat, disk-shaped meal platform."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "Place the tall, narrow-necked glass container onto the slatted framework designed for organization.",
        "Set the long-necked glass vessel for liquids on the multi-bar support structure.",
        "Move the sealed beverage bottle onto the storage frame that supports objects in designated slots.",
        "Rest the bottle-shaped liquid container on the slatted organizer assembly.",
        "Put the upright glass vessel onto the rack-like holder used for orderly placement.",
        "Transfer the tall glass container to the supporting stand composed of parallel bars.",
        "Place the glass beverage vessel onto the holder structure meant to keep items off the counter surface.",
        "Set the tall narrow vessel on the slatted surface where items are positioned for storage.",
        "Move the drink-containing glass bottle onto the organized support frame with bars.",
        "Position the sealed liquid container onto the rack so it is held by the supporting structure."
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
