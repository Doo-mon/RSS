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
        "Unseal the centrally positioned pull-out compartment of the multi-tier storage structure that is commonly found in kitchens.",
        "Expose the mid-level sliding bay of the tall wooden enclosure designed for keeping household items organized.",
        "Draw out the intermediate compartment from the box-like organizer with stacked sections, located between the top and bottom ones.",
        "Open the pull-out cavity situated roughly halfway between the uppermost and lowermost storage areas of the furniture piece.",
        "Extend the central storage slot of the vertical unit that is intended for stowing small utensils and miscellaneous supplies.",
        "Access the middle retrieval compartment of the tall storage body by gripping its handle and pulling it outward smoothly.",
        "Slide outward the intermediate containment tray that is built into the standing fixture for keeping various items.",
        "Reveal the center-positioned sliding chamber in the large rectangular assembly, making its interior visible and reachable.",
        "Pull open the mid-height containment compartment that is embedded within the storage housing for daily access.",
        "Open the intermediate pull-out section of the storage unit so its interior space becomes accessible for placing or retrieving objects."
    ],

    "put_the_bowl_on_the_stove": [
        "Place the concave container meant for holding ingredients or prepared food onto the surface associated with applying cooking heat.",
        "Set the rounded vessel used for mixing ingredients or serving meals onto the appliance top where heat is generated for cooking.",
        "Move the open-topped food container onto the flat heating platform that is designed to warm cookware during meal preparation.",
        "Position the small, curved receptacle for holding soups or salads on the area designated for cooking and temperature regulation.",
        "Transfer the ingredient-holding vessel onto the heater-topped cooking apparatus commonly found in most kitchens.",
        "Rest the container with an interior cavity for both liquids and solids onto the primary cooking surface where burners are located.",
        "Put the mixing or serving receptacle onto the heat-providing cooking station's upper plane, ensuring it sits securely.",
        "Place the concave dish-like container on the top surface where burners deliver thermal energy for frying or boiling.",
        "Set the food-holding vessel on the cooking appliance's heating region, which is typically made of metal or ceramic.",
        "Move the round receptacle intended to contain food items onto the heated cooking platform, away from the burners if needed."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "Place the tall, narrow-necked glass vessel designed for storing fermented beverages onto the highest flat surface of the storage unit.",
        "Set the sealed container typically used for pouring grape-based drinks on the upper face of the standing kitchen furniture.",
        "Move the elongated beverage container that often holds alcoholic drinks onto the topmost plane of the boxy storage structure.",
        "Rest the upright glass drink vessel on the upper surface of the large storage enclosure, where items are less likely to be disturbed.",
        "Put the long-necked liquid container, which is usually made of dark glass, onto the highest horizontal panel of the storage fixture.",
        "Position the sealed beverage vessel on the top surface of the rectangular storage body, ensuring it won't roll off the edge.",
        "Place the glass container for storing drinks onto the uppermost platform of the kitchen unit, often used for decorative displays.",
        "Transfer the tall drink container onto the top plane of the cabinet-like storage structure, which is at eye level or higher.",
        "Set the bottle-shaped vessel for liquids on the highest surface of the storage furniture, clearing space below for other tasks.",
        "Move the wine-holding container to the top surface of the storage cabinet-like unit, where it can be stored temporarily or permanently."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Unseal the highest pull-out compartment of the storage unit and place the concave food container within its interior space.",
        "Slide open the uppermost storage bay, then deposit the ingredient-holding vessel into that compartment for safekeeping.",
        "Open the top sliding chamber and move the rounded receptacle for food into the drawer cavity, ensuring it fits properly.",
        "Expose the upper pull-out compartment of the furniture, then place the open-topped container inside the storage space.",
        "Pull out the uppermost compartment by its handle and store the curved food vessel within it, closing it afterward if needed.",
        "Open the highest sliding section of the vertical storage unit, then insert the concave container into the interior cavity.",
        "Extend the top pull-out compartment fully and place the mixing or serving receptacle into it, positioning it securely.",
        "Open the upper storage chamber of the cabinet and put the bowl-shaped vessel inside the compartment, away from other items.",
        "Slide the topmost compartment outward and position the food-holding container within the drawer space, near the back.",
        "Open the highest drawer-like compartment of the storage fixture and transfer the concave receptacle into the interior."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "Place the concave container used for holding ingredients or prepared dishes onto the highest horizontal surface of the storage structure.",
        "Set the open-topped food receptacle on the topmost plane of the cabinet-like storage unit, which is often used for displaying items.",
        "Move the rounded vessel intended for mixing ingredients or serving food onto the upper face of the standing kitchen furniture.",
        "Rest the curved container for holding soups or cereals on the highest flat panel of the storage enclosure, away from the edge.",
        "Put the ingredient-holding receptacle on the top surface of the large rectangular storage body, ensuring it is stable.",
        "Transfer the bowl-shaped food container to the uppermost platform of the storage unit, which is at a convenient height for storage.",
        "Position the concave vessel on the cabinet's highest surface, which serves as a resting plane for various kitchen items.",
        "Place the food container with an interior cavity onto the top plane of the storage fixture, clearing space below for other tasks.",
        "Set the rounded receptacle on the uppermost surface of the cabinet-like furniture, making sure it doesn't block anything.",
        "Move the serving or mixing container onto the top surface of the storage cabinet structure, where it can be easily accessed later."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Propel the flat, circular food-supporting surface toward the forward edge of the heat-producing cooking platform.",
        "Slide the shallow, flat serving surface commonly used for presenting meals closer to the front boundary of the cooking appliance.",
        "Move the circular, flat dish used for holding individual portions of food toward the forward-most side of the heating surface.",
        "Nudge the food-bearing flatware, typically made of ceramic or glass, toward the front region of the apparatus used for cooking.",
        "Shift the flat serving disk forward to the leading edge of the heated cooking area, making it easier to reach.",
        "Push the low-profile food surface, which is designed to hold slices of bread or pieces of meat, toward the front side of the burner-equipped platform.",
        "Advance the flat dish used for presenting appetizers or main courses to the front portion of the cooking surface.",
        "Move the flat, round food platform toward the edge closest to the user on the cooking heater, ensuring it stays on the surface.",
        "Slide the serving plate forward until it occupies the front zone of the heating appliance, near the control knobs.",
        "Propel the flat dish, which is meant for serving cooked items, toward the forward boundary of the stove's top surface."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "Place the soft, spreadable dairy product packaged in a foil-wrapped block into the concave container intended to hold ingredients.",
        "Transfer the creamy cheese spread, often used on bagels or in dips, into the open-topped vessel used for mixing or serving.",
        "Move the packaged soft dairy spread, which is typically refrigerated, into the rounded receptacle with an interior cavity.",
        "Deposit the spreadable cheese substance, known for its smooth texture, into the curved food container for further preparation.",
        "Put the soft dairy block or spread, which comes in a rectangular shape, into the ingredient-holding receptacle.",
        "Place the creamy dairy item, commonly found in breakfast dishes, into the concave vessel used to contain food components.",
        "Insert the spreadable cheese product, which is white and mild in flavor, into the bowl-shaped container for mixing.",
        "Move the soft cheese spread, often used in cheesecakes or frosting, into the container designed to keep liquids and solids together.",
        "Deposit the dairy spread typically used on bread or crackers into the mixing receptacle, where it can be combined with other items.",
        "Transfer the creamy cheese product, which is smooth and easily spreadable, into the concave food vessel for temporary storage or mixing."
    ],

    "turn_on_the_stove": [
        "Activate the heat-generating cooking apparatus so it enters an operating state, producing warmth for food preparation.",
        "Switch the cooking heat source from an inactive state to an active state, allowing burners to begin heating.",
        "Enable the burner-equipped surface to begin producing thermal energy for boiling water or frying food.",
        "Turn the heating function of the cooking appliance on, typically by rotating a knob or pressing a button.",
        "Initiate operation of the device used to apply heat to cookware during meal preparation in the kitchen.",
        "Change the cooking platform's power state so it becomes active, ready for placing pots or pans on it.",
        "Engage the heating mechanism of the stovetop cooking station, which may use gas or electricity as fuel.",
        "Set the cooking heat provider to its on state, ensuring that the burners are ready to warm up.",
        "Power up the appliance responsible for delivering cooking heat at the top surface, commonly located in residential kitchens.",
        "Activate the cooking heater so it begins functioning as a heat source, allowing for subsequent cooking tasks."
    ],

    "put_the_bowl_on_the_plate": [
        "Place the concave food container, which is deeper than a typical dish, onto the flat dish that serves as a base for holding food.",
        "Set the rounded receptacle with an interior cavity, used for holding liquids or solid foods, on top of the flat serving surface.",
        "Move the bowl-shaped vessel, designed for containing soups or cereals, onto the shallow, flat food-supporting dish.",
        "Rest the open-topped container, which can hold multiple servings, on the flat disk used for presenting individual food portions.",
        "Put the ingredient-holding vessel, often made of ceramic or glass, onto the flat dish that functions as a serving base.",
        "Position the concave container, which has curved sides, atop the flat plate-like surface used for meals, ensuring stability.",
        "Transfer the mixing or serving receptacle, which is wider than it is tall, onto the flat food-support surface beneath it.",
        "Place the curved container, which is designed to hold contents without spilling, on the flat dish so it sits supported by it.",
        "Set the bowl-like vessel, which has a rounded bottom, on the flat serving disk used to carry food to the table.",
        "Move the food receptacle, which is typically used for individual portions, onto the flat plate-shaped base for presentation or stacking."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "Place the tall beverage container, often made of glass with a narrow neck, onto the slatted holding structure designed for organizing bottles.",
        "Set the long-necked glass drink vessel, which contains fermented grape liquid, on the multi-bar support stand for storage.",
        "Move the sealed beverage container, which may have a cork or screw cap, onto the storage frame that supports objects in elevated slots.",
        "Rest the bottle-shaped liquid vessel, which is cylindrical in form, on the slatted organizer structure commonly found in kitchens or cellars.",
        "Put the upright drink container, designed for pouring, onto the rack-like holder used for placement and organization of similar items.",
        "Transfer the glass liquid container, which may hold red or white wine, to the supporting stand composed of parallel rails or wires.",
        "Place the fermented-beverage container, which benefits from horizontal storage, onto the holder structure meant to keep bottles separated.",
        "Set the tall glass vessel, which may have a punt at the bottom, on the rack surface where objects are positioned for aging or display.",
        "Move the drink bottle, which contains an alcoholic beverage, onto the organized support frame with bars or slots for secure placement.",
        "Position the sealed beverage vessel, which is heavier when full, onto the rack so it is supported by the rack structure without tipping."
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
