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
        "Before you even think about what's on the counter, could you do me a favor and just pull open the middle drawer on that cabinet there?",
        "I know you're looking around, but first things first—go ahead and open the cabinet's middle drawer for me, just so we have it ready.",
        "Take a quick step to the side so you're facing the cabinet directly, and once you're there, gently slide the middle drawer open.",
        "If you wouldn't mind pausing for a second, just reach out and open the middle drawer on the cabinet—no need to look inside or anything.",
        "While you're standing near the counter, just lean over and pull the middle drawer of the cabinet open, even if just a little bit.",
        "After you've gotten your bearings in the room, the very first thing to do is to open that cabinet's middle drawer, nice and slow.",
        "Just to give your hands something to do while you think, reach out and open the middle drawer of the cabinet, right in the center there.",
        "Even if it feels like a small, insignificant step, just go ahead and open the cabinet's middle drawer now, before anything else.",
        "Once you've turned to face the cabinet, don't worry about the top or bottom—just focus on the middle one and pull it open.",
        "For a little bit of organization before we start, I need you to open the middle drawer of the cabinet for me, if you would."
    ],

    "put_the_bowl_on_the_stove": [
        "I see you've got that bowl in your hand—when you're ready, just set it right down on the stove surface, even if you're not gonna cook yet.",
        "Never mind the pots and pans for a moment; just take that bowl and place it directly on the stove, wherever there's space.",
        "To get a bit of room on the counter, why don't you go ahead and move that bowl onto the stove? Just set it there gently.",
        "After you've taken a quick look at the burners, just place the bowl you're holding onto the stove top for the time being.",
        "It's a real simple move—don't overthink it, just put the bowl down on the stove and leave it there for now.",
        "Before you start fiddling with the knobs or anything, could you just put that bowl on the stove for me? That'd be great.",
        "If the stove isn't cluttered, I'd like you to take that bowl and set it right on top of it, right in the center if possible.",
        "Nice and easy does it—just lift the bowl slightly and move it over to the stove, placing it down carefully on the surface.",
        "Once you've got a firm, steady grip on that bowl, I want you to reach over and place it on the stove, just leave it there.",
        "Even if it's just a temporary spot, put that bowl on the stove for now, just so it's out of the way for a second."
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "When you get a free moment later, could you take that wine bottle and place it up on top of the cabinet for safekeeping?",
        "I know we're not opening it right now, but if you could set that wine bottle on the cabinet's top surface, that'd help clear space.",
        "To give us a little more elbow room down here, why don't you lift that wine bottle up and put it on top of the cabinet?",
        "Carefully pick up that wine bottle—watch the neck—and just rest it up on the top of the cabinet, away from everything.",
        "Without moving any of the other things around down here, just reach up and put that wine bottle on the cabinet top.",
        "If you're done looking at the label or whatever, go ahead and place the wine bottle up high on top of the cabinet.",
        "For a tidier and less cluttered workspace, I'd like you to put that wine bottle on the very top of the cabinet.",
        "Take it slow, especially if it's heavy—just set that wine bottle up on the cabinet's upper surface, right in the corner.",
        "Just as a small organizing step to make things look nicer, please place the wine bottle on top of the cabinet now.",
        "Once you've got a steady hold on it, reach up and put the wine bottle on the cabinet top, making sure it won't roll off."
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "Here's what I need: first, pull open the top drawer of that cabinet, and then take the bowl and place it neatly inside.",
        "When you've got a second, just open up the top drawer for me, and once it's open, tuck that bowl away inside it.",
        "To keep things out of the way and stored properly, I want you to open the top drawer and then put the bowl right in.",
        "Don't worry about the other items in the drawer; just open the top one and place the bowl inside however it fits.",
        "Take a quick look at the stack of drawers there—go for the top one, open it up, and then put the bowl inside it.",
        "It's just a simple two-step task for you: open the top drawer, and then set the bowl down inside it carefully.",
        "If you want to get that bowl stored away for now, just open the top drawer and put it inside, nice and easy.",
        "Go ahead and reach for the upper drawer, slide it open, and then place the bowl into it gently, without forcing it.",
        "After you've got a good grip on the bowl with one hand, use the other to open the top drawer and slide the bowl in.",
        "Start by opening the top drawer, and once it's fully open and stable, go ahead and put the bowl inside it."
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "When it's convenient for you, could you just lift that bowl up and place it on top of the cabinet, out of the way?",
        "To free up some counter space, the best thing to do is set that bowl up on the cabinet's top surface for now.",
        "Without touching or rearranging anything else on the counter, just move that bowl up onto the top of the cabinet.",
        "Carefully lift that bowl with both hands if you need to, and just place it up on the cabinet top, right in the middle.",
        "If you're trying to tidy up the workspace a bit, putting that bowl up on top of the cabinet would really help.",
        "It's just a simple relocation task: I need you to set that bowl on the cabinet's upper surface, wherever it fits.",
        "Once you've got a steady, secure hold on that bowl, please place it on top of the cabinet for me, gently now.",
        "Even if it's just a temporary spot while we do other things, could you put that bowl on the cabinet top for now?",
        "After quickly checking that the cabinet top is clear enough, just place that bowl up there, nice and easy.",
        "Go ahead and move that bowl from where it is now onto the cabinet's top surface, careful not to scrape it."
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "Before you do anything fancy with the burners, just take that plate and push it toward the front edge of the stove.",
        "If it helps with reaching things later, just slide that plate forward so it ends up right at the front of the stove.",
        "Gently nudge that plate forward with your fingertips until it's positioned at the very front of the stove there.",
        "Without worrying about the other utensils or pots, just push that plate to the front-most part of the stove.",
        "Take it slow and steady—just push that plate until it's sitting right at the front edge of the stove area.",
        "For a cleaner and more accessible layout, why don't you move that plate forward to the front of the stove?",
        "Even a small adjustment is fine here—just push the plate a bit so it's at the front side of the stove top.",
        "When you have a good angle and can reach easily, just push that plate toward the front of the stove surface.",
        "Just reposition it slightly: give that plate a push so it ends up at the front of the stove top, aligned nicely.",
        "After a quick glance at the burner area to make sure it's clear, just push the plate to the front of the stove."
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "When you're ready to move on to the next ingredient, just take that cream cheese and put it into the bowl, nice and simple.",
        "Don't worry about turning on the stove or anything yet—just focus on putting that cream cheese in the bowl right there.",
        "To start prepping the ingredients for later, I need you to put that cream cheese into the bowl, just drop it in.",
        "Even if you're not planning to mix it up yet, go ahead and place that cream cheese in the bowl for now.",
        "Take that block of cream cheese and just drop it carefully into the bowl, making sure not to splash anything.",
        "It's just an ingredient prep step: put the cream cheese in the bowl and leave it there for the time being.",
        "After double-checking that you're holding the right item, just put that cream cheese into the bowl, gently now.",
        "Don't worry about the other items on the counter—just put that cream cheese in the bowl, that's all.",
        "If you have a steady grip on it, just place that cream cheese into the bowl gently, right in the center.",
        "For now, the task is simple: move that cream cheese from wherever it is into the bowl, nice and easy."
    ],

    "turn_on_the_stove": [
        "When you're ready to start cooking, just go ahead and turn on the stove for me, whichever knob you need.",
        "Even if you're not putting a pan on yet, I need you to just turn on the stove now, just to get it going.",
        "After checking that everything's clear around the burners, just switch the stove on, nice and simple.",
        "It's just a quick, simple step for you—turn on the stove and leave everything else exactly as it is.",
        "Without moving any of the pots or bowls around, just reach for the knob and turn on the stove now.",
        "If you've found the right control knob there, just go ahead and turn the stove on, to the low setting.",
        "Take a moment to focus on the stove, and when you're ready, just turn it on for me, please.",
        "For the very next step in the recipe, I simply need you to turn on the stove, that's all.",
        "Once you're set and have your bearings, just activate the stove by turning the knob.",
        "Go ahead and turn on the stove carefully, making sure you hear that click and see the flame or heat."
    ],

    "put_the_bowl_on_the_plate": [
        "When you're ready, I'd like you to just pick up that bowl and place it on top of the plate, nice and centered.",
        "Never mind the stove for a second—just take that bowl and set it right down on top of the plate there.",
        "To stack these items neatly for now, could you put that bowl on the plate, so it's sitting nicely?",
        "Carefully lift the bowl and place it onto the plate, making sure it sits flat and doesn't wobble.",
        "Even if it's just a temporary arrangement, just set the bowl on the plate for now, that's fine.",
        "Take it slow and steady, and just place the bowl on the plate without sliding it around too much.",
        "If you've got a good grip on the bowl, just put it onto the plate gently, nice and easy now.",
        "It's a simple arrangement step: I need you to place that bowl on the plate, right on top.",
        "Without changing anything else on the counter, just move the bowl so it's sitting on the plate.",
        "Go ahead and position that bowl on the plate, maybe try to get it centered if you can, please."
    ],

    "put_the_wine_bottle_on_the_rack": [
        "When you get a chance later, could you just take that wine bottle and place it on the rack for storage?",
        "To keep it stored neatly and safely, I'd like you to put that wine bottle onto the rack, right in one of the slots.",
        "Don't move any other items around—just take that wine bottle and set it on the rack, gently now.",
        "Carefully pick up that wine bottle and just rest it on the rack, making sure it's stable and won't fall.",
        "Even if you're not going to use it anytime soon, just place that wine bottle on the rack for now, out of the way.",
        "Take it slow with that bottle—just put it on the rack carefully so it doesn't tip over or anything.",
        "If the rack is within easy reach, just set that wine bottle on it gently, right in a secure spot.",
        "As a quick organizing step to tidy up, just place that wine bottle on the rack, please and thank you.",
        "After quickly checking the rack area to make sure it's clear, just put the wine bottle onto it, right there.",
        "Go ahead and position that wine bottle on the rack securely, so it's not going to roll off or get knocked."
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
