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


SPATIAL_PROMPT = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate":[
        "pick up the black bowl located between the plate and the ramekin and place it onto the plate",
        "pick up the black bowl positioned between the plate and the ramekin and place it on the plate",
        "grasp the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl that is between the plate and the ramekin and put it on the plate",
        "lift the black bowl between the plate and the ramekin and place it onto the plate",
        "pick up the black bowl situated between the plate and the ramekin and place it on the plate",
        "grasp the black bowl located between the plate and the ramekin and set it on the plate",
        "pick up the black bowl between the plate and the ramekin and move it onto the plate",
        "pick up the black bowl positioned between the plate and the ramekin and set it on the plate",
    ],
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate":[
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl beside the ramekin and place it onto the plate",
        "grasp the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl that is next to the ramekin and put it on the plate",
        "lift the black bowl next to the ramekin and place it onto the plate",
        "pick up the black bowl located next to the ramekin and place it on the plate",
        "grasp the black bowl beside the ramekin and set it on the plate",
        "pick up the black bowl next to the ramekin and move it onto the plate",
        "pick up the black bowl positioned next to the ramekin and set it on the plate",
    ],
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate":[
        "pick up the black bowl from the center of the table and place it on the plate",
        "pick up the black bowl at the table center and place it onto the plate",
        "grasp the black bowl from the center of the table and place it on the plate",
        "pick up the black bowl located at the center of the table and put it on the plate",
        "lift the black bowl from the table center and place it onto the plate",
        "pick up the black bowl situated at the center of the table and place it on the plate",
        "grasp the black bowl at the table center and set it on the plate",
        "pick up the black bowl from the center of the table and move it onto the plate",
        "pick up the black bowl positioned at the table center and set it on the plate",
    ],
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate":[
        "pick up the black bowl resting on the cookie box and place it on the plate",
        "pick up the black bowl on top of the cookie box and place it onto the plate",
        "grasp the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl that is on the cookie box and put it on the plate",
        "lift the black bowl on the cookie box and place it onto the plate",
        "pick up the black bowl located on the cookie box and place it on the plate",
        "grasp the black bowl on top of the cookie box and set it on the plate",
        "pick up the black bowl on the cookie box and move it onto the plate",
        "pick up the black bowl positioned on the cookie box and set it on the plate",
    ],
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate":[
        "pick up the black bowl from the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl inside the top drawer of the wooden cabinet and place it onto the plate",
        "grasp the black bowl from the wooden cabinet's top drawer and place it on the plate",
        "pick up the black bowl located in the top drawer of the wooden cabinet and put it on the plate",
        "lift the black bowl from inside the top drawer of the wooden cabinet and place it onto the plate",
        "pick up the black bowl situated in the wooden cabinet's top drawer and place it on the plate",
        "grasp the black bowl inside the top drawer of the wooden cabinet and set it on the plate",
        "pick up the black bowl from the top drawer of the wooden cabinet and move it onto the plate",
        "pick up the black bowl positioned in the top drawer of the wooden cabinet and set it on the plate",
    ],
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate":[
        "pick up the black bowl on top of the ramekin and place it on the plate",
        "pick up the black bowl resting on the ramekin and place it onto the plate",
        "grasp the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl that is on the ramekin and put it on the plate",
        "lift the black bowl from the ramekin and place it onto the plate",
        "pick up the black bowl located on the ramekin and place it on the plate",
        "grasp the black bowl resting on the ramekin and set it on the plate",
        "pick up the black bowl on the ramekin and move it onto the plate",
        "pick up the black bowl positioned on the ramekin and set it on the plate",
    ],
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate":[
        "pick up the black bowl beside the cookie box and place it on the plate",
        "grasp the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl located next to the cookie box and put it on the plate",
        "lift the black bowl next to the cookie box and place it onto the plate",
        "pick up the black bowl positioned beside the cookie box and place it on the plate",
        "grasp the black bowl beside the cookie box and set it on the plate",
        "pick up the black bowl next to the cookie box and move it onto the plate",
        "pick up the black bowl that is beside the cookie box and put it on the plate",
        "lift the black bowl located next to the cookie box and set it on the plate",
    ],
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate":[
        "pick up the black bowl on top of the stove and place it on the plate",
        "pick up the black bowl resting on the stove and place it onto the plate",
        "grasp the black bowl on the stove and place it on the plate",
        "pick up the black bowl that is on the stove and put it on the plate",
        "lift the black bowl from the stove and place it onto the plate",
        "pick up the black bowl located on the stove and place it on the plate",
        "grasp the black bowl resting on the stove and set it on the plate",
        "pick up the black bowl on the stove and move it onto the plate",
        "pick up the black bowl positioned on the stove and set it on the plate",
    ],
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate":[
        "pick up the black bowl beside the plate and place it on the plate",
        "grasp the black bowl next to the plate and place it on the plate",
        "pick up the black bowl located next to the plate and put it on the plate",
        "lift the black bowl next to the plate and place it onto the plate",
        "pick up the black bowl positioned beside the plate and place it on the plate",
        "grasp the black bowl beside the plate and set it on the plate",
        "pick up the black bowl next to the plate and move it onto the plate",
        "pick up the black bowl that is beside the plate and put it on the plate",
        "lift the black bowl located next to the plate and set it on the plate",
    ],
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate":[
        "pick up the black bowl on top of the wooden cabinet and place it on the plate",
        "pick up the black bowl resting on the wooden cabinet and place it onto the plate",
        "grasp the black bowl on the wooden cabinet and place it on the plate",
        "pick up the black bowl that is on the wooden cabinet and put it on the plate",
        "lift the black bowl from the wooden cabinet and place it onto the plate",
        "pick up the black bowl located on the wooden cabinet and place it on the plate",
        "grasp the black bowl resting on the wooden cabinet and set it on the plate",
        "pick up the black bowl on the wooden cabinet and move it onto the plate",
        "pick up the black bowl positioned on the wooden cabinet and set it on the plate",
    ],
}
OBJECT_PROMPT = {
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket":[
        "pick up the alphabet soup and put it in the basket",
        "grasp the alphabet soup and place it in the basket",
        "pick up the alphabet soup and move it into the basket",
        "lift the alphabet soup and place it in the basket",
        "pick up the alphabet soup and set it in the basket",
        "pick up the alphabet soup and put it into the basket",
        "grasp the alphabet soup and set it in the basket",
        "lift the alphabet soup and put it in the basket",
        "pick up the alphabet soup and place it into the basket",
    ],
    "pick_up_the_cream_cheese_and_place_it_in_the_basket":[
        "pick up the cream cheese and put it in the basket",
        "grasp the cream cheese and place it in the basket",
        "pick up the cream cheese and move it into the basket",
        "lift the cream cheese and place it in the basket",
        "pick up the cream cheese and set it in the basket",
        "pick up the cream cheese and put it into the basket",
        "grasp the cream cheese and set it in the basket",
        "lift the cream cheese and put it in the basket",
        "pick up the cream cheese and place it into the basket",
    ],
    "pick_up_the_salad_dressing_and_place_it_in_the_basket":[
        "pick up the salad dressing and put it in the basket",
        "grasp the salad dressing and place it in the basket",
        "pick up the salad dressing and move it into the basket",
        "lift the salad dressing and place it in the basket",
        "pick up the salad dressing and set it in the basket",
        "pick up the salad dressing and put it into the basket",
        "grasp the salad dressing and set it in the basket",
        "lift the salad dressing and put it in the basket",
        "pick up the salad dressing and place it into the basket",
    ],
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket":[
        "pick up the bbq sauce and put it in the basket",
        "grasp the bbq sauce and place it in the basket",
        "pick up the bbq sauce and move it into the basket",
        "lift the bbq sauce and place it in the basket",
        "pick up the bbq sauce and set it in the basket",
        "pick up the bbq sauce and put it into the basket",
        "grasp the bbq sauce and set it in the basket",
        "lift the bbq sauce and put it in the basket",
        "pick up the bbq sauce and place it into the basket",
    ],
    "pick_up_the_ketchup_and_place_it_in_the_basket":[
         "pick up the ketchup and put it in the basket",
        "grasp the ketchup and place it in the basket",
        "pick up the ketchup and move it into the basket",
        "lift the ketchup and place it in the basket",
        "pick up the ketchup and set it in the basket",
        "pick up the ketchup and put it into the basket",
        "grasp the ketchup and set it in the basket",
        "lift the ketchup and put it in the basket",
        "pick up the ketchup and place it into the basket",
    ],
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket":[
        "pick up the tomato sauce and put it in the basket",
        "grasp the tomato sauce and place it in the basket",
        "pick up the tomato sauce and move it into the basket",
        "lift the tomato sauce and place it in the basket",
        "pick up the tomato sauce and set it in the basket",
        "pick up the tomato sauce and put it into the basket",
        "grasp the tomato sauce and set it in the basket",
        "lift the tomato sauce and put it in the basket",
        "pick up the tomato sauce and place it into the basket",
    ],
    "pick_up_the_butter_and_place_it_in_the_basket":[
        "pick up the butter and put it in the basket",
        "grasp the butter and place it in the basket",
        "pick up the butter and move it into the basket",
        "lift the butter and place it in the basket",
        "pick up the butter and set it in the basket",
        "pick up the butter and put it into the basket",
        "grasp the butter and set it in the basket",
        "lift the butter and put it in the basket",
        "pick up the butter and place it into the basket",
    ],
    "pick_up_the_milk_and_place_it_in_the_basket":[
        "pick up the milk and put it in the basket",
        "grasp the milk and place it in the basket",
        "pick up the milk and move it into the basket",
        "lift the milk and place it in the basket",
        "pick up the milk and set it in the basket",
        "pick up the milk and put it into the basket",
        "grasp the milk and set it in the basket",
        "lift the milk and put it in the basket",
        "pick up the milk and place it into the basket",
    ],
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket":[
        "pick up the chocolate pudding and put it in the basket",
        "grasp the chocolate pudding and place it in the basket",
        "pick up the chocolate pudding and move it into the basket",
        "lift the chocolate pudding and place it in the basket",
        "pick up the chocolate pudding and set it in the basket",
        "pick up the chocolate pudding and put it into the basket",
        "grasp the chocolate pudding and set it in the basket",
        "lift the chocolate pudding and put it in the basket",
        "pick up the chocolate pudding and place it into the basket",
    ],
    "pick_up_the_orange_juice_and_place_it_in_the_basket":[
        "pick up the orange juice and put it in the basket",
        "grasp the orange juice and place it in the basket",
        "pick up the orange juice and move it into the basket",
        "lift the orange juice and place it in the basket",
        "pick up the orange juice and set it in the basket",
        "pick up the orange juice and put it into the basket",
        "grasp the orange juice and set it in the basket",
        "lift the orange juice and put it in the basket",
        "pick up the orange juice and place it into the basket",
    ],
}
GOAL_PROMPT = {
    "open_the_middle_drawer_of_the_cabinet": [
        "open up the middle drawer of the cabinet",
        "open the cabinet's middle drawer",
        "pull open the middle drawer of the cabinet",
        "open the middle drawer in the cabinet",
        "open the middle cabinet drawer",
        "open the cabinet middle drawer",
        "pull the middle drawer of the cabinet open",
        "open up the cabinet's middle drawer",
        "open the middle drawer located in the cabinet",
    ],

    "put_the_bowl_on_the_stove": [
        "place the bowl on the stove",
        "set the bowl on the stove",
        "put the bowl onto the stove",
        "move the bowl onto the stove",
        "place the bowl on top of the stove",
        "set the bowl onto the stove",
        "move the bowl on the stove",
        "put the bowl on top of the stove",
        "position the bowl on the stove",
    ],

    "put_the_wine_bottle_on_top_of_the_cabinet": [
        "place the wine bottle on top of the cabinet",
        "set the wine bottle on top of the cabinet",
        "put the wine bottle onto the top of the cabinet",
        "move the wine bottle onto the cabinet top",
        "place the wine bottle on the cabinet top",
        "set the wine bottle onto the cabinet",
        "position the wine bottle on top of the cabinet",
        "put the wine bottle on the cabinet top",
        "move the wine bottle to the top of the cabinet",
    ],

    "open_the_top_drawer_and_put_the_bowl_inside": [
        "open the top drawer and place the bowl inside",
        "open up the top drawer and put the bowl in",
        "open the top drawer and set the bowl inside",
        "open the top drawer and move the bowl into it",
        "open the upper drawer and place the bowl inside",
        "open the top drawer and put the bowl into it",
        "open up the top drawer and set the bowl inside",
        "open the top drawer then place the bowl inside",
        "open the upper drawer and put the bowl inside",
    ],

    "put_the_bowl_on_top_of_the_cabinet": [
        "place the bowl on top of the cabinet",
        "set the bowl on top of the cabinet",
        "put the bowl onto the top of the cabinet",
        "move the bowl onto the cabinet top",
        "place the bowl on the cabinet top",
        "set the bowl onto the cabinet",
        "position the bowl on top of the cabinet",
        "put the bowl on the cabinet top",
        "move the bowl to the top of the cabinet",
    ],

    "push_the_plate_to_the_front_of_the_stove": [
        "push the plate toward the front of the stove",
        "slide the plate to the front of the stove",
        "push the plate to the stove's front",
        "move the plate to the front of the stove",
        "push the plate forward to the front of the stove",
        "slide the plate toward the stove front",
        "move the plate toward the front of the stove",
        "push the plate forward on the stove",
        "slide the plate to the stove front",
    ],

    "put_the_cream_cheese_in_the_bowl": [
        "place the cream cheese in the bowl",
        "put the cream cheese into the bowl",
        "set the cream cheese inside the bowl",
        "move the cream cheese into the bowl",
        "place the cream cheese inside the bowl",
        "drop the cream cheese into the bowl",
        "put the cream cheese inside the bowl",
        "set the cream cheese into the bowl",
        "move the cream cheese into the bowl",
    ],

    "turn_on_the_stove": [
        "switch on the stove",
        "activate the stove",
        "turn the stove on",
        "power on the stove",
        "start the stove",
        "switch the stove on",
        "activate the stove burner",
        "turn on the stove burner",
        "power up the stove",
    ],

    "put_the_bowl_on_the_plate": [
        "place the bowl on the plate",
        "set the bowl on the plate",
        "put the bowl onto the plate",
        "move the bowl onto the plate",
        "position the bowl on the plate",
        "set the bowl onto the plate",
        "move the bowl to the plate",
        "put the bowl on top of the plate",
        "place the bowl onto the plate",
    ],

    "put_the_wine_bottle_on_the_rack": [
        "place the wine bottle on the rack",
        "set the wine bottle on the rack",
        "put the wine bottle onto the rack",
        "move the wine bottle onto the rack",
        "position the wine bottle on the rack",
        "set the wine bottle onto the rack",
        "place the wine bottle on the bottle rack",
        "move the wine bottle to the rack",
        "put the wine bottle on the wine rack",
    ],
}
LONG_PROMPT = {
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": [
        "place both the alphabet soup and the tomato sauce into the basket",
        "put both the alphabet soup and the tomato sauce in the basket",
        "put the alphabet soup and the tomato sauce together in the basket",
        "move both the alphabet soup and the tomato sauce into the basket",
        "set both the alphabet soup and the tomato sauce in the basket",
        "place the alphabet soup and the tomato sauce inside the basket",
        "put both the tomato sauce and the alphabet soup into the basket",
        "move the alphabet soup along with the tomato sauce into the basket",
        "set the alphabet soup and the tomato sauce together in the basket",
    ],

    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": [
        "place both the cream cheese box and the butter into the basket",
        "put both the cream cheese box and the butter in the basket",
        "put the cream cheese box and the butter together in the basket",
        "move both the cream cheese box and the butter into the basket",
        "set both the cream cheese box and the butter in the basket",
        "place the cream cheese box and the butter inside the basket",
        "put both the butter and the cream cheese box into the basket",
        "move the cream cheese box along with the butter into the basket",
        "set the cream cheese box and the butter together in the basket",
    ],

    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": [
        "switch on the stove and place the moka pot on it",
        "turn on the stove and put the moka pot on it",
        "activate the stove and put the moka pot on top of it",
        "turn the stove on and place the moka pot onto it",
        "switch the stove on and set the moka pot on it",
        "power on the stove and put the moka pot on it",
        "activate the stove and place the moka pot on it",
        "turn on the stove then put the moka pot on it",
        "switch on the stove and move the moka pot onto it",
    ],

    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": [
        "place the black bowl into the bottom drawer of the cabinet and close it",
        "put the black bowl in the bottom drawer of the cabinet and close it",
        "put the black bowl inside the cabinet's bottom drawer and close it",
        "move the black bowl into the bottom drawer of the cabinet and close it",
        "set the black bowl in the bottom drawer of the cabinet and close it",
        "place the black bowl inside the bottom drawer and close it",
        "put the black bowl into the cabinet's bottom drawer and close it",
        "move the black bowl inside the bottom drawer and close it",
        "set the black bowl in the cabinet's bottom drawer and close it",
    ],

    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": [
        "place the white mug on the left plate and place the yellow and white mug on the right plate",
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
        "set the white mug on the left plate and set the yellow and white mug on the right plate",
        "put the white mug onto the left plate and the yellow and white mug onto the right plate",
        "move the white mug to the left plate and move the yellow and white mug to the right plate",
        "place the white mug on the left plate and the yellow and white mug on the right plate",
        "set the white mug onto the left plate and place the yellow and white mug onto the right plate",
        "put the white mug on the left plate then put the yellow and white mug on the right plate",
        "move the white mug onto the left plate and set the yellow and white mug on the right plate",
    ],

    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": [
        "pick up the book and put it into the back compartment of the caddy",
        "pick up the book and place it in the back compartment of the caddy",
        "grasp the book and place it in the caddy's back compartment",
        "pick up the book and move it into the back compartment of the caddy",
        "lift the book and place it in the back compartment of the caddy",
        "pick up the book and set it inside the back compartment of the caddy",
        "grasp the book and put it in the back compartment of the caddy",
        "pick up the book and place it inside the caddy's back compartment",
        "lift the book and move it into the back compartment of the caddy",
    ],

    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": [
        "place the white mug on the plate and place the chocolate pudding to the right of the plate",
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
        "set the white mug on the plate and set the chocolate pudding to the right of the plate",
        "put the white mug onto the plate and move the chocolate pudding to the right of the plate",
        "move the white mug to the plate and put the chocolate pudding to the right of the plate",
        "place the white mug on the plate and move the chocolate pudding to the plate's right side",
        "set the white mug onto the plate and place the chocolate pudding to the right of the plate",
        "put the white mug on the plate then put the chocolate pudding to the right of the plate",
        "move the white mug onto the plate and set the chocolate pudding to the right of the plate",
    ],

    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": [
        "place both the alphabet soup and the cream cheese box into the basket",
        "put both the alphabet soup and the cream cheese box in the basket",
        "put the alphabet soup and the cream cheese box together in the basket",
        "move both the alphabet soup and the cream cheese box into the basket",
        "set both the alphabet soup and the cream cheese box in the basket",
        "place the alphabet soup and the cream cheese box inside the basket",
        "put both the cream cheese box and the alphabet soup into the basket",
        "move the alphabet soup along with the cream cheese box into the basket",
        "set the alphabet soup and the cream cheese box together in the basket",
    ],

    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": [
        "place both moka pots on the stove",
        "put both moka pots on the stove",
        "set both moka pots on the stove",
        "put the two moka pots onto the stove",
        "move both moka pots onto the stove",
        "place the two moka pots on the stove",
        "set the moka pots on the stove",
        "move both moka pots to the stove",
        "put both of the moka pots on the stove",
    ],

    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": [
        "place the yellow and white mug into the microwave and close it",
        "put the yellow and white mug in the microwave and close it",
        "put the yellow and white mug inside the microwave and close it",
        "move the yellow and white mug into the microwave and close it",
        "set the yellow and white mug in the microwave and close it",
        "place the yellow and white mug inside the microwave and close it",
        "put the yellow and white mug into the microwave then close it",
        "move the yellow and white mug inside the microwave and close it",
        "set the yellow and white mug into the microwave and close it",
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
    task_suite_name: str = "all_wo_90" # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, all, all_wo_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    model_name: str = "pi0_libero"  # Name for save
    video_out_path: str = "./sim_output/libero_shortcut"  # Path to save videos

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

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
        NEW_PROMPT = SPATIAL_PROMPT
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
        NEW_PROMPT = OBJECT_PROMPT
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
        NEW_PROMPT = GOAL_PROMPT
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
        NEW_PROMPT = LONG_PROMPT
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
        NEW_PROMPT = {}
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
        new_prompt_list.append(task_description)
        
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
