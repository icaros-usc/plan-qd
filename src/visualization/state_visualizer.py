import copy
import math
import os

import numpy as np
import pygame
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.pygame_utils import (
    MultiFramePygameImage, blit_on_new_surface_of_size,
    run_static_resizeable_window, scale_surface_by_factor, vstack_surfaces)
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

_current_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."

DATA_DIR = _current_dir + "/data/"
GRAPHICS_DIR = os.path.join(DATA_DIR, "graphics")

SPRITE_LENGTH = 30  # length of each sprite square
HUD_HEIGHT = 140  # height of the HUD
EMPTY = " "
COUNTER = "X"
ONION_DISPENSER = "O"
TOMATO_DISPENSER = "T"
POT = "P"
DISH_DISPENSER = "D"
SERVING_LOC = "S"
SINK = "W"
BOARD_KNIFE = "B"
STEAK_DISPENSER = "M"
GRILL = "G"
CHICKEN_DISPENSER = "C"
USELESS = "U"
"""
SteakhouseStateVisualizer class inherits from StateVisualizer with added items to be displayed
"""


class SteakhouseStateVisualizer(StateVisualizer):
    TERRAINS_IMG = MultiFramePygameImage(
        os.path.join(GRAPHICS_DIR, "terrain.png"),
        os.path.join(GRAPHICS_DIR, "terrain.json"),
    )
    OBJECTS_IMG = MultiFramePygameImage(
        os.path.join(GRAPHICS_DIR, "objects.png"),
        os.path.join(GRAPHICS_DIR, "objects.json"),
    )
    CHEFS_IMG = MultiFramePygameImage(
        os.path.join(GRAPHICS_DIR, "chefs.png"),
        os.path.join(GRAPHICS_DIR, "chefs.json"),
    )

    TILE_TO_FRAME_NAME = {
        EMPTY: "floor",
        COUNTER: "counter",
        ONION_DISPENSER: "onions",
        TOMATO_DISPENSER: "tomatoes",
        POT: "pot",
        DISH_DISPENSER: "dishes",
        SERVING_LOC: "serve",
        SINK: "sink",
        BOARD_KNIFE: "board_knife",
        STEAK_DISPENSER: "steaks",
        GRILL: "grill",
        CHICKEN_DISPENSER: "chickens",
        USELESS: "tomatoes", #for now, we draw useless as tomato
    }
    # def __init__(self, **kwargs):
    #     params = copy.deepcopy(self.DEFAULT_VALUES)
    #     params.update(kwargs)
    #     self.configure(**params)
    #     self.reload_fonts()
    #     self.TERRAINS_IMG =TERRAINS_IMG

    @staticmethod
    def _frame_name(ingredients_names, status):
        # num_meat = ingredients_names.count("meat")
        if len(ingredients_names) == 1:
            ingredients_names = ingredients_names[0]
        else:
            ingredients_names = "_".join(ingredients_names)
        return "%s_%s" % (
            ingredients_names,
            status,
            # num_meat,
        )

    def configure(self, **kwargs):
        SteakhouseStateVisualizer._check_config_validity(kwargs)
        for param_name, param_value in copy.deepcopy(kwargs).items():
            setattr(self, param_name, param_value)

    @staticmethod
    def default_hud_data(state, **kwargs):
        result = {
            # "timestep": state.timestep,
            # "completed_orders": [r for r in state._complete_orders],
            "served_dish": [r for r in state.bonus_orders],
            # "order_list": [r for r in state.order_list],
            "orders": [r for r in state._order_display_list],
        }
        result.update(copy.deepcopy(kwargs))
        return result

    @staticmethod
    def default_hud_data_from_trajectories(trajectories, trajectory_idx=0):
        scores = cumulative_rewards_from_rew_list(
            trajectories["ep_rewards"][trajectory_idx]
        )
        return [
            SteakhouseStateVisualizer.default_hud_data(state, score=scores[i])
            for i, state in enumerate(trajectories["ep_states"][trajectory_idx])
        ]

    def display_rendered_trajectory(
        self,
        trajectories,
        trajectory_idx=0,
        hud_data=None,
        action_probs=None,
        img_directory_path=None,
        img_extension=".png",
        img_prefix="",
        ipython_display=True,
    ):
        """
        saves images of every timestep from trajectory in img_directory_path (or temporary directory if not path is not specified)
        trajectories (dict): trajectories dict, same format as used by AgentEvaluator
        trajectory_idx(int): index of trajectory in case of multiple trajectories inside trajectories param
        img_path (str): img_directory_path - path to directory where consequtive images will be saved
        ipython_display(bool): if True render slider with rendered states
        hud_data(list(dict)): hud data for every timestep
        action_probs(list(list((list(float))))): action probs for every player and timestep acessed in the way action_probs[timestep][player][action]
        """
        states = trajectories["ep_states"][trajectory_idx]
        grid = trajectories["mdp_params"][trajectory_idx]["terrain"]
        if hud_data is None:
            if self.is_rendering_hud:
                hud_data = SteakhouseStateVisualizer.default_hud_data_from_trajectories(
                    trajectories, trajectory_idx
                )
            else:
                hud_data = [None] * len(states)

        if action_probs is None:
            action_probs = [None] * len(states)

        if not img_directory_path:
            img_directory_path = generate_temporary_file_path(
                prefix="steakhouse_visualized_trajectory", extension=""
            )
        os.makedirs(img_directory_path, exist_ok=True)
        img_pathes = []
        for i, state in enumerate(states):
            img_name = img_prefix + str(i) + img_extension
            img_path = os.path.join(img_directory_path, img_name)
            img_pathes.append(
                self.display_rendered_state(
                    state=state,
                    hud_data=hud_data[i],
                    action_probs=action_probs[i],
                    grid=grid,
                    img_path=img_path,
                    ipython_display=False,
                    window_display=False,
                )
            )

        if ipython_display:
            return show_ipython_images_slider(img_pathes, "timestep")

        return img_directory_path

    @property
    def scale_by_factor(self):
        return self.tile_size / SteakhouseStateVisualizer.UNSCALED_TILE_SIZE

    def render_state(self, state, grid, hud_data=None, action_probs=None):
        """
        returns surface with rendered game state scaled to selected size,
        decoupled from display_rendered_state function to make testing easier
        """
        pygame.init()
        pygame.font.init()
        grid = grid or self.grid
        assert grid
        grid_surface = pygame.surface.Surface(self._unscaled_grid_pixel_size(grid))
        self._render_grid(grid_surface, grid)
        self._render_players(grid_surface, state.players)
        self._render_objects(grid_surface, state.objects, grid)
        if self.scale_by_factor != 1:
            grid_surface = scale_surface_by_factor(grid_surface, self.scale_by_factor)

        # render text after rescaling as text looks bad when is rendered small resolution and then rescalled to bigger one
        if self.is_rendering_cooking_timer:
            self._render_cooking_timers(grid_surface, state.objects, grid)

        # arrows does not seem good when rendered in very small resolution
        if self.is_rendering_action_probs and action_probs is not None:
            self._render_actions_probs(grid_surface, state.players, action_probs)

        if self.is_rendering_hud and hud_data:
            hud_width = self.width or grid_surface.get_width()
            hud_surface = pygame.surface.Surface(
                (hud_width, self._calculate_hud_height(hud_data))
            )
            hud_surface.fill(self.background_color)
            self._render_hud_data(hud_surface, hud_data)
            rendered_surface = vstack_surfaces(
                [hud_surface, grid_surface], self.background_color
            )
        else:
            hud_width = None
            rendered_surface = grid_surface

        result_surface_size = (
            self.width or rendered_surface.get_width(),
            self.height or rendered_surface.get_height(),
        )

        if result_surface_size != rendered_surface.get_size():
            result_surface = blit_on_new_surface_of_size(
                rendered_surface,
                result_surface_size,
                background_color=self.background_color,
            )
        else:
            result_surface = rendered_surface

        return result_surface

    def _render_players(self, surface, players):
        def chef_frame_name(direction_name, held_object_name):
            frame_name = direction_name
            if held_object_name:
                frame_name += "-" + held_object_name
            return frame_name

        def hat_frame_name(direction_name, player_color_name):
            return "%s-%shat" % (direction_name, player_color_name)

        for player_num, player in enumerate(players):
            player_color_name = self.player_colors[player_num]
            direction_name = Direction.DIRECTION_TO_NAME[player.orientation]

            held_obj = player.held_object
            if held_obj is None:
                held_object_name = ""
            else:
                if held_obj.name == "soup":
                    if "onion" in held_obj.ingredients:
                        held_object_name = "soup-onion"
                    else:
                        held_object_name = "soup-tomato"
                elif held_obj.name == "boiled_chicken":
                    held_object_name = "chicken-dish"
                elif held_obj.name == "boiled_chicken_onion":
                    held_object_name = "chicken-onion-dish"

                elif held_obj.name == "steak":
                    held_object_name = "steak-dish"

                elif held_obj.name == "steak_onion":
                    held_object_name = "steak-onion-dish"

                else:
                    held_object_name = held_obj.name

            self.CHEFS_IMG.blit_on_surface(
                surface,
                self._position_in_unscaled_pixels(player.position),
                chef_frame_name(direction_name, held_object_name),
            )
            self.CHEFS_IMG.blit_on_surface(
                surface,
                self._position_in_unscaled_pixels(player.position),
                hat_frame_name(direction_name, player_color_name),
            )

    def _render_objects(self, surface, objects, grid):
        def render_steak(surface, obj, grid):
            (x_pos, y_pos) = obj.position
            if grid[y_pos][x_pos] == GRILL:
                if obj.is_ready:
                    status = "cooked"
                else:
                    status = "idle"
            else:  # grid[x][y] != POT
                status = "done"
            frame_name = SteakhouseStateVisualizer._frame_name(obj.ingredients, status)
            self.OBJECTS_IMG.blit_on_surface(
                surface,
                self._position_in_unscaled_pixels(obj.position),
                frame_name,
            )

        def render_chicken(surface, obj, grid):
            (x_pos, y_pos) = obj.position
            if grid[y_pos][x_pos] == POT:
                if obj.is_ready:
                    status = "cooked"
                else:
                    status = "idle"
            else:
                status = "done"
            frame_name = SteakhouseStateVisualizer._frame_name(obj.ingredients, status)
            self.OBJECTS_IMG.blit_on_surface(
                surface,
                self._position_in_unscaled_pixels(obj.position),
                frame_name,
            )

        def render_garnish(surface, obj, grid):
            (x_pos, y_pos) = obj.position
            if grid[y_pos][x_pos] == BOARD_KNIFE:
                if obj.is_ready:
                    status = "chopped"
                else:
                    status = "idle"
            frame_name = SteakhouseStateVisualizer._frame_name(obj.ingredients, status)
            self.OBJECTS_IMG.blit_on_surface(
                surface,
                self._position_in_unscaled_pixels(obj.position),
                frame_name,
            )

        for obj in objects.values():
            if obj.name == "steak":
                render_steak(surface, obj, grid)
            elif obj.name == "boiled_chicken":
                render_chicken(surface, obj, grid)
            elif obj.name == "garnish":
                render_garnish(surface, obj, grid)
            else:
                if obj.name == "steak_onion":
                    frame_name = "steak-onion-dish"
                elif obj.name == "boiled_chicken_onion":
                    frame_name = "chicken-onion-dish"
                else:
                    frame_name = obj.name
                self.OBJECTS_IMG.blit_on_surface(
                    surface,
                    self._position_in_unscaled_pixels(obj.position),
                    frame_name,
                )

    def _render_cooking_timers(self, surface, objects, grid):
        for key, obj in objects.items():
            (x_pos, y_pos) = obj.position
            if (obj.name == "steak" and grid[y_pos][x_pos] == GRILL) or (
                obj.name == "boiled_chicken" and grid[y_pos][x_pos] == POT
            ):
                if obj._cooking_tick != -1 and (
                    obj._cooking_tick <= obj.cook_time or self.show_timer_when_cooked
                ):
                    text_surface = self.cooking_timer_font.render(
                        str(obj._cooking_tick / 10),
                        True,
                        self.cooking_timer_font_color,
                    )
                    (tile_pos_x, tile_pos_y) = self._position_in_scaled_pixels(
                        obj.position
                    )

                    # calculate font position to be in center on x axis, and 0.9 from top on y axis
                    font_position = (
                        tile_pos_x
                        + int((self.tile_size - text_surface.get_width()) * 0.5),
                        tile_pos_y
                        + int((self.tile_size - text_surface.get_height()) * 0.9),
                    )
                    surface.blit(text_surface, font_position)
            if (obj.name == "garnish" and grid[y_pos][x_pos] == BOARD_KNIFE) or (
                obj.name == "clean_plate" and grid[y_pos][x_pos] == SINK
            ):
                if obj._cooking_tick != -1 and (
                    obj._cooking_tick <= obj.cook_time or self.show_timer_when_cooked
                ):
                    text_surface = self.cooking_timer_font.render(
                        str(obj._cooking_tick),
                        True,
                        self.cooking_timer_font_color,
                    )
                    (tile_pos_x, tile_pos_y) = self._position_in_scaled_pixels(
                        obj.position
                    )

                    # calculate font position to be in center on x axis, and 0.9 from top on y axis
                    font_position = (
                        tile_pos_x
                        + int((self.tile_size - text_surface.get_width()) * 0.5),
                        tile_pos_y
                        + int((self.tile_size - text_surface.get_height()) * 0.9),
                    )
                    surface.blit(text_surface, font_position)

    @staticmethod
    def _check_config_validity(config):
        assert set(config.keys()).issubset(
            set(SteakhouseStateVisualizer.DEFAULT_VALUES.keys())
        )

    def _render_grid(self, surface, grid):
        for y_tile, row in enumerate(grid):
            for x_tile, tile in enumerate(row):
                self.TERRAINS_IMG.blit_on_surface(
                    surface,
                    self._position_in_unscaled_pixels((x_tile, y_tile)),
                    SteakhouseStateVisualizer.TILE_TO_FRAME_NAME[tile],
                )

    def framename2ingradient(self, dish_name):
        # map dish_name to its ingredient, for example, steak_onion_dish to {"ingredients" : ["meat","onion"]},
        if dish_name == "steak_dish":
            return {"ingredients": ["meat"]}
        elif dish_name == "boiled_chicken_dish":
            return {"ingredients": ["chicken"]}
        elif dish_name == "steak_onion_dish":
            return {"ingredients": ["meat", "onion"]}
        elif dish_name == "boiled_chicken_onion_dish":
            return {"ingredients": ["chicken", "onion"]}
        elif dish_name == "steak_dish_tick":
            return {"ingredients": ["meat", "tick"]}
        elif dish_name == "boiled_chicken_dish_tick":
            return {"ingredients": ["chicken", "tick"]}
        elif dish_name == "steak_onion_dish_tick":
            return {"ingredients": ["meat", "onion", "tick"]}
        elif dish_name == "boiled_chicken_onion_dish_tick":
            return {"ingredients": ["chicken", "onion", "tick"]}

    def ingradient2framename(self, ingradient):
        # map ingradient to its dish_name, for example, {"ingredients" : ["meat","onion"]} to steak_onion_dish
        if ingradient == ["meat"]:
            return "steak_dish"
        elif ingradient == ["chicken"]:
            return "boiled_chicken_dish"
        elif ingradient == ["meat", "onion"]:
            return "steak_onion_dish"
        elif ingradient == ["chicken", "onion"]:
            return "boiled_chicken_onion_dish"
        elif ingradient == ["meat", "tick"]:
            return "steak_dish_tick"
        elif ingradient == ["chicken", "tick"]:
            return "boiled_chicken_dish_tick"
        elif ingradient == ["meat", "onion", "tick"]:
            return "steak_onion_dish_tick"
        elif ingradient == ["chicken", "onion", "tick"]:
            return "boiled_chicken_onion_dish_tick"

    def _render_hud_data(self, surface, hud_data):
        def hud_text_position(line_num):
            return (
                self.hud_margin_left,
                self.hud_margin_top + self.hud_line_height * line_num,
            )

        def hud_recipes_position(text_surface, text_surface_position):
            (text_surface_x, text_surface_y) = text_surface_position
            return (text_surface_x + text_surface.get_width(), text_surface_y)

        def get_hud_recipes_surface(orders_dicts):
            order_width = order_height = self.hud_order_size
            scaled_order_size = (order_width, order_width)
            orders_surface_height = order_height
            orders_surface_width = (
                len(orders_dicts) * order_width
                + (len(orders_dicts) - 1) * self.hud_distance_between_orders
            )
            unscaled_order_size = (
                self.UNSCALED_TILE_SIZE,
                self.UNSCALED_TILE_SIZE,
            )

            recipes_surface = pygame.surface.Surface(
                (orders_surface_width, orders_surface_height)
            )
            recipes_surface.fill(self.background_color)
            next_surface_x = 0
            for order_dict in orders_dicts:
                frame_name = SteakhouseStateVisualizer._frame_name(
                    order_dict["ingredients"], "done"
                )
                unscaled_order_surface = pygame.surface.Surface(unscaled_order_size)
                unscaled_order_surface.fill(self.background_color)
                self.OBJECTS_IMG.blit_on_surface(
                    unscaled_order_surface, (0, 0), frame_name
                )
                if scaled_order_size == unscaled_order_size:
                    scaled_order_surface = unscaled_order_surface
                else:
                    scaled_order_surface = pygame.transform.scale(
                        unscaled_order_surface, (order_width, order_width)
                    )
                recipes_surface.blit(scaled_order_surface, (next_surface_x, 0))
                next_surface_x += order_width + self.hud_distance_between_orders
            return recipes_surface

        for hud_line_num, (key, value) in enumerate(self._sorted_hud_items(hud_data)):
            hud_text = self._key_to_hud_text(key)
            if key not in [
                "all_orders",
                "start_all_orders",
                "start_bonus_orders",
                "order_list",
                "orders",
                "served_dish",
            ]:
                hud_text += str(value)

            text_surface = self.hud_font.render(hud_text, True, self.hud_font_color)
            text_surface_position = hud_text_position(hud_line_num)
            surface.blit(text_surface, text_surface_position)

            if (
                key
                in [
                    "all_orders",
                    "start_all_orders",
                    "start_bonus_orders",
                ]
                and value
            ):
                recipes_surface_position = hud_recipes_position(
                    text_surface, text_surface_position
                )
                recipes_surface = get_hud_recipes_surface(value)
                assert (
                    recipes_surface.get_width() + text_surface.get_width()
                    <= surface.get_width()
                ), "surface width is too small to fit recipes in single line"
                surface.blit(recipes_surface, recipes_surface_position)
            if key in ["order_list", "orders", "served_dish"] and value:
                recipes_surface_position = hud_recipes_position(
                    text_surface, text_surface_position
                )
                ingradients = []
                for dish_name in value:
                    ingradient = self.framename2ingradient(dish_name)
                    ingradients.append(ingradient)
                recipes_surface = get_hud_recipes_surface(ingradients)
                if recipes_surface.get_width() + text_surface.get_width() <= surface.get_width():
                    surface.blit(recipes_surface, recipes_surface_position)
                else:
                    print(f"RYAN: recipes_surface.get_width() + text_surface.get_width() > surface.get_width()")
                #assert (
                #    recipes_surface.get_width() + text_surface.get_width()
                #    <= surface.get_width()
                #), "surface width is too small to fit recipes in single line"
                surface.blit(recipes_surface, recipes_surface_position)

    # based on agent knowledge base, render tiles that agent cant see with fog
    def render_fog(self, surface, env, limited_vision_agent):
        if (
            limited_vision_agent.__class__.__name__ == "SteakLimitVisionHumanModel"
            and limited_vision_agent.vision_limit == True
        ):
            agent_knowledge_base = limited_vision_agent.knowledge_base
            grid = env.mdp.terrain_mtx
            state = env.state
            for y, row in enumerate(grid):
                for x, tile in enumerate(row):
                    in_view = self._check_viewpoint(
                        state.players[limited_vision_agent.agent_index].position,
                        state.players[limited_vision_agent.agent_index].orientation,
                        x,
                        y,
                        view_angle=limited_vision_agent.vision_bound,
                    )
                    if not in_view:
                        curr_pos = pygame.Rect(
                            x * SPRITE_LENGTH,
                            y * SPRITE_LENGTH + HUD_HEIGHT,
                            SPRITE_LENGTH,
                            SPRITE_LENGTH,
                        )
                        fog_pgobj = pygame.Surface((SPRITE_LENGTH, SPRITE_LENGTH))
                        fog_pgobj.fill((0, 0, 0))
                        fog_pgobj.set_alpha(100)
                        surface.blit(fog_pgobj, curr_pos)

                        # self.TERRAINS_IMG.blit_on_surface(
                        #     surface,
                        #     self._position_in_unscaled_pixels((x_tile, y_tile)),
                        #     "fog",
                        # )
        return surface

    def _check_viewpoint(self, player_pos, player_ori, x, y, view_angle=145):
        ori = Direction.DIRECTION_TO_INDEX[player_ori]
        # center_pt = np.array(player_pos)
        # if ori == 0: # north
        #     center_pt[1] += 1
        # elif ori == 2: # east
        #     center_pt[0] -= 1
        # elif ori == 1: # south
        #     center_pt[1] -= 1
        # elif ori == 3: # west
        #     center_pt[0] += 1

        item_ang = np.arctan2((y - player_pos[1]), (x - player_pos[0])) * 180 / np.pi

        # if x == center_pt[0] and y == center_pt[1]:
        #     return False
        if (x, y) == player_pos:
            return True
        if ori == 1:  # north
            if item_ang <= (90 + (view_angle / 2)) and (
                item_ang >= (90 - (view_angle / 2))
            ):
                return True
            if (
                player_pos[1] == y
                and ((player_pos[0] == x - 1) or (player_pos[0] == x + 1))
                and view_angle >= 120
            ):
                return True
        elif ori == 2:  # east
            if item_ang <= (0 + (view_angle / 2)) and (
                item_ang >= (0 - (view_angle / 2))
            ):
                return True
            if (
                player_pos[0] == x
                and ((player_pos[1] == y - 1) or (player_pos[1] == y + 1))
                and view_angle >= 120
            ):
                return True
        elif ori == 0:  # south
            if item_ang <= (-90 + (view_angle / 2)) and (
                item_ang >= (-90 - (view_angle / 2))
            ):
                return True
            if (
                player_pos[1] == y
                and ((player_pos[0] == x - 1) or (player_pos[0] == x + 1))
                and view_angle >= 120
            ):
                return True
        elif ori == 3:  # west
            if item_ang <= (-180 + (view_angle / 2)) or (
                item_ang >= (180 - (view_angle / 2))
            ):
                return True
            if (
                player_pos[0] == x
                and ((player_pos[1] == y - 1) or (player_pos[1] == y + 1))
                and view_angle >= 120
            ):
                return True
        return False
