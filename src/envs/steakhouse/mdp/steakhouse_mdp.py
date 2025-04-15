import copy
import itertools
import os
import logging
import numpy as np
from typing import Dict, List
from collections import Counter, defaultdict

import pygame
import pygame_gui

import random

from src.visualization.state_visualizer import SteakhouseStateVisualizer
from src.prompt.utils import clean_name, get_grid_language_reference
from src.prompt.utils import NAMES

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from overcooked_ai_py.mdp.overcooked_mdp import (
    Action,
    Direction,
    ObjectState,
    OvercookedGridworld,
    Recipe,
    SoupState,
)
from overcooked_ai_py.utils import load_from_json, read_layout_dict


logger = logging.getLogger(__name__)


class PlayerState:
    """State of a player in SteakhouseGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
        None if there is no such object.
    num_ingre_held (int): Number of times the player has held an ingredient
        object (onion or tomato).
    num_plate_held (int): Number of times the player has held a plate
    num_served (int): Number of times the player has served food
    """

    def __init__(
        self,
        position,
        orientation,
        held_object=None,
        workload: Dict = None,
        action_history: List = None,
    ):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object

        self.workload = (
            workload
            if workload is not None
            else {
                "num_onion_picked": 0,
                "num_meat_picked": 0,
                "num_meat_put_on_grill": 0,
                "num_chicken_picked": 0,
                "num_chicken_put_in_pot": 0,
                "num_dirty_dish_picked": 0,
                "num_clean_dish_picked": 0,
                "num_dish_served": 0,
                "num_dish_put_in_sink": 0,
                "num_onion_put_on_board": 0,
                "num_onion_chopped": 0,
            }
        )

        # major action history that follows workload changes
        self.action_history = action_history if action_history is not None else []

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, ObjectState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return self.position, self.orientation

    def get_pos_and_or(self):
        return self.position, self.orientation

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj

    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj

    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(
            self.position,
            self.orientation,
            new_obj,
            copy.deepcopy(self.workload),
            copy.deepcopy(self.action_history),
        )

    def __eq__(self, other):
        return (
            isinstance(other, PlayerState)
            and self.position == other.position
            and self.orientation == other.orientation
            and self.held_object == other.held_object
        )

    def __hash__(self):
        return hash(
            (self.position, self.orientation, self.held_object, self.workload.values())
        )

    def __repr__(self):
        return (
            f"{self.position} facing {self.orientation} holding "
            f"{str(self.held_object)}"
        )

    def to_dict(self, all=True):
        data = {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": (
                self.held_object.to_dict() if self.held_object is not None else None
            ),
        }
        if all:
            data["workload"] = self.workload
            data["action_history"] = self.action_history

        return data

    def get_workload(self):
        return self.workload

    def get_action_history(self):
        return self.action_history

    @staticmethod
    def from_dict(player_dict):
        player_dict = copy.deepcopy(player_dict)
        held_obj = player_dict["held_object"]
        if held_obj is not None:
            player_dict["held_object"] = ObjectState.from_dict(held_obj)
        return PlayerState(**player_dict)


class Steakhouse_Recipe(Recipe):
    MAX_NUM_INGREDIENTS = 2
    CHICKEN = "chicken"
    MEAT = "meat"
    ONION = "onion"

    ALL_INGREDIENTS = [CHICKEN, MEAT, ONION]
    STR_REP = {CHICKEN: "@", MEAT: "!", ONION: "ø"}

    @classmethod
    def configure(cls, conf):
        cls._conf = conf
        cls._configured = True
        cls._computed = False
        cls.MAX_NUM_INGREDIENTS = conf.get("max_num_ingredients", 2)

        cls._cook_time = None
        cls.delivery_reward = None
        cls.in_order_delivery_reward = None
        cls._value_mapping = None
        cls._time_mapping = None
        cls._onion_value = None
        cls._steak_time = None
        cls._chicken_value = None
        cls._chicken_time = None

        ## Basic checks for validity ##

        # Mutual Exclusion
        if (
            "chicken_time" in conf
            and not "steak_time" in conf
            or "steak_time" in conf
            and not "chicken_time" in conf
        ):
            raise ValueError("Must specify both 'steak_time' and 'chicken_time'")
        if (
            "chicken_value" in conf
            and not "steak_value" in conf
            or "steak_value" in conf
            and not "chicken_value" in conf
        ):
            raise ValueError("Must specify both 'steak_value' and 'chicken_value'")
        if "chicken_value" in conf and "delivery_reward" in conf:
            raise ValueError("'delivery_reward' incompatible with '<ingredient>_value'")
        if "chicken_value" in conf and "recipe_values" in conf:
            raise ValueError("'recipe_values' incompatible with '<ingredient>_value'")
        if "recipe_values" in conf and "delivery_reward" in conf:
            raise ValueError("'delivery_reward' incompatible with 'recipe_values'")
        if "chicken_time" in conf and "cook_time" in conf:
            raise ValueError("'cook_time' incompatible with '<ingredient>_time")
        if "chicken_time" in conf and "recipe_times" in conf:
            raise ValueError("'recipe_times' incompatible with '<ingredient>_time'")
        if "recipe_times" in conf and "cook_time" in conf:
            raise ValueError("'delivery_reward' incompatible with 'recipe_times'")

        # recipe_ lists and orders compatibility
        if "recipe_values" in conf:
            if not "all_orders" in conf or not conf["all_orders"]:
                raise ValueError(
                    "Must specify 'all_orders' if 'recipe_values' specified"
                )
            if not len(conf["all_orders"]) == len(conf["recipe_values"]):
                raise ValueError(
                    "Number of recipes in 'all_orders' must be the same as number in 'recipe_values"
                )
        if "recipe_times" in conf:
            if not "all_orders" in conf or not conf["all_orders"]:
                raise ValueError(
                    "Must specify 'all_orders' if 'recipe_times' specified"
                )
            if not len(conf["all_orders"]) == len(conf["recipe_times"]):
                raise ValueError(
                    "Number of recipes in 'all_orders' must be the same as number in 'recipe_times"
                )

        ## Conifgure ##

        if "cook_time" in conf:
            cls._cook_time = conf["cook_time"]

        if "delivery_reward" in conf:
            cls.delivery_reward = conf["delivery_reward"]

        if "in_order_delivery_reward" in conf:
            cls.in_order_delivery_reward = conf["in_order_delivery_reward"]

        if "recipe_values" in conf:
            cls._value_mapping = {
                cls.from_dict(recipe): value
                for (recipe, value) in zip(conf["all_orders"], conf["recipe_values"])
            }

        if "recipe_times" in conf:
            cls._time_mapping = {
                cls.from_dict(recipe): time
                for (recipe, time) in zip(conf["all_orders"], conf["recipe_times"])
            }

        if "chicken_time" in conf:
            cls._chicken_time = conf["chicken_time"]

        if "steak_time" in conf:
            cls._steak_time = conf["steak_time"]

        if "chicken_value" in conf:
            cls._chicken_value = conf["chicken_value"]

        if "steak_value" in conf:
            cls._steak_value = conf["steak_value"]

        if "wash_time" in conf:
            cls._wash_time = conf["wash_time"]


class IdObjectState(ObjectState):
    def __init__(self, id, name, position):
        self.id = id
        super(IdObjectState, self).__init__(
            name=name,
            position=position,
        )

    def deepcopy(self):
        return IdObjectState(self.id, self.name, self.position)

    def __eq__(self, other):
        return (
            isinstance(other, IdObjectState)
            and self.name == other.name
            and self.position == other.position
        )

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return IdObjectState(**obj_dict)

    def is_valid(self):
        return self.name in [
            "dirty_plate",
            "meat",
            "dish",
            "chicken",
            "onion",
            "steak",
            "boiled_chicken",
            "steak_onion",
            "boiled_chicken_onion",
        ]


class ChickenState(IdObjectState):
    def __init__(
        self,
        id,
        name,
        position,
        ingredients=[],
        cooking_tick=-1,
        cook_time=-1,
        **kwargs,
    ):
        """
        Represents a soup object. An object becomes a soup the instant it is placed in a pot. The
        soup's recipe is a list of ingredient names used to create it. A soup's recipe is undetermined
        until it has begun cooking.

        position (tupe): (x, y) coordinates in the grid
        ingrdients (list(ObjectState)): Objects that have been used to cook this soup. Determiens @property recipe
        cooking (int): How long the soup has been cooking for. -1 means cooking hasn't started yet
        cook_time(int): How long soup needs to be cooked, used only mostly for getting soup from dict with supplied cook_time, if None self.recipe.time is used
        """
        super(ChickenState, self).__init__(id, name, position)
        self._ingredients = ingredients
        self._cooking_tick = cooking_tick
        self._recipe = None
        self._cook_time = (
            cook_time if cook_time > 0 else Steakhouse_Recipe._chicken_time
        )

    def __eq__(self, other):
        return (
            isinstance(other, ChickenState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(ChickenState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(ChickenState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def __str__(self):
        res = "{"
        for ingredient in sorted(self.ingredients):
            res += Steakhouse_Recipe.STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def recipe(self):
        if self.is_idle:
            raise ValueError("Recipe is not determined until soup begins cooking")
        if not self._recipe:
            self._recipe = Steakhouse_Recipe(self.ingredients)
        return self._recipe

    @property
    def value(self):
        return self.recipe.value

    @property
    def cook_time(self):
        # used mostly when cook time is supplied by state dict
        if self._cook_time is not None:
            return self._cook_time
        else:
            return self.recipe.time

    @property
    def cook_time_remaining(self):
        return max(0, self.cook_time - self._cooking_tick)

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self.cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return (
            not self.is_idle
            or len(self.ingredients) == Steakhouse_Recipe.MAX_NUM_INGREDIENTS
        )

    def is_valid(self):
        if not all(
            [ingredient.position == self.position for ingredient in self._ingredients]
        ):
            return False
        if len(self.ingredients) > Steakhouse_Recipe.MAX_NUM_INGREDIENTS:
            return False
        return True

    def auto_finish(self):
        if len(self.ingredients) == 0:
            raise ValueError("Cannot finish chicken with no ingredients")
        self._cooking_tick = 0
        self._cooking_tick = self.cook_time

    def add_ingredient(self, ingredient):
        if not ingredient.name in Steakhouse_Recipe.ALL_INGREDIENTS:
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in recipe")
        ingredient.position = self.position
        self._ingredients.append(ingredient)

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def pop_ingredient(self):
        if not self.is_idle:
            raise ValueError(
                "Cannot remove an ingredient from this Chicken at this time"
            )
        if len(self._ingredients) == 0:
            raise ValueError("No ingredient to remove")
        return self._ingredients.pop()

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this chicken soup at this time")
        if len(self.ingredients) == 0:
            raise ValueError(
                "Must add at least one ingredient to chicken soup before you can begin cooking"
            )
        self._cooking_tick = 0

    def cook(self):
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    def deepcopy(self):
        return ChickenState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
            self._cook_time,
        )

    def to_dict(self):
        info_dict = super(ChickenState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "soup":
            return super(ChickenState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= Steakhouse_Recipe._chicken_time
            if ingredient == Steakhouse_Recipe.CHICKEN:
                return ChickenState.get_soup(
                    obj_dict["position"],
                    num_chicken=num_ingredient,
                    cooking_tick=cooking_tick,
                    finished=finished,
                )
        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)

    @classmethod
    def get_chicken(
        cls, position, num_chicken=0, cooking_tick=-1, finished=False, **kwargs
    ):
        if num_chicken < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_chicken > Steakhouse_Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for this soup")
        if cooking_tick >= 0 and num_chicken == 0:
            raise ValueError("_cooking_tick must be -1 for empty soup")
        if finished and num_chicken == 0:
            raise ValueError("Empty soup cannot be finished")
        chicken = [
            IdObjectState(Steakhouse_Recipe.CHICKEN, position)
            for _ in range(num_chicken)
        ]
        ingredients = chicken
        soup = cls(position, ingredients, cooking_tick)
        if finished:
            soup.auto_finish()
        return soup


class PlateState(IdObjectState):
    def __init__(self, id, name, position, rinse_total=3, rinse_count=-1, **kwargs):
        super(PlateState, self).__init__(id, name, position)
        self._cook_time = rinse_total
        self._cooking_tick = rinse_count

    def __eq__(self, other):
        return (
            isinstance(other, PlateState)
            and self.id == other.id
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
        )

    def __hash__(self):
        supercls_hash = super(PlateState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick))

    def __repr__(self):
        supercls_str = super(PlateState, self).__repr__()
        return "{}\nRinse Count:\t{}".format(supercls_str, self._cooking_tick)

    def __str__(self):
        res = "{"
        if self.is_rinsing:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    @ObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos

    @property
    def is_rinsing(self):
        return not self.is_idle and not self.is_ready

    @property
    def cook_time(self):
        # used mostly when cook time is supplied by state dict
        if self._cook_time is not None:
            return self._cook_time
        else:
            return 2

    def is_valid(self):
        return self.name in ["clean_plate", "dirty_plate"]

    @property
    def rinse_time_remaining(self):
        return max(0, self._cook_time - self._cooking_tick)

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self._cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return not self.is_idle

    def auto_finish(self):
        self._cooking_tick = 0
        self._cooking_tick = self.cook_time

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos

    def begin_rinsing(self):
        if not self.is_idle:
            raise ValueError("Cannot begin rinse at this time")
        self._cooking_tick = 0

    def rinse(self):
        if self.is_idle:
            raise ValueError("Must begin rinsing before advancing rinse tick")
        if self.is_ready:
            raise ValueError("Cannot rinse a plate that is already done")
        self._cooking_tick += 1

    def deepcopy(self):
        return PlateState(
            self.id, self.name, self.position, self._cook_time, self._cooking_tick
        )

    def to_dict(self):
        info_dict = super(PlateState, self).to_dict()
        info_dict["rinse_count"] = self._cooking_tick
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["rinse_total"] = self._cook_time
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "clean_plate" or obj_dict["name"] != "dirty_plate":
            return super(SoupState, cls).from_dict(obj_dict)

    # @classmethod
    # def get_plate(cls, position, rinse_total=2):
    #     return cls(position, rinse_total)


class SteakState(SoupState):
    def __init__(
        self, id, name, position, ingredients=[], cooking_tick=-1, cook_time=-1
    ):
        super(SteakState, self).__init__(position, ingredients)
        self.id = id
        self.name = name
        self._cooking_tick = cooking_tick
        self._cook_time = cook_time if cook_time > 0 else Steakhouse_Recipe._steak_time

    def __eq__(self, other):
        return (
            isinstance(other, SteakState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(SteakState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(SteakState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def __str__(self):
        res = "{"
        for ingredient in sorted(self.ingredients):
            res += Steakhouse_Recipe.STR_REP[ingredient]
        if self.is_cooking:
            res += str(self._cooking_tick)
        elif self.is_ready:
            res += str("✓")
        return res

    def is_valid(self):
        return self.name in ["steak"]

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    @property
    def ingredients(self):
        return [ingredient.name for ingredient in self._ingredients]

    @property
    def is_cooking(self):
        return not self.is_idle and not self.is_ready

    @property
    def cook_time(self):
        return self._cook_time

    @property
    def is_ready(self):
        if self.is_idle:
            return False
        return self._cooking_tick >= self._cook_time

    @property
    def is_idle(self):
        return self._cooking_tick < 0

    @property
    def is_full(self):
        return (
            not self.is_idle
            or len(self.ingredients) == Steakhouse_Recipe.MAX_NUM_INGREDIENTS
        )

    def auto_finish(self):
        if len(self.ingredients) == 0:
            raise ValueError("Cannot finish steak with no ingredients")
        self._cooking_tick = 0
        self._cooking_tick = self.cook_time

    def add_ingredient(self, ingredient):
        if not ingredient.name in Steakhouse_Recipe.ALL_INGREDIENTS:
            raise ValueError("Invalid ingredient")
        if self.is_full:
            raise ValueError("Reached maximum number of ingredients in recipe")
        ingredient.position = self.position
        self._ingredients.append(ingredient)

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(None, ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    def pop_ingredient(self):
        if not self.is_idle:
            raise ValueError("Cannot remove an ingredient from this steak at this time")
        if len(self._ingredients) == 0:
            raise ValueError("No ingredient to remove")
        return self._ingredients.pop()

    def begin_cooking(self):
        if not self.is_idle:
            raise ValueError("Cannot begin cooking this steak at this time")
        if len(self.ingredients) == 0:
            raise ValueError(
                "Must add at least one ingredient to steak before you can begin cooking"
            )
        self._cooking_tick = 0

    def cook(self):
        if self.is_idle:
            raise ValueError("Must begin cooking before advancing cook tick")
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    @classmethod
    def get_steak(cls, position, num_meat=1, cooking_tick=-1, finished=False, **kwargs):
        if num_meat < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_meat > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for steak")
        if cooking_tick >= 0 and num_meat == 0:
            raise ValueError("_cooking_tick must be -1 for empty grill")
        if finished and num_meat == 0:
            raise ValueError("Empty grill cannot be finished")
        meats = [ObjectState(Recipe.MEAT, position) for _ in range(num_meat)]
        ingredients = meats
        steak = cls(position, cooking_tick)
        if finished:
            steak.auto_finish()
        return steak

    def to_dict(self):
        info_dict = super(SteakState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "steak":
            return super(SteakState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= Steakhouse_Recipe._steak_time
            return SteakState.get_steak(
                obj_dict["position"],
                num_meat=num_ingredient,
                cooking_tick=cooking_tick,
                finished=finished,
            )

        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)

    def deepcopy(self):
        return SteakState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
        )


class GarnishState(SoupState):
    def __init__(self, id, name, position, ingredients=[], chop_count=-1, chop_time=2):
        super(GarnishState, self).__init__(position, ingredients)
        self.id = id
        self.name = name
        self._cooking_tick = chop_count
        self._cook_time = chop_time

    def __eq__(self, other):
        return (
            isinstance(other, GarnishState)
            and self.name == other.name
            and self.position == other.position
            and self._cooking_tick == other._cooking_tick
            and all(
                [
                    this_i == other_i
                    for this_i, other_i in zip(self._ingredients, other._ingredients)
                ]
            )
        )

    def __hash__(self):
        ingredient_hash = hash(tuple([hash(i) for i in self._ingredients]))
        supercls_hash = super(GarnishState, self).__hash__()
        return hash((supercls_hash, self._cooking_tick, ingredient_hash))

    def __repr__(self):
        supercls_str = super(GarnishState, self).__repr__()
        ingredients_str = self._ingredients.__repr__()
        return "{}\nIngredients:\t{}\nCooking Tick:\t{}".format(
            supercls_str, ingredients_str, self._cooking_tick
        )

    def is_valid(self):
        return self.name in ["garnish"]

    def begin_chop(self):
        if not self.is_idle:
            raise ValueError("Cannot begin rinse at this time")
        self._cooking_tick = 0

    def chop(self):
        if self.is_ready:
            raise ValueError("Cannot cook a soup that is already done")
        self._cooking_tick += 1

    @IdObjectState.position.setter
    def position(self, new_pos):
        self._position = new_pos
        for ingredient in self._ingredients:
            ingredient.position = new_pos

    def add_ingredient_from_str(self, ingredient_str):
        ingredient_obj = IdObjectState(None, ingredient_str, self.position)
        self.add_ingredient(ingredient_obj)

    @classmethod
    def get_garnish(
        cls, position, num_onion=1, chop_count=-1, finished=False, **kwargs
    ):
        if num_onion < 0:
            raise ValueError("Number of active ingredients must be positive")
        if num_onion > Recipe.MAX_NUM_INGREDIENTS:
            raise ValueError("Too many ingredients specified for garnish")
        if chop_count >= 0 and num_onion == 0:
            raise ValueError("_chop_count must be -1 for empty board")
        if finished and num_onion == 0:
            raise ValueError("Empty board cannot be finished")
        # onions = [
        #     ObjectState(Recipe.ONION, position) for _ in range(num_onions)
        # ]
        # ingredients = onions
        garnish = cls(position, chop_count)
        if finished:
            garnish.auto_finish()
        return garnish

    def deepcopy(self):
        return GarnishState(
            self.id,
            self.name,
            self.position,
            [ingredient.deepcopy() for ingredient in self._ingredients],
            self._cooking_tick,
        )

    def to_dict(self):
        info_dict = super(GarnishState, self).to_dict()
        ingrdients_dict = [ingredient.to_dict() for ingredient in self._ingredients]
        info_dict["_ingredients"] = ingrdients_dict
        info_dict["cooking_tick"] = self._cooking_tick
        info_dict["is_cooking"] = self.is_cooking
        info_dict["is_ready"] = self.is_ready
        info_dict["is_idle"] = self.is_idle
        info_dict["cook_time"] = -1 if self.is_idle else self.cook_time

        # This is for backwards compatibility w/ overcooked-demo
        # Should be removed once overcooked-demo is updated to use 'cooking_tick' instead of '_cooking_tick'
        info_dict["_cooking_tick"] = self._cooking_tick
        return info_dict

    @classmethod
    def from_dict(cls, obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        if obj_dict["name"] != "garnish":
            return super(GarnishState, cls).from_dict(obj_dict)

        if "state" in obj_dict:
            # Legacy soup representation
            ingredient, num_ingredient, time = obj_dict["state"]
            cooking_tick = -1 if time == 0 else time
            finished = time >= 10
            return GarnishState.get_garnish(
                obj_dict["position"],
                num_onion=num_ingredient,
                cooking_tick=cooking_tick,
                finished=finished,
            )

        ingredients_objs = [
            IdObjectState.from_dict(ing_dict) for ing_dict in obj_dict["_ingredients"]
        ]
        obj_dict["ingredients"] = ingredients_objs
        return cls(**obj_dict)


class SteakhouseState(OvercookedState):
    def __init__(
        self,
        players,
        objects,
        bonus_orders=[],
        all_orders=[],
        complete_orders=[],
        order_display_list=[],
        order_list=[],
        timestep=0,
        obj_count=0,
        **kwargs,
    ):
        self.obj_count = obj_count
        all_orders = [Steakhouse_Recipe.from_dict(order) for order in all_orders]
        self._all_orders = all_orders
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        self._bonus_orders = bonus_orders
        self._complete_orders = complete_orders
        self._order_display_list = order_display_list
        self.order_list = order_list
        self.timestep = timestep
        # assert len(set(self.bonus_orders)) == len(
        #     self.bonus_orders
        # ), "Bonus orders must not have duplicates"
        assert len(set(self.all_orders)) == len(
            self.all_orders
        ), "All orders must not have duplicates"
        # assert set(self.bonus_orders).issubset(
        #     set(self.all_orders)
        # ), "Bonus orders must be a subset of all orders"

    def deepcopy(self):
        return SteakhouseState(
            players=[player.deepcopy() for player in self.players],
            objects={pos: obj.deepcopy() for pos, obj in self.objects.items()},
            bonus_orders=[order for order in self._bonus_orders],
            all_orders=[order.to_dict() for order in self.all_orders],
            timestep=self.timestep,
            obj_count=self.obj_count,
            order_list=[order for order in self.order_list],
            order_display_list=[order for order in self._order_display_list],
        )

    def time_independent_equal(self, other):
        order_lists_equal = self.all_orders == other.all_orders

        return (
            isinstance(other, SteakhouseState)
            and self.players == other.players
            and set(self.objects.items()) == set(other.objects.items())
            and order_lists_equal
        )

    def to_dict(self, all=True):
        return {
            "players": [p.to_dict(all=all) for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "bonus_orders": [order for order in self.bonus_orders],
            "all_orders": [order.to_dict() for order in self.all_orders],
            "order_list": self.order_list,
            "timestep": self.timestep,
        }

    @property
    def all_orders(self):
        return (
            sorted(self._all_orders)
            if self._all_orders
            else sorted(Steakhouse_Recipe.ALL_RECIPES)
        )

    @property
    def curr_order(self):
        return self.order_list[0]

    @property
    def num_orders_remaining(self):
        return len(self.order_list)

    @classmethod
    def from_players_pos_and_or(
        cls,
        players_pos_and_or,
        bonus_orders=[],
        all_orders=[],
        order_list=[],
        order_display_list=[],
    ):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """
        return cls(
            [
                PlayerState(*player_pos_and_or)
                for player_pos_and_or in players_pos_and_or
            ],
            objects={},
            bonus_orders=bonus_orders,
            all_orders=all_orders,
            order_list=order_list,
            order_display_list=order_list,
        )

    @classmethod
    def from_player_positions(
        cls,
        player_positions,
        bonus_orders=[],
        all_orders=[],
        order_list=[],
        order_display_list=[],
    ):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return cls.from_players_pos_and_or(
            dummy_pos_and_or, bonus_orders, all_orders, order_list, order_display_list
        )

    @staticmethod
    def from_dict(state_dict, obj_count=0):
        state_dict = copy.deepcopy(state_dict)
        state_dict["players"] = [
            PlayerState.from_dict(p) for p in state_dict["players"]
        ]
        object_list = [IdObjectState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = {ob.position: ob for ob in object_list}
        return SteakhouseState(**state_dict, obj_count=obj_count)

    # below methods ported from ICAROS qd-humans framework for RL agents

    # def print_player_workload(
    #     self,
    # ):
    #     for idx, player in enumerate(self.players):
    #         logger.info(f"Player {idx + 1}")
    #         player.print_workload()

    def get_player_workload(
        self,
    ):
        workloads = []
        for idx, player in enumerate(self.players):
            workloads.append(player.get_workload())
        return workloads

    def cal_concurrent_active_frequency(
        self,
    ):
        """Proportion of time in which both agents are active (\in [0,1])"""
        concurrent_active_log = self.cal_concurrent_active_log()
        return np.mean(concurrent_active_log)

    def cal_concurrent_active_sum(
        self,
    ):
        concurrent_active_log = self.cal_concurrent_active_log()
        res = np.sum(concurrent_active_log)

        return res

    def cal_concurrent_active_log(
        self,
    ):
        active_logs = self.get_player_active_log()
        if len(active_logs[0]) == 0:
            return []

        return np.array(active_logs[0]) & np.array(active_logs[1])

    def get_player_active_log(
        self,
    ):
        active_log = []
        for idx, player in enumerate(self.players):
            active_log.append(player.active_log)
        return active_log

    def cal_mean_stuck_time(
        self,
    ):
        """Proportion of time in which both agents are stuck (\in [0,1])"""
        stuck_logs = self.get_player_stuck_log()
        return np.mean(stuck_logs[0])

    def cal_total_stuck_time(
        self,
    ):
        stuck_logs = self.get_player_stuck_log()
        res = sum(stuck_logs[0])
        return res

    def get_player_stuck_log(
        self,
    ):
        stuck_log = []
        for idx, player in enumerate(self.players):
            stuck_log.append(player.stuck_log)
        return stuck_log


def dishname2ingradient(dish_name):
    # map dish_name to its ingredient, for example, steak_onion_dish to {"ingredients" : ["meat","onion"]},
    if dish_name == "steak_dish":
        return {"ingredients": ["meat"]}
    elif dish_name == "boiled_chicken_dish":
        return {"ingredients": ["chicken"]}
    elif dish_name == "steak_onion_dish":
        return {"ingredients": ["meat", "onion"]}
    elif dish_name == "boiled_chicken_onion_dish":
        return {"ingredients": ["chicken", "onion"]}


def ingradient2dishname(ingradient):
    # map ingradient to its dish_name, for example, {"ingredients" : ["meat","onion"]} to steak_onion_dish
    if ingradient == ["meat"]:
        return "steak_dish"
    elif ingradient == ["chicken"]:
        return "boiled_chicken_dish"
    elif ingradient == ["meat", "onion"]:
        return "steak_onion_dish"
    elif ingradient == ["chicken", "onion"]:
        return "boiled_chicken_onion_dish"


DISH_TYPES = [
    "steak_dish",
    "boiled_chicken_dish",
    "steak_onion_dish",
    "boiled_chicken_onion_dish",
]

EVENT_TYPES = [
    # Onion events
    "onion_pickup",
    "useful_onion_pickup",
    "onion_drop",
    "useful_onion_drop",
    "potting_onion",
    # Meat events
    "meat_pickup",
    "useful_meat_pickup",
    "meat_drop",
    "useful_meat_drop",
    # chicken events,
    "chicken_pickup",
    "useful_chicken_pickup",
    "chicken_drop",
    "useful_chicken_drop",
    "potting_chicken",
    # Dish events
    "useful_steak_pickup",
    "useful_steak_drop",
    "steak_cooking",
    "dish_pickup",
    "steak_pickup",
    "boiled_chicken_pickup",
    "boiled_chicken_drop",
    "useful_boiled_chicken_pickup",
    "useful_dish_pickup",
    "dish_drop",
    "steak_drop",
    "boiled_chicken_onion_drop",
    "useful_dish_drop",
    "useful_steak_drop",
    "useful_boiled_chicken_drop",
    "dish_delivery",
    "steak_onion_pickup",
    "boiled_chicken_onion_pickup",
    "useful_steak_onion_pickup",
    "useful_boiled_chicken_onion_pickup",
    "steak_onion_drop",
    "boiled_onion_drop",
    "useful_steak_onion_drop",
    "useful_boiled_chicken_onion_drop",
    "steak_onion_dish_delivery",
    "boiled_chicken_onion_delivery",
    "steak_dish_delivery",
    "boiled_chicken_delivery",
    # Soup events
    "soup_pickup",
    "soup_delivery",
    "soup_drop",
    # Potting events
    "optimal_onion_potting",
    "optimal_tomato_potting",
    "viable_onion_potting",
    "viable_tomato_potting",
    "catastrophic_onion_potting",
    "catastrophic_tomato_potting",
    "useless_onion_potting",
    "useless_tomato_potting",
    # Chopping events
    "chop_onion",
    "onion_chopping",
    # Rinsing events
    "plate_rinsing",
    "dirty_plate_drop",
    "dirty_plate_pickup",
    "rinse_dirty_plate",
    "clean_plate_pickup",
    "useful_clean_plate_pickup",
]


class SteakhouseGridworld(OvercookedGridworld):
    def __init__(
        self,
        terrain,
        start_player_positions,
        start_all_orders=None,
        order_list=None,
        randomize_order_list=False,
        dynamic_order_list=False,
        collision: bool = True,
        bonus_list=None,
        cook_time=10,
        num_items_for_steak=1,
        num_items_for_chicken=1,
        num_items_for_soup=3,
        chop_time=3,
        in_order_delivery_reward=10,
        delivery_reward=5,
        rew_shaping_params=None,
        explicit_language=False,
        layout_name="unnamed_layout",
        object_id_dict={},
        misc={'shared_counter_locations': []}, # shared counter misc
        **kwargs,
    ):
        super().__init__(
            terrain=terrain,
            start_player_positions=start_player_positions,
            start_all_orders=start_all_orders,
            cook_time=cook_time,
            num_items_for_soup=num_items_for_soup,
            delivery_reward=delivery_reward,
            rew_shaping_params=rew_shaping_params,
            layout_name=layout_name,
        )
        self.misc = misc # set misc

        self.steak_cook_time = cook_time
        self.chop_time = chop_time
        self.object_id_dict = object_id_dict
        self.num_items_for_steak = num_items_for_steak
        self.num_items_for_chicken = num_items_for_chicken
        self.order_list = order_list
        self.randomize_order_list = randomize_order_list
        self.dynamic_order_list = dynamic_order_list
        self.delivery_reward = delivery_reward
        self.in_order_delivery_reward = in_order_delivery_reward
        self.viewer = None
        self.manager = None
        
        self.text_box = None
        self.pause_button = None
        self.paused = False
        
        self.collision = collision

        self.symbols = [
            "<font color=#FF5733 size=4.5>" f"<b>{NAMES[0]}: </b></font>",
            "<font color=#1E90FF size=4.5>" f"<b>{NAMES[1]}: </b></font>",
            "<font color=#C27DFF size=4.5>" f"<b>{NAMES[2]}: </b></font>",
            "<font color=#FF5733 size=4.5>" f"<b>{NAMES[3]}: </b></font>",
            "<font color=#1E90FF size=4.5>" f"<b>{NAMES[4]}: </b></font>",
            "<font color=#C27DFF size=4.5>" f"<b>{NAMES[5]}: </b></font>",
        ]

        # explicit naming of the objects
        self.explicit_language = explicit_language

        # precompute the grid item names in terrain
        self.grid_language_referenece = get_grid_language_reference(
            terrain,
            misc.get('shared_counter_locations', None),
        )

        self._configure_steakhouse_recipes(
            start_all_orders, num_items_for_chicken, num_items_for_steak, **kwargs
        )
        self.start_all_orders = (
            [r.to_dict() for r in Steakhouse_Recipe.ALL_RECIPES]
            if not start_all_orders
            else start_all_orders
        )

        self._all_possible_orders = copy.copy(order_list)
    
    def _append_response(self, agent_index, response):
        """Append a chat response to the response recording.
        """
        symbol = self.symbols[agent_index]
        self.text_box.append_html_text(symbol + response + ("<br>"))        

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params["grid"]
        del base_layout_params["grid"]
        base_layout_params["layout_name"] = layout_name
        if "start_state" in base_layout_params:
            base_layout_params["start_state"] = SteakhouseState.from_dict(
                base_layout_params["start_state"]
            )

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return SteakhouseGridworld.from_grid(
            grid, base_layout_params, params_to_overwrite
        )

    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), "Ragged grid"

        # Borders must not be free spaces
        def is_not_free(c):
            return c in "XOPDCWBSGT"

        for y in range(height):
            assert is_not_free(grid[y][0]), "Left border must not be free"
            assert is_not_free(grid[y][-1]), "Right border must not be free"
        for x in range(width):
            assert is_not_free(grid[0][x]), "Top border must not be free"
            assert is_not_free(grid[-1][x]), "Bottom border must not be free"

        all_elements = [element for row in grid for element in row]
        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(
            range(1, num_players + 1)
        ), "Some players were missing"
        # TODO: change this to allow more terrain, inherite.
        assert all(
            c in "XOPDSTWBMCG123456789 " for c in all_elements
        ), "Invalid character in grid"
        assert all_elements.count("1") == 1, "'1' must be present exactly once"
        assert all_elements.count("D") >= 1, "'D' must be present at least once"
        assert all_elements.count("S") >= 1, "'S' must be present at least once"
        # assert all_elements.count("P") >= 1, "'P' must be present at least once"
        # assert (
        #     all_elements.count("G") >= 1
        # ), "'G' must be present at least once"
        # assert (
        #     all_elements.count("M") >= 1
        # ), "'M' must be present at least once"

    @staticmethod
    def from_grid(
        layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False, explicit_language=True, misc={}, dynamic_order_list=False
    ):
        """
        Returns instance of OvercookedGridworld with terrain and starting
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = copy.deepcopy(base_layout_params)

        layout_grid = [[c for c in row] for row in layout_grid]
        SteakhouseGridworld._assert_valid_grid(layout_grid)

        if "layout_name" not in mdp_config:
            layout_name = "|".join(["".join(line) for line in layout_grid])
            mdp_config["layout_name"] = layout_name

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    layout_grid[y][x] = " "

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert (
                        player_positions[int(c) - 1] is None
                    ), "Duplicate player in grid"
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config.get(k, None)
            if debug:
                print(
                    "Overwriting mdp layout standard config value {}:{} -> {}".format(
                        k, curr_val, v
                    )
                )
            mdp_config[k] = v
            
        if "dynamic_order_list" not in mdp_config:
            mdp_config["dynamic_order_list"] = dynamic_order_list

        return SteakhouseGridworld(**mdp_config, misc=misc, explicit_language=explicit_language)

    def _configure_steakhouse_recipes(
        self, start_all_orders, num_items_for_chicken, num_items_for_steak, **kwargs
    ):
        self.recipe_config = {
            "num_items_for_chicken": num_items_for_chicken,
            "num_items_for_steak": num_items_for_steak,
            "all_orders": start_all_orders,
            **kwargs,
        }
        Steakhouse_Recipe.configure(self.recipe_config)

    #####################
    # BASIC CLASS UTILS #
    #####################

    def __eq__(self, other):
        return (
            np.array_equal(self.terrain_mtx, other.terrain_mtx)
            and self.start_player_positions == other.start_player_positions
            and self.start_all_orders == other.start_all_orders
            and self.steak_cook_time == other.steak_cook_time
            and self.delivery_reward == other.delivery_reward
            and self.in_order_delivery_reward == other.in_order_delivery_reward
            and self.reward_shaping_params == other.reward_shaping_params
            and self.layout_name == other.layout_name
        )

    def copy(self):
        return SteakhouseGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_all_orders=(
                None if self.start_all_orders is None else list(self.start_all_orders)
            ),
            cook_time=self.steak_cook_time,
            delivery_reward=self.delivery_reward,
            in_order_delivery_reward=self.in_order_delivery_reward,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name,
            object_id_dict=copy.deepcopy(self.object_id_dict),
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_all_orders": self.start_all_orders,
            "cook_time": self.soup_cook_time,
            "delivery_reward": self.delivery_reward,
            "in_order_delivery_reward": self.in_order_delivery_reward,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params),
        }

    ##############
    # GAME LOGIC #
    ##############
    def get_standard_start_state(self):
        if self.randomize_order_list:
            random.shuffle(self.order_list)

        if self.start_state:
            return self.start_state
        start_state = SteakhouseState.from_player_positions(
            self.start_player_positions,
            all_orders=self.start_all_orders,
            order_list=self.order_list,
        )
        return start_state

    def get_state_transition(
        self, state, joint_action, display_phi=False, motion_planner=None
    ):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered,
        shaped reward is given only for completion of subgoals
        (not soup deliveries).
        """
        events_infos = {event: [False] * self.num_players for event in EVENT_TYPES}
        assert not self.is_terminal(
            state
        ), "Trying to find successor of a terminal state: {}".format(state)

        for action, action_set in zip(joint_action, self.get_actions(state, self.collision)):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))

        new_state = state.deepcopy()
        # Resolve interacts first
        (
            sparse_reward_by_agent,
            shaped_reward_by_agent,
        ) = self.resolve_interacts(new_state, joint_action, events_infos)
        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations

        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        self.step_environment_effects(new_state)

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)
        infos = {
            "event_infos": events_infos,
            "sparse_reward_by_agent": sparse_reward_by_agent,
            "shaped_reward_by_agent": shaped_reward_by_agent,
        }
        if display_phi:
            assert (
                motion_planner is not None
            ), "motion planner must be defined if display_phi is true"
            infos["phi_s"] = self.potential_function(state, motion_planner)
            infos["phi_s_prime"] = self.potential_function(new_state, motion_planner)
        return new_state, infos

    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        new_positions, new_orientations = self.compute_new_positions_and_orientations(
            state.players, joint_action
        )
        for player_state, new_pos, new_o in zip(
            state.players, new_positions, new_orientations
        ):
            player_state.update_pos_and_or(new_pos, new_o)

    def compute_new_positions_and_orientations(self, old_player_states, joint_action):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(
            zip(
                *[
                    self._move_if_direction(p.position, p.orientation, a)
                    for p, a in zip(old_player_states, joint_action)
                ]
            )
        )
        old_positions = tuple(p.position for p in old_player_states)
        
        if self.collision:
            new_positions = self._handle_collisions(old_positions, new_positions)

        return new_positions, new_orientations

    def _handle_collisions(self, old_positions, new_positions):
        """Handles collision between agents.

        If agents' actions lead to collision, they remain in their old positions.

        Args:
            old_positions: Positions of agents before actions.
            new_positions: Positions of agents after actions.

        Returns:
            - Positions after collision handling.
            - A collision list that, for each agent, contains a list of agent indicies
              of other agents in collision.
        """
        # collisions = [[] for _ in range(self.num_players)]
        final_positions = [None] * self.num_players

        # Check for players crossing paths
        for idx0, idx1 in itertools.combinations(range(self.num_players), 2):
            p0_old, p1_old = old_positions[idx0], old_positions[idx1]
            p0_new, p1_new = new_positions[idx0], new_positions[idx1]

            if p0_new == p1_old and p0_old == p1_new:
                final_positions[idx0] = p0_old
                final_positions[idx1] = p1_old

        # Check for players ending up at same location. Just checking new positions is
        # not enough since new positions can overlap -> those agents don't move -> new
        # position of another agent overlaps with these old positions...
        agents_in_pos = defaultdict(list)
        for i, p in enumerate(final_positions):
            if p is None:
                agents_in_pos[new_positions[i]].append(i)
            else:
                agents_in_pos[p].append(i)

        collision_resolved = False
        while not collision_resolved:
            collision_resolved = True
            agents_in_pos_copy = copy.deepcopy(agents_in_pos)
            for pos, agent_list in agents_in_pos_copy.items():
                if len(agent_list) > 1:
                    collision_resolved = False
                    for i in agent_list:
                        agents_in_pos[pos].remove(i)
                        agents_in_pos[old_positions[i]].append(i)

        for pos, agent_list in agents_in_pos_copy.items():
            if len(agent_list) == 1:
                final_positions[agent_list[0]] = pos
            elif len(agent_list) > 1:
                raise ValueError("Collisions are not resolved")

        return final_positions

    def resolve_interacts(self, new_state, joint_action, events_infos, rollout=True):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        # We divide reward by agent to keep track of who contributed
        sparse_reward, shaped_reward = [0] * self.num_players, [0] * self.num_players

        for player_idx, (player, action) in enumerate(
            zip(new_state.players, joint_action)
        ):
            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)
            if not rollout:
                obj_count = len(self.object_id_dict)
            else:
                obj_count = new_state.obj_count

            # NOTE: we always log pickup/drop before performing it, as that's
            # what the logic of determining whether the pickup/drop is useful assumes
            if terrain_type == "X":
                shared_counters = self.misc.get('shared_counter_locations', [])
                shared_counters = [tuple(pt) for pt in shared_counters]

                if self.explicit_language:
                    counter_type = self.grid_language_referenece[i_pos]
                else:
                    counter_type = "counter"
                    if i_pos in shared_counters:
                        counter_type = f"shared counter {shared_counters.index(i_pos) + 1}"

                if player.has_object() and not new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    self.log_object_drop(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )

                    # Drop object on counter
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)

                    name = clean_name(obj.name)
                    player.action_history.append(
                        [f"put down {name} on {counter_type}", new_state.timestep]
                    )

                elif not player.has_object() and new_state.has_object(i_pos):
                    obj_name = new_state.get_object(i_pos).name
                    self.log_object_pickup(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )

                    # Pick up object from counter
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)

                    name = clean_name(obj.name)
                    player.action_history.append(
                        [f"picked up {name} from {counter_type}", new_state.timestep]
                    )

                elif player.has_object() and new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    player_obj = player.remove_object()

                    # Pick up object from counter
                    self.log_object_pickup(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)

                    # Drop object on counter
                    self.log_object_drop(
                        events_infos, new_state, obj_name, pot_states, player_idx
                    )
                    new_state.add_object(player_obj, i_pos)

                    name = clean_name(obj.name)
                    player.action_history.append(
                        [
                            f"picked up and put down {name} on {counter_type}",
                            new_state.timestep,
                        ]
                    )

            elif terrain_type == "O" and player.held_object is None:
                # Onion pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "onion", pot_states, player_idx
                )
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "onion", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                player.workload["num_onion_picked"] += 1

                if self.explicit_language:
                    dispenser_name = self.grid_language_referenece[i_pos]
                    player.action_history.append(
                        [f"picked up onion from {dispenser_name}", new_state.timestep]
                    )
                else:
                    player.action_history.append(["picked up onion from dispenser", new_state.timestep])

            elif terrain_type == "M" and player.held_object is None:
                # meat pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "meat", pot_states, player_idx
                )
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "meat", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                player.workload["num_meat_picked"] += 1

                if self.explicit_language:
                    dispenser_name = self.grid_language_referenece[i_pos]
                    player.action_history.append(
                        [f"picked up meat from {dispenser_name}", new_state.timestep]
                    )
                else:
                    player.action_history.append(["picked up raw meat from dispenser", new_state.timestep])

            elif (
                terrain_type == "C" and player.held_object is None
            ):  # chicken pickup from dispenser
                self.log_object_pickup(
                    events_infos, new_state, "chicken", pot_states, player_idx
                )

                new_o_id = obj_count
                o = IdObjectState(new_o_id, "chicken", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                player.workload["num_chicken_picked"] += 1

                if self.explicit_language:
                    dispenser_name = self.grid_language_referenece[i_pos]
                    player.action_history.append(
                        [f"picked up chicken from {dispenser_name}", new_state.timestep]
                    )
                else:
                    player.action_history.append(["picked up chicken from dispenser", new_state.timestep])

            elif terrain_type == "D" and player.held_object is None:
                self.log_object_pickup(
                    events_infos, new_state, "dirty_plate", pot_states, player_idx
                )
                # player.num_dirty_plate_held += 1

                # Give shaped reward if pickup is useful
                # if self.is_dirty_plate_pickup_useful(new_state, pot_states):
                #     shaped_reward[player_idx] += self.reward_shaping_params[
                #         "DIRTY_PLATE_PICKUP_REWARD"]

                # Perform dirty plate pickup from dispenser
                new_o_id = obj_count
                o = IdObjectState(new_o_id, "dirty_plate", pos)
                if not rollout:
                    self.object_id_dict[new_o_id] = o
                obj_count += 1
                player.set_object(o)
                player.workload["num_dirty_dish_picked"] += 1

                if self.explicit_language:
                    dispenser_name = self.grid_language_referenece[i_pos]
                    player.action_history.append(
                        [f"picked up dirty plate from {dispenser_name}", new_state.timestep]
                    )
                else:
                    player.action_history.append(
                        ["picked up dirty plate from dispenser", new_state.timestep]
                    )

            elif terrain_type == "W":
                if player.held_object is None:
                    # pick up clean plates
                    if self.plate_clean_at_location(new_state, i_pos):
                        self.log_object_pickup(
                            events_infos,
                            new_state,
                            "clean_plate",
                            pot_states,
                            player_idx,
                        )
                        obj = new_state.remove_object(i_pos)
                        player.set_object(obj)
                        # Give shaped reward if pickup is useful
                        # if self.is_dirty_plate_pickup_useful(new_state, pot_states):
                        # shaped_reward[player_idx] += self.reward_shaping_params["DIRTY_PLATE_PICKUP_REWARD"]

                        player.workload["num_clean_dish_picked"] += 1

                        if self.explicit_language:
                            player.action_history.append(
                                [
                                    f"picked up clean plate from {self.grid_language_referenece[i_pos]}",
                                    new_state.timestep,
                                ]
                            )
                        else:
                            player.action_history.append(
                                ["picked up clean plate from sink", new_state.timestep]
                            )

                    # rinse dirty plates
                    else:
                        if new_state.has_object(i_pos):
                            obj = new_state.get_object(i_pos)
                            if not obj.is_ready:
                                # print("rinse", obj, new_state)
                                obj.rinse()

                                events_infos["plate_rinsing"][player_idx] = True

                                if self.explicit_language:
                                    player.action_history.append(
                                        [
                                            f"rinsed dirty plate in {self.grid_language_referenece[i_pos]}",
                                            new_state.timestep,
                                        ]
                                    )
                                else:
                                    player.action_history.append(
                                        ["rinsed dirty plate in sink", new_state.timestep]
                                    )

                else:  # sink is empty and put dirty plate
                    if (
                        player.get_object().name == "dirty_plate"
                        and not new_state.has_object(i_pos)
                    ):
                        obj_name = player.get_object().name
                        self.log_object_drop(
                            events_infos, new_state, obj_name, pot_states, player_idx
                        )

                        # Drop object on counter
                        obj = player.remove_object()
                        new_o_id = obj_count
                        new_obj = PlateState(new_o_id, "clean_plate", i_pos)
                        if not rollout:
                            self.object_id_dict[new_o_id] = new_obj
                        obj_count += 1
                        new_obj.begin_rinsing()
                        # print("begin rinsing", new_obj,new_state)
                        new_state.add_object(new_obj, i_pos)  # rinse time = 0
                        player.workload["num_dish_put_in_sink"] += 1

                        if self.explicit_language:
                            player.action_history.append(
                                [
                                    f"put dirty plate in {self.grid_language_referenece[i_pos]}",
                                    new_state.timestep,
                                ]
                            )
                        else:
                            player.action_history.append(
                                ["put dirty plate in sink", new_state.timestep]
                            )

            elif terrain_type == "P" and player.has_object():
                # ready to pickup chicken from pot
                if (
                    player.get_object().name == "clean_plate"
                    and self.chicken_ready_at_location(new_state, i_pos)
                ):
                    self.log_object_pickup(
                        events_infos,
                        new_state,
                        "boiled_chicken",
                        pot_states,
                        player_idx,
                    )
                    # pickup chicken
                    player.remove_object()  # Remove the clean plate
                    obj = new_state.remove_object(i_pos)  # Get boiled chicken
                    player.set_object(obj)
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "SOUP_PICKUP_REWARD"
                    ]

                    if self.explicit_language:
                        player.action_history.append(
                            [
                                f"picked up boiled chicken from {self.grid_language_referenece[i_pos]}",
                                new_state.timestep,
                            ]
                        )
                    else:
                        player.action_history.append(
                            [f"picked up boiled chicken from pot", new_state.timestep]
                        )
                elif player.get_object().name in Steakhouse_Recipe.ALL_INGREDIENTS:
                    item_type = player.get_object().name
                    if item_type != "chicken":
                        break

                    if not new_state.has_object(i_pos):
                        # Pot was empty, add boiled_chicken to it
                        new_o_id = obj_count
                        new_obj = ChickenState(new_o_id, "boiled_chicken", i_pos, [])
                        if not rollout:
                            self.object_id_dict[new_o_id] = new_obj
                        obj_count += 1
                        new_state.add_object(new_obj)
                    chicken_soup = new_state.get_object(i_pos)
                    if not chicken_soup.is_full:
                        old_soup = chicken_soup.deepcopy()
                        obj = player.remove_object()
                        chicken_soup.add_ingredient(obj)
                        chicken_soup.begin_cooking()
                        shaped_reward[player_idx] += self.reward_shaping_params[
                            "PLACEMENT_IN_POT_REW"
                        ]
                        # Log meat cooking
                        # Log potting TODO: commented for now
                        # self.log_object_potting(
                        #     events_infos,
                        #     new_state,
                        #     old_soup,
                        #     chicken_soup,
                        #     obj.name,
                        #     player_idx,
                        # # )
                        if obj.name == Steakhouse_Recipe.CHICKEN:
                            events_infos["potting_chicken"][player_idx] = True
                        player.workload["num_chicken_put_in_pot"] += 1

                        if self.explicit_language:
                            player.action_history.append(
                                [
                                    f"put chicken in {self.grid_language_referenece[i_pos]}",
                                    new_state.timestep,
                                ]
                            )
                        else:
                            player.action_history.append(
                                ["put chicken in pot", new_state.timestep]
                            )

            elif terrain_type == "G" and player.has_object():
                if (
                    player.get_object().name == "clean_plate"
                    and self.steak_ready_at_location(new_state, i_pos)
                ):
                    self.log_object_pickup(
                        events_infos, new_state, "steak", pot_states, player_idx
                    )

                    # Pick up steak
                    player.remove_object()  # Remove the clean plate
                    obj = new_state.remove_object(i_pos)  # Get steak
                    player.set_object(obj)
                    # shaped_reward[player_idx] += self.reward_shaping_params[
                    # "STEAK_PICKUP_REWARD"]

                    if self.explicit_language:
                        player.action_history.append(
                            [
                                f"picked up steak from {self.grid_language_referenece[i_pos]} with clean plate",
                                new_state.timestep,
                            ]
                        )
                    else:
                        player.action_history.append(
                            [
                                "picked up steak from grill with clean plate",
                                new_state.timestep,
                            ]
                        )

                elif player.get_object().name in Steakhouse_Recipe.ALL_INGREDIENTS:
                    item_type = player.get_object().name
                    if item_type != "meat":
                        break
                    if not new_state.has_object(i_pos):
                        # Pot was empty, add meat to it
                        obj = player.remove_object()
                        new_o_id = obj_count
                        new_obj = SteakState(new_o_id, "steak", i_pos, [])

                        if not rollout:
                            self.object_id_dict[new_o_id] = new_obj
                        obj_count += 1
                        new_obj.add_ingredient(obj)
                        new_obj.begin_cooking()
                        new_state.add_object(new_obj, i_pos)

                        shaped_reward[player_idx] += self.reward_shaping_params[
                            "PLACEMENT_IN_POT_REW"
                        ]

                        # Log meat cooking
                        events_infos["steak_cooking"][player_idx] = True
                        player.workload["num_meat_put_on_grill"] += 1

                        if self.explicit_language:
                            player.action_history.append(
                                [
                                    f"put meat on {self.grid_language_referenece[i_pos]}",
                                    new_state.timestep,
                                ]
                            )
                        else:
                            player.action_history.append(
                                ["put meat on grill", new_state.timestep]
                            )

            elif terrain_type == "S" and player.has_object():
                obj = player.get_object()
                dish_name = obj.name + "_dish"
                # if (dish_name in new_state.order_list) or (dish_name in new_state._complete_orders):
                if dish_name in DISH_TYPES:
                    new_state, delivery_rew = self.deliver_dish(new_state, player, obj)
                    sparse_reward[player_idx] += delivery_rew
                    player.workload["num_dish_served"] += 1

                    # Log dish delivery
                    events_infos["dish_delivery"][player_idx] = True

                    clean_dish_name = clean_name(dish_name)

                    if self.explicit_language:
                        player.action_history.append(
                            [
                                f"delivered {clean_dish_name}",
                                new_state.timestep,
                            ]
                        )
                    else:
                        player.action_history.append(
                            [f"delivered {clean_dish_name}", new_state.timestep]
                        )

                    # If last soup necessary was delivered, stop resolving interacts
                    if (
                        new_state.order_list is not None
                        and len(new_state.order_list) == 0
                    ):
                        break

            elif terrain_type == "B":
                if player.held_object is None:
                    if new_state.has_object(i_pos):
                        obj = new_state.get_object(i_pos)
                        assert (
                            obj.name == "garnish"
                        ), "Object on chopping board was not garnish"
                        if not obj.is_ready:
                            obj.chop()
                            # shaped_reward[
                            #     player_idx] += self.reward_shaping_params[
                            #         "CHOPPING_ONION_REW"]

                            # Log onion chopping
                            events_infos["onion_chopping"][player_idx] = True

                            player.workload["num_onion_chopped"] += 1

                            if self.explicit_language:
                                player.action_history.append(
                                    [
                                        f"chopped onion on {self.grid_language_referenece[i_pos]}",
                                        new_state.timestep,
                                    ]
                                )
                            else:
                                player.action_history.append(
                                    ["chopped onion", new_state.timestep]
                                )

                elif player.get_object().name == "onion" and not new_state.has_object(
                    i_pos
                ):
                    # Chopping board was empty, add onion to it
                    obj = player.remove_object()
                    new_o_id = obj_count
                    new_obj = GarnishState(new_o_id, "garnish", i_pos, [])
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    new_obj.add_ingredient(obj)
                    new_obj.begin_chop()
                    new_state.add_object(new_obj, i_pos)
                    # shaped_reward[
                    # player_idx] += self.reward_shaping_params[
                    # "PLACEMENT_ON_BOARD_REW"]

                    # Log onion potting
                    events_infos["onion_chopping"][player_idx] = True

                    player.workload["num_onion_put_on_board"] += 1

                    if self.explicit_language:
                        player.action_history.append(
                            [
                                f"put onion on {self.grid_language_referenece[i_pos]}",
                                new_state.timestep,
                            ]
                        )
                    else:
                        player.action_history.append(
                            ["put onion on chopping board", new_state.timestep]
                        )

                # Pick up garnish
                elif (
                    player.get_object().name == "steak"
                    and self.garnish_ready_at_location(new_state, i_pos)
                ):
                    player.remove_object()  # Remove the clean plate
                    self.log_object_pickup(
                        events_infos, new_state, "steak", pot_states, player_idx
                    )

                    _ = new_state.remove_object(i_pos)  # Get steak
                    new_o_id = obj_count
                    new_obj = IdObjectState(new_o_id, "steak_onion", pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    player.set_object(new_obj)
                    # shaped_reward[player_idx] += self.reward_shaping_params[
                    #     "GARNISH_STEAK_REWARD"]

                    if self.explicit_language:
                        player.action_history.append(
                            [
                                f"added onion garnish from {self.grid_language_referenece[i_pos]} to steak dish",
                                new_state.timestep,
                            ]
                        )
                    else:
                        player.action_history.append(
                            ["added onion garnish to steak dish", new_state.timestep]
                        )

                # Pick up garnish
                elif (
                    player.get_object().name == "boiled_chicken"
                    and self.garnish_ready_at_location(new_state, i_pos)
                ):
                    player.remove_object()  # Remove the clean plate
                    self.log_object_pickup(
                        events_infos,
                        new_state,
                        "boiled_chicken",
                        pot_states,
                        player_idx,
                    )

                    _ = new_state.remove_object(i_pos)  # Get steak
                    new_o_id = obj_count
                    new_obj = IdObjectState(new_o_id, "boiled_chicken_onion", pos)
                    if not rollout:
                        self.object_id_dict[new_o_id] = new_obj
                    obj_count += 1
                    player.set_object(new_obj)
                    # shaped_reward[player_idx] += self.reward_shaping_params[
                    #     "GARNISH_STEAK_REWARD"]

                    if self.explicit_language:
                        player.action_history.append(
                            [
                                f"added onion garnish from {self.grid_language_referenece[i_pos]} to boiled chicken dish",
                                new_state.timestep,
                            ]
                        )
                    else:
                        player.action_history.append(
                            [
                                "added onion garnish to boiled chicken dish",
                                new_state.timestep,
                            ]
                        )
            else:
                continue

            new_state.obj_count = obj_count

        return sparse_reward, shaped_reward

    def deliver_dish(self, state, player, dish_obj):
        """
        Deliver the steak, and get reward if there is no order list
        or if the type of the delivered steak matches the next order.
        """
        player.remove_object()

        if state.order_list is None:
            return state, self._delivery_reward

        # If the delivered soup is the one currently required
        # assert not self.is_terminal(state)
        current_order = state.order_list[0]
        dish = dish_obj.name + "_dish"
        if dish in current_order:
            # dish served in order
            state.order_list = state.order_list[1:]
            if self.dynamic_order_list:
                state.order_list.append(random.choice(self._all_possible_orders))

            state._bonus_orders.append(dish + "_tick")
            state._complete_orders.append(dish + "_tick")
            state._order_display_list = state.order_list  # + state._complete_orders
            return state, self.in_order_delivery_reward
        elif dish in state.order_list:
            # dish served in not in order, but in order list
            state.order_list.remove(dish)
            state._bonus_orders.append(dish + "_tick")
            state._complete_orders.append(dish + "_tick")
            state._order_display_list = state.order_list  # + state._complete_orders
            # print("bonus orders",state._bonus_orders)
            return state, self.delivery_reward
        else:
            # dish served not in order list
            # TODO: now the dish is just lost, should log it
            state._bonus_orders.append(dish)
            # print("bonus orders",state._bonus_orders)
            return state, 0

    def step_environment_effects(self, state):
        state.timestep += 1

        for obj in state.objects.values():
            if obj.name == "steak":
                # automatically starts cooking when the pot has 1 ingredients
                if self.old_dynamics and (
                    not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) == 1
                ):
                    obj.begin_cooking()
                if obj.is_cooking:
                    obj.cook()
            elif obj.name == "boiled_chicken":
                # automatically starts cooking when the pot has 1 ingredients
                if (
                    not obj.is_cooking
                    and not obj.is_ready
                    and len(obj.ingredients) == 1
                ):
                    obj.begin_cooking()
                if obj.is_cooking:
                    obj.cook()

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        if len(state.order_list) <= 0:
            return True
        return False

    #######################
    # LAYOUT / STATE INFO #
    #######################

    def get_chopping_board_locations(self):
        return list(self.terrain_pos_dict["B"])

    def get_meat_dispenser_locations(self):
        return list(self.terrain_pos_dict["M"])

    def get_chicken_dispenser_locations(self):
        return list(self.terrain_pos_dict["C"])

    def get_sink_locations(self):
        return list(self.terrain_pos_dict["W"])

    def get_dirty_plate_locations(self):
        return list(self.terrain_pos_dict["D"])

    def get_grill_locations(self):
        return list(self.terrain_pos_dict["G"])

    def get_key_objects_locations(self):
        return (
            self.mdp.get_onion_dispenser_locations()
            + self.mdp.get_chopping_board_locations()
            + self.mdp.get_meat_dispenser_locations()
            + self.mdp.get_grill_locations()
            + self.mdp.get_pot_locations()
            + self.mdp.get_dirty_plate_dispenser_locations()
            + self.mdp.get_sink_locations()
        )

    def get_pot_states(self, state, pots_states_dict=None, valid_pos=None):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         onion: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        if pots_states_dict is None:
            pots_states_dict = defaultdict(list)

        get_pot_info = []
        if valid_pos is not None:
            for pot_pos in self.get_pot_locations():
                if pot_pos in valid_pos:
                    get_pot_info.append(pot_pos)
        else:
            get_pot_info = self.get_pot_locations()

        for pot_pos in get_pot_info:
            if not state.has_object(pot_pos):
                pots_states_dict["empty"].append(pot_pos)
            else:
                soup = state.get_object(pot_pos)
                assert soup.name == "soup" or "chicken", (
                    "soup at "
                    + str(pot_pos)
                    + " is not a chicken/soup but a "
                    + soup.name
                )
                if soup.is_ready:
                    pots_states_dict["ready"].append(pot_pos)
                elif soup.is_cooking:
                    pots_states_dict["cooking"].append(pot_pos)
                else:
                    num_ingredients = len(soup.ingredients)
                    pots_states_dict["{}_items".format(num_ingredients)].append(pot_pos)

        return pots_states_dict

    def get_grill_states(self, state, grills_states_dict=None, valid_pos=None):
        """Returns dict with structure:
        {
         empty: [positions of empty pots]
        'x_items': [grill objects with x items that have yet to start grilling],
        'cooking': [grill objs that are grilling but not ready]
        'ready': [ready grill objs],
        }
        NOTE: all returned grills are just grill positions
        """
        if grills_states_dict is None:
            grills_states_dict = defaultdict(list)

        get_grill_info = []
        if valid_pos is not None:
            for grill_pos in self.get_grill_locations():
                if grill_pos in valid_pos:
                    get_grill_info.append(grill_pos)
        else:
            get_grill_info = self.get_grill_locations()

        for grill_pos in get_grill_info:
            if not state.has_object(grill_pos):
                grills_states_dict["empty"].append(grill_pos)
            else:
                steak = state.get_object(grill_pos)
                assert steak.name == "steak", (
                    "steak at " + grill_pos + " is not a steak but a " + steak.name
                )
                if steak.is_ready:
                    grills_states_dict["ready"].append(grill_pos)
                else:  # steak is_cooking
                    grills_states_dict["cooking"].append(grill_pos)

        return grills_states_dict

    def get_ready_grills(self, grill_states):
        return grill_states["ready"]

    def get_cooking_grills(self, grill_states):
        return grill_states["cooking"]

    def get_sink_states(self, state):
        empty_sink = []
        full_sink = []
        ready_sink = []
        sink_locations = self.get_sink_locations()
        for loc in sink_locations:
            if not state.has_object(loc):  # board is empty
                empty_sink.append(loc)
            else:
                obj = state.get_object(loc)
                if obj.is_ready:
                    ready_sink.append(loc)
                else:
                    full_sink.append(loc)
        return {"empty": empty_sink, "full": full_sink, "ready": ready_sink}

    def get_chopping_board_states(self, state):
        empty_board = []
        full_board = []
        ready_board = []
        board_locations = self.get_chopping_board_locations()
        for loc in board_locations:
            if not state.has_object(loc):  # board is empty
                empty_board.append(loc)
            else:
                obj = state.get_object(loc)
                if obj.is_ready:
                    ready_board.append(loc)
                else:
                    full_board.append(loc)
        return {"empty": empty_board, "full": full_board, "ready": ready_board}

    def steak_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == "steak", "Object in grill was not steak"
        return obj.is_ready

    def steak_to_be_cooked_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        return obj.name == "steak" and not obj.is_cooking and not obj.is_ready

    def plate_clean_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == "clean_plate", "Object in sink was not clean plate"
        return obj._cooking_tick >= obj._cook_time

    def garnish_ready_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == "garnish", "Object on chopping board was not garnish"
        prep_time = obj._cooking_tick
        return prep_time >= obj._cook_time

    # TODO: change above objectname_ready_at_location to object_ready_at_location
    def chicken_ready_at_location(self, state, pos):
        obj_name = "boiled_chicken"
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        assert obj.name == obj_name, "Object at location was not {}".format(obj_name)
        return obj.is_ready

    ################################
    # EVENT LOGGING HELPER METHODS #
    ################################

    def log_object_drop(self, events_infos, state, obj_name, pot_states, player_index):
        """Player dropped the object on a counter"""
        obj_drop_key = obj_name + "_drop"
        if obj_drop_key not in events_infos:
            # TODO: add support for tomato event logging
            if obj_name in [
                "meat",
                "clean_plate",
                "steak",
                "garnish",
                "chicken",
                "boiled_chicken",
            ]:
                return
            raise ValueError("Unknown event {}".format(obj_drop_key))

    def is_potting_optimal(self, state, old_soup, new_soup):
        """
        True if the highest valued soup possible is the same before and after the potting
        """
        old_recipe = (
            Steakhouse_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        new_recipe = Steakhouse_Recipe(new_soup.ingredients)
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return old_val == new_val

    def is_potting_viable(self, state, old_soup, new_soup):
        """
        True if there exists a non-zero reward soup possible from new ingredients
        """
        new_recipe = Steakhouse_Recipe(new_soup.ingredients)
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return new_val > 0

    def is_potting_catastrophic(self, state, old_soup, new_soup):
        """
        True if no non-zero reward soup is possible from new ingredients
        """
        old_recipe = (
            Steakhouse_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        new_recipe = Steakhouse_Recipe(new_soup.ingredients)
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        new_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, new_recipe)
        )
        return old_val > 0 and new_val == 0

    def is_potting_useless(self, state, old_soup, new_soup):
        """
        True if ingredient added to a soup that was already gauranteed to be worth at most 0 points
        """
        old_recipe = (
            Steakhouse_Recipe(old_soup.ingredients) if old_soup.ingredients else None
        )
        old_val = self.get_recipe_value(
            state, self.get_optimal_possible_recipe(state, old_recipe)
        )
        return old_val == 0

    #####################
    # TERMINAL GRAPHICS #
    #####################

    def state_string(self, state):
        """String representation of the current state"""
        # TODO
        return ""

    ###################
    # RENDER FUNCTION #
    ###################

    def render(
        self,
        state: SteakhouseState,
        mode: str,
        time_step_left: int = None,
        time_passed=None,
        chat_messages=[],
    ):
        """Function that renders the game.

        Args:
            state: State to render.
            mode: Only for compatibility, not used.
            time_step_left: Timesteps remaining in the episode.
            time_passed: Only for compatibility, not used.
        """
        if self.viewer is None:
            pygame.init()
            pygame.font.init()
            # create viewer
            # self.viewer = pygame.display.set_mode(
            #     (self.width * 30, self.height * 30 + 140), pygame.RESIZABLE
            # )
            CONVERSATION_BOX_WIDTH = 500
            STATUS_BAR_HEIGHT = 50
            STATUS_BAR_WIDTH = 100

            COLOR_OVERCOOKED = (155, 101, 0)
            
            screen_width = self.width * 30 + CONVERSATION_BOX_WIDTH + 70
            screen_height = self.height * 30 + 140

            self.viewer = pygame.display.set_mode(
                (
                    screen_width,
                    screen_height
                ), pygame.RESIZABLE
            )
            self.viewer.fill(COLOR_OVERCOOKED)
            
            self.clock = pygame.time.Clock()

            self.manager = pygame_gui.UIManager((screen_width, screen_height))

            # chat box
            self.text_box = pygame_gui.elements.UITextBox(
                html_text="<b> ---Chat--- </b> <br> ",
                relative_rect=pygame.Rect(
                    (self.width * 30 + 40, STATUS_BAR_HEIGHT + 10),
                    (CONVERSATION_BOX_WIDTH, screen_height - STATUS_BAR_HEIGHT - 10),
                ),
                manager=self.manager,
                object_id="abc",
            )
            
            PAUSE_BUTTON_HEIGHT = 50
            PAUSE_BUTTON_WIDTH = 100
            
            # pause button
            # self.pause_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((self.width-PAUSE_BUTTON_WIDTH - 30 ,0), (PAUSE_BUTTON_WIDTH, PAUSE_BUTTON_HEIGHT)),
            #     text='PAUSE',
            #     manager=self.manager
            # )

            # pygame.font.init()

            TIMER, t = pygame.USEREVENT + 1, 1000
            pygame.time.set_timer(TIMER, t)

            # Create the visualizer class
            ds = load_from_json(
                f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}"
                f"/data/config/kitchen_config.json"
            )
            test_dict = copy.deepcopy(ds)
            self.state_visualizer = SteakhouseStateVisualizer(**test_dict["config"])

            pygame.font.init()

        for agent_idx, timestep, message in chat_messages:
            self._append_response(agent_idx, message)
            
        for event in pygame.event.get():
            # Pass events to the UIManager
            self.manager.process_events(event)

        # Update UIManager
        time_delta = self.clock.tick(60) / 1000.0  # Convert milliseconds to seconds
        self.manager.update(time_delta)

        # TODO: Set the score correctly based on the sparse reward. It should be
        # something like `sum(infos["sparse_reward_by_agent"])` where `infos` is what is
        # calculated in `self.get_state_transition()`
        kitchen = self.state_visualizer.render_state(
            state,
            self.terrain_mtx,
            hud_data=self.state_visualizer.default_hud_data(
                state,
                time_left=time_step_left,
                score=0,
            ),
        )
        
        self.viewer.blit(kitchen, (0, 0))
        self.manager.draw_ui(self.viewer)
        pygame.display.update()

        pygame.display.flip()