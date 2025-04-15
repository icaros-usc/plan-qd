import math
from typing import Dict
from collections import deque
from overcooked_ai_py.mdp.actions import Action

NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
]


def get_grid_language_reference(grid_layout, shared_counter_locations=None):
    """Creates a dict, something like {"Counter": [(0,0), (1,0)]}
    """
    counter = {}
    grid_dict = {}

    for j in range(len(grid_layout)):
        for i in range(len(grid_layout[j])):
            item = grid_layout[j][i]
            if item in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                item = " "

            name = ""
            if item == "X":
                name = "Counter"
                if shared_counter_locations is not None:
                    if (i, j) in shared_counter_locations:
                        name = "Shared counter"
                    else:
                        name = "General counter"
            elif item == "P":
                name = "Pot"
            elif item == "D":
                name = "Dirty dish dispenser"
            elif item == "O":
                name = "Onion dispenser"
            elif item == "T":
                name = "Tomato dispenser"
            elif item == "S":
                name = "Delivery location"
            elif item == "G":
                name = "Grill"
            elif item == "C":
                name = "Chicken dispenser"
            elif item == "W":
                name = "Sink"
            elif item == "M":
                name = "Raw meat dispenser"
            elif item == "B":
                name = "Chopping board"
            elif item == " ":
                continue

            if counter.get(name):
                counter[name] += 1
            else:
                counter[name] = 1
            
            grid_dict[(i, j)] = name + " " + str(counter[name])

    return grid_dict

def replace_all(text, dic):
    """Replacement helper function that replaces all instances of a substring in a string
    with another substring. Helpful for any prompt formatting for LLMs."""
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def clean_name(name):
    """Clean the name of the item"""
    if name == "steak":
        return "steak dish"
    elif name == "steak_onion":
        return "steak onion dish"
    elif name == "meat":
        return "raw meat"
    elif name == "boiled_chicken":
        return "boiled chicken dish"
    elif name == "boiled_chicken_onion":
        return "boiled chicken onion dish"
    return name.replace("_", " ").lower()

class Prompt:
    def __init__(self, grid_layout) -> None:
        self.grid_layout = grid_layout  # store gridlayout for continuous use

        # names to be assigned to the chefs
        self.names = NAMES

    # common functions

    def set_grid_layout(self, grid_layout):
        self.grid_layout = grid_layout

    def set_grid_config(self, grid_config):
        self.grid_config = grid_config

        self.grid_language_reference = get_grid_language_reference(
            self.grid_config['grid'],
            shared_counter_locations=self.grid_config.get('shared_counter_locations', None)
        )
        
    def format_other_agents_description(self, agent_index, agent_types):
        """Construct the chef's description based on the agent types
        """
        description = []
        ai_agents = []

        for i, agent_type in enumerate(agent_types):
            if i == agent_index: # don't include the agent's own description
                continue

            description.append( self.names[i] )

            if agent_type != "LLMAgent":
                ai_agents.append( self.names[i] )
                # chefs_description += f"{self.names[i]} is an AI agent. They cannot communicate with you."

        ai_agents_language = ""
        if len(ai_agents) != 0:
            ai_agents_language = ". " + ", ".join(ai_agents) + " are AI agents. They cannot hear you, but can send you messages describing their current action"

        return "You are playing the game Overcooked along with " + ", ".join(description) + ai_agents_language

    def format_chat_history_as_language(self, agent_index, chat_history, timestep):
        # construct chat history as language
        messages_as_language = []
        for m in chat_history:
            prefix = f"You ({self.names[m[0]]}) ({timestep - m[1]} timesteps ago): "
            if m[0] != agent_index:
                prefix = self.names[m[0]] + f" ({timestep - m[1]} timesteps ago): "
            messages_as_language.append(prefix + m[2])

        if len(messages_as_language) == 0:
            messages_as_language = ["No messages"]

        return "\n".join(messages_as_language)
    
    def select_recent_chat_from_agent(self, agent_index, chat_history, timestep):
        recent_chat = None
        recent_timestamp = 0
        for m in chat_history[::-1]:
            if m[0] == agent_index and timestep - m[1] < 5:
                recent_chat = m[2]
                recent_timestamp = m[1]
                break
        if recent_chat == None or recent_timestamp > 20:
            recent_chat = "No message."

        return f"Message from {NAMES[agent_index]}: ({timestep - recent_timestamp} timesteps ago): {recent_chat}"

    def format_action_history_as_language(self, action_history, timestep, memory_depth=0):
        if memory_depth == 0:
            return "You can't remember your past actions."

        # construct action history as language
        if len(action_history) == 0:
            return "You have not done anything recently yet!"
        
        action_history_as_language = []
        for a, t in action_history[::-1][:memory_depth]:
            action_history_as_language.append(f"You {a} {timestep - t} timesteps ago.")

        return "\n".join(action_history_as_language)

    def format_inventory_as_language(self, state, agent_index):
        """Format the inventory of all the agents in the perspective of `agent_index`.
        """
        current_agent_state = state.players[agent_index].to_dict()

        # first person agent state
        your_object = "nothing"
        if current_agent_state["held_object"] is not None:
            object_clean = clean_name(current_agent_state["held_object"]["name"])
            your_object = "a " + object_clean

        inventory = f"You ({NAMES[agent_index]}) are holding {your_object}. " + " ".join(
            [
                f"{self.names[i]} is holding a {clean_name(state.players[i].held_object.name)}."
                if state.players[i].held_object is not None
                else f"{self.names[i]} is holding nothing."
                for i in range(len(state.players)) if i != agent_index
            ]
        )
        return inventory

    def format_prompt_given_states(
        self,
        prompt: str,
        world_state,
        current_agent_state,
        other_agents_states,
        other_chef_message="No message",
        advisor_message="No message",
        verbose=False,
        other_agent_indices=None,
    ):
        """format the prompt given world states

        does not format the task list

        Input:
            prompt: the layout read from the layout file

        Returns:
            formatted prompt
        """

        # resolve list
        if type(other_agents_states) is not list:
            other_agents_states = [other_agents_states]

        # TODO: resolve this more elegantly
        if other_agent_indices is None:
            other_agent_indices = [1]

        # current state
        current_state = self.get_agent_state_as_language(
            current_agent_state, first_person=True
        )
        prompt = prompt.replace("{current_state}", current_state)

        # for each other chef, format their state
        all_other_chef_states = ""
        for i, other_agent_state in enumerate(other_agents_states):
            # get the agent state with thier name
            other_chef_state = self.get_agent_state_as_language(
                other_agent_state, first_person=False, chef_name=self.names[i] # f"Chef {other_agent_indices[i]}"
            )
            all_other_chef_states += f"{other_chef_state}\n"

        prompt = prompt.replace("{other_chef_state}", all_other_chef_states)

        # update kitchen state
        kitchen_overview, kitchen_items = self.get_kitchen_as_language(
            world_state, current_agent_state, other_agents_states, verbose=verbose, other_agent_indices=other_agent_indices
        )
        prompt = prompt.replace("{kitchen_items}", kitchen_items)
        prompt = prompt.replace("{kitchen_overview}", kitchen_overview)

        # check if chef_message is enabled in the prompt
        if prompt.find("{other_chef_message}") != -1:
            # add the chef message to the prompt
            prompt = prompt.replace("{other_chef_message}", other_chef_message)

        if prompt.find("{advisor_message}") != -1:
            # add the advisor message to the prompt
            prompt = prompt.replace("{advisor_message}", advisor_message)
        return prompt

    def format_low_level_prompt_given_states(
        self,
        prompt: str,
        world_state,
        current_agent_state,
        other_agent_state,
        subtask_in_mind,
        other_chef_message="No messsage",
        verbose=True,
    ):
        """format the prompt given world states

        does format the task list

        Input:
            prompt: the layout read from the layout file

        Returns:
            formatted prompt
        """
        # format the prompt

        # current state
        current_state = self.get_agent_state_as_language(
            current_agent_state, first_person=True)
        
        prompt = prompt.replace("{current_state}", current_state)

        # other chef state
        other_chef_state = self.get_agent_state_as_language(
            other_agent_state, first_person=False
        )
        prompt = prompt.replace("{other_chef_state}", other_chef_state)

        # update kitchen state
        kitchen_overview, kitchen_items = self.get_kitchen_as_language(
            world_state, current_agent_state, other_agent_state, verbose=verbose
        )
        prompt = prompt.replace("{kitchen_items}", kitchen_items)
        prompt = prompt.replace("{kitchen_overview}", kitchen_overview)

        # we don't need these to make atomic decision. but i
        # check if chef_message is enabled in the prompt
        if prompt.find("{other_chef_message}") != -1:
            # add the chef message to the prompt
            prompt = prompt.replace("{other_chef_message}", other_chef_message)

        # check for task_list
        if prompt.find("{subtask_in_mind}") != -1:
            prompt = prompt.replace("{subtask_in_mind}", subtask_in_mind)
        return prompt

    def get_agent_state_as_language(self, state, first_person=False, chef_name=None):
        """Construct the agent state as a string from a dictionary containing its contents"""

        # pronouns resolver
        if chef_name:
            if first_person:
                pronouns = [f"You ({chef_name}) are", "Your", "You are"]
            else:
                pronouns = [f"{chef_name} is", f"{chef_name}'s", f"{chef_name} is"]
        else:
            if first_person:
                pronouns = ["You are", "Your", "You are"]
            else:
                pronouns = ["The other chef is", "Their", "They are"]

        # held object resolver
        if state["held_object"] is not None:
            name = state["held_object"]["name"]
            if name == "soup":
                soup_ingredients = state["held_object"]["_ingredients"]

                ingredients = []
                for ingredient in soup_ingredients:
                    ingredients.append(ingredient["name"])

                ingredients = ", ".join(ingredients)

                name = f"finished soup dish with ingredients: {ingredients}"

            held_object = "a " + name
        else:
            held_object = "nothing"
        orientation_to_string = {
            (1, 0): "right",
            (-1, 0): "left",
            (0, 1): "down",
            (0, -1): "up",
            (0, 0): "staying",
        }

        # construct and return state string
        return f"""1. {pronouns[0]} is at the coordinates {state['position']}
    2. {pronouns[1]} orientation is facing {orientation_to_string[state['orientation']]}
    3. {pronouns[2]} holding {held_object}
        """

    def get_kitchen_as_language(
        self,
        world_state,
        current_agent_state,
        other_agents_states,
        verbose=False,
        informative=False,
        other_agent_indices=None,
    ):
        """Construct the kitchen state as a string from a dictionary containing its contents

        Returns:
            1) overview of the kitchen
            2) items in the kitchen. if verbose=False, won't populate irrelevant items
        """

        # resolve agent states
        if type(other_agents_states) is not list:
            other_agents_states = [other_agents_states]

        # resolve other agent indices
        # TODO: resolve this more elegantly
        if other_agent_indices is None:
            other_agent_indices = [1]

        grid_layout = self.grid_layout

        # construct kitchen overview
        x_dim, y_dim = len(grid_layout[0]), len(grid_layout)
        kitchen_overview = f"The kitchen is a {x_dim}x{y_dim} grid world. The top left corner is (0, 0) and the bottom right corner is ({x_dim-1}, {y_dim-1})."

        # construct kitchen items
        kitchen_items = []

        for i in range(len(grid_layout)):
            for j in range(len(grid_layout[i])):
                necessary = False  # only include necessary information

                item_state = ""

                item = grid_layout[i][j]
                if item == "X":
                    item_name = "Counter"

                    if informative:
                        item_state = "The counter is empty."

                    # resolve counter contents (not that efficient)
                    for counter in world_state:
                        if counter["position"][0] == j and counter["position"][1] == i:
                            necessary = True
                            item_state = f"The counter has a {counter['name']} on it."

                            if counter["name"] == "steak":
                                item_state += " There is a steak on the counter. You can deliver this as is, or add garnish to it."
                            if counter["name"] == "steak_onion":
                                item_state += " There is a steak with garnish on the counter. You can deliver this dish."
                            break

                elif item == "P":
                    necessary = True
                    item_name = "Pot"

                    pot_state = None
                    # find the pot at this position
                    for pot in world_state:
                        if (
                            pot["name"] == "soup"
                            and pot["position"][0] == j
                            and pot["position"][1] == i
                        ):
                            pot_state = pot
                            break

                    # special case resolution for pot
                    # item_state = "The pot is empty."
                    item_state = self.get_pot_state_as_language(pot_state)
                elif item == "D":
                    item_name = "Dish dispenser"
                    necessary = True
                    if informative:
                        item_state = "The dish dispenser has infinite empty dishes."
                elif item == "O":
                    item_name = "Onion dispenser"
                    necessary = True
                    if informative:
                        item_state = "The onion dispenser has infinite onions."
                elif item == "T":
                    item_name = "Tomato dispenser"
                    necessary = True
                    if informative:
                        item_state = "The tomato dispenser has infinite tomatoes."
                elif item == "S":
                    item_name = "Delivery location"
                    necessary = True
                    # item_state = "The delivery location is empty."
                    item_state = ""
                elif item == "G":
                    necessary = True
                    item_name = "Grill"

                    grill_state = None
                    # find the grill at this position
                    for grill in world_state:
                        if (
                            grill["name"] == "steak"
                            and grill["position"][0] == j
                            and grill["position"][1] == i
                        ):
                            grill_state = grill
                            break

                    item_state = self.get_grill_state_as_language(grill_state)
                elif item == "C":
                    item_name = "Chicken dispenser"
                    necessary = True
                    if informative:
                        item_state = "The chicken dispenser has infinite chickens."
                elif item == "W":
                    necessary = True
                    item_name = "Sink"

                    sink_state = None
                    for sink in world_state:
                        if (
                            sink["name"] == "clean_plate"
                            and sink["position"][0] == j
                            and sink["position"][1] == i
                        ):
                            sink_state = sink
                            break

                    item_state = self.get_sink_state_as_language(sink_state)
                elif item == "M":
                    item_name = "Meat dispenser"
                    necessary = True

                    if informative:
                        item_state = "The meat dispenser has infinite meat."
                elif item == "B":
                    item_name = "Cutting Board"
                    necessary = True
                    cutting_board_state = None

                    # find the cutting board at this position
                    for cutting_board in world_state:
                        if (
                            cutting_board["name"] == "garnish"
                            and cutting_board["position"][0] == j
                            and cutting_board["position"][1] == i
                        ):
                            cutting_board_state = cutting_board
                            break

                    item_state = self.get_cutting_board_state_as_language(
                        cutting_board_state
                    )
                else:
                    item_name = "Empty square"

                    # resolve state based on where chefs are standing
                    if (
                        current_agent_state["position"][0] == j
                        and current_agent_state["position"][1] == i
                    ):
                        necessary = True
                        item_state = "You are standing here."
                    else:
                        for agent_state, agent_index in zip(other_agents_states, other_agent_indices):
                            if (
                                agent_state["position"][0] == j
                                and agent_state["position"][1] == i
                            ):
                                necessary = True
                                item_state = f"Chef {agent_index} is currently standing here."
                                break

                        if not necessary and informative:
                            item_state = "You can stand here."

                if verbose or necessary:
                    kitchen_items.append(f"\t({j},{i}): {item_name}. {item_state}")

        # format with newline operator
        return kitchen_overview, "\n".join(kitchen_items)

    def get_pot_state_as_language(self, pot_state):
        """Construct the pot state as a string from a dictionary containing its contents"""

        # resolve if pot has no ingredients
        if pot_state is None:
            return "The pot is empty. It has 0 ingredients."

        # obtain the pot state
        number_of_ingredients = len(pot_state["_ingredients"])
        is_ready = pot_state["is_ready"]
        is_cooking = pot_state["is_cooking"]
        cook_time = pot_state["cook_time"]
        cooking_timer = pot_state["_cooking_tick"]
        soup_ingredients = pot_state["_ingredients"]

        ingredients = []
        for ingredient in soup_ingredients:
            ingredients.append(ingredient["name"])

        ingredients = ", ".join(ingredients)

        pot_items = f"The pot is not empty. There are already {number_of_ingredients} ingredients in the pot: {ingredients}. "

        if not is_cooking:
            if is_ready:
                pot_items += \
                    f"The soup in the pot is finished cooking. It is ready to be picked up with a dish and delivered."
            else:
                pot_items += f"The soup has not started cooking yet."
        else:
            pot_items += f"The soup has already started cooking, but is not finished cooking. It is {cooking_timer} out of {cook_time} timesteps cooked."

        return pot_items

    def format_task_list(self, task_list):
        return "\n".join(
            [f"\tOption {i}: {task}" for i, task in enumerate(task_list, 1)]
        )

    def get_soup_as_language(self, soup_state):
        """Construct a soup as language"""

        return soup_state

    # uncommon functions: implementations in their respective inherited versions

    def get_relevant_subtasks(self, current_agent_state):
        raise NotImplementedError


class OriginalPrompt(Prompt):
    def __init__(self, grid_layout) -> None:
        super().__init__(grid_layout)

    def get_relevant_subtasks(self, current_agent_state):
        """obtain the relevant subtasks given the current agent state

        Return:
            1) list of subtask indices
            2) indices that cross reference to valid subtask indices
        """

        # get current held object
        held_object = current_agent_state["held_object"]

        if held_object is None:
            # remove subtask that need a held object
            # availlable_subtasks = "1. Pick up onion\n2. Pick up dish\n3. Pick up tomato\n5. Start cooking pot\n10. Do nothing"
            cross_reference = [1, 2, 5]
            task_list = [
                "Pick up the nearest onion",
                "Pick up the nearest dish",
                "Start cooking the nearest pot",
            ]
        else:
            # construct task list based on held object, and add to cross reference with switch case
            task_list = [f"Place {held_object['name']} on the nearest counter"]

            # construct cross reference list

            if held_object["name"] == "onion":
                # remove subtask that cant do with an onion held
                task_list.append("Put the onion in the nearest pot")
                cross_reference = [6, 8]

            elif held_object["name"] == "dish":
                # remove subtask that cant do with a dish held
                task_list.append("Pick up soup from the nearest pot with a dish")
                cross_reference = [6, 4]

            elif held_object["name"] == "soup":
                # remove subtask that cant do with a soup held

                ingredients_list = "onion, onion, onion"

                task_list.append(
                    f"Deliver the finished soup with ingredients: {ingredients_list} to the delivery location"
                )
                cross_reference = [6, 7]

            # TODO: add back bottom once tomato is reintroduced
            # elif held_object['name'] == "tomato":
            # remove subtask that cant do with a tomato held
            # availlable_subtasks = "6. Place object on counter\n9. Put tomato in pot\n10. Do nothing"

        # add the do nothing action at the end (applies to every action subset)
        cross_reference.append(10)
        task_list.append("Do nothing")

        return task_list, cross_reference


class SteakhousePrompt(Prompt):
    """Steakhouse is the advanced version of the Overcooked game"""

    def __init__(self, grid_layout) -> None:
        super().__init__(grid_layout)

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def build(
        self,
        state,
        mlam,
        agent_index: int,
        prompt: Dict[str, str],
        prompt_overrides: Dict[str, str] = {},
        ml_action_log: list = [],
        memory_depth: int = 0,
        **kwargs
    ):
        # obtain relevant state information
        current_agent_state = state.players[agent_index].to_dict()
        other_agents_states = [agent.to_dict() for i, agent in enumerate(state.players) if i != agent_index]
        grill_states = mlam.mdp.get_grill_states(state)
        cutting_board_states = mlam.mdp.get_chopping_board_states(state)
        world_state = state.to_dict().pop("objects")

        references = dict(prompt_overrides)

        task_list, cross_reference = self.get_relevant_subtasks(
            current_agent_state, world_state, grill_states, cutting_board_states
        )

        other_agent_indices = []
        for i, _ in enumerate(kwargs['agent_types']):
            # don't include the agent's own description
            if i != agent_index:
                other_agent_indices.append(i)
            
        chefs_description = self.format_other_agents_description(agent_index, kwargs['agent_types'])
        context = context.replace("{chef_extra_info}", chefs_description)

        chat_history_as_language = self.format_chat_history_as_language(agent_index, kwargs['chat_history'], state.timestep)

        # format prompt layout given the current states (this will replace all the placeholders in the prompt layout)
        dynamic_state = self.format_prompt_given_states(
            prompt['state'],
            world_state,
            current_agent_state,
            other_agents_states,
            chat_history_as_language,
            verbose=False,
            other_agent_indices=other_agent_indices
        )

        # if using memory, add memory to dynamic contents
        # if self.use_memory:
        #     dynamic_state = dynamic_state + f"\n{self.prompt['memory']}"

        order_list = state.order_list

        if len(order_list) > 0:
            recipe = None 
            if order_list[0] == "steak_dish":
                current_order = "steak"
                recipe = """
#### Steak dish (1 ingredient: meat)
    1. Pick up meat from the counter.
    2. Cook meat on the grill.
    3. Place a dirty plate in the sink.
    4. Clean the plate by interacting with the sink 3 times.
    5. Take the plate out of the sink and put cooked meat on your plate once you have the option to pick up steak from a grill.
    6. Once you are carrying a steak, deliver it to the delivery location immediately for a big tip.

    If you are holding a steak, you have a clean plate. So then deliver the steak to the delivery location. You do not have to prepare for another steak then.    
"""
            elif order_list[0] == "steak_onion_dish": 
                current_order = "steak and onion"
                recipe = """
#### Steak and onion dish (2 ingredients: meat and onion):
    1. Pick up meat from the counter. Make sure to check if any grills are empty whenever you pick up a meat. If there are no empty grills available, wait for a grill to be free. Otherwise, move on to the next step.
    2. Cook meat on the grill.
    3. Clean a dirty plate in the sink.
    4. Put cooked meat on the plate and place the plate on the counter if needed.
    5. Pick up an onion from the counter.
    6. Place the onion on the chopping board.
    7. Interact and chop the onion 2 times.
    8. After completing the steak dish (which includes a clean plate and steak), you will garnish it with chopped onions. To do so, take the plate with the steak (referred to as 'steak_dish') to the chopping board where the onions were previously chopped. Engage with the chopping board to add the onions as a garnish. This task should be performed by only one agent.
    9. Deliver the dish (called steak_onion_dish) to the delivery location (this is very important).

### Order of Actions to chop an onion on the chopping board
    1. Place the onion on the chopping board.
    2. Interact with the chopping board 2 times to chop the onion. Keep note of it.
    3. Once you are done, you must pick up the garnish with a clean plate THAT ALREADY HAS A STEAK ON IT. THIS IS VERY IMPORTANT.
"""

        # log all the items to replace in the dynamic_state
        to_replace = {
            "{current_order}": current_order,
            "{recipe}": recipe,
            "{task_list}": self.format_task_list(task_list),
        }

        # format the dynamic state
        dynamic_state = replace_all(dynamic_state, to_replace)

        # add relevant info to the references
        references['goal'] = replace_all(prompt.get('goal', None), {"{current_order}": current_order})
        references['state'] = dynamic_state
        references['context'] = context

        new_xref = []
        am = mlam

        counter_objects = mlam.mdp.get_counter_objects_dict(
            state, list(mlam.mdp.terrain_pos_dict["X"])
        )
        grill_states_dict = mlam.mdp.get_grill_states(state)
        for subtask_index in cross_reference:
            motion_goals = None
            # construct MLAM motion goals based on the subtask index
            if subtask_index == 1:
                motion_goals = am.pickup_onion_actions(counter_objects)
            elif subtask_index == 2:
                motion_goals = am.pickup_dirty_plate_actions(counter_objects)
            elif subtask_index == 3:
                motion_goals = am.pickup_counter_steak_actions(counter_objects)
                motion_goals += am.pickup_counter_onion_steak_actions(counter_objects)
            elif subtask_index == 4:
                motion_goals = am.chop_onion_on_board_actions(state)
            elif subtask_index == 5:
                motion_goals = am.pickup_meat_actions(counter_objects)
            elif subtask_index == 6:
                motion_goals = am.pickup_clean_plate_from_sink_actions(
                    counter_objects, state
                )
            elif subtask_index == 7:
                motion_goals = am.put_onion_on_board_actions(state)
            elif subtask_index == 8:
                motion_goals = am.put_meat_in_grill_actions(grill_states_dict)
            elif subtask_index == 9:
                motion_goals = am.rinse_plate_in_sink_actions(state)
            elif subtask_index == 10:
                motion_goals = am.put_dirty_plate_in_sink_actions(counter_objects, state)
            elif subtask_index == 11:
                motion_goals = am.add_garnish_to_steak_actions(state)
            elif subtask_index == 12:
                motion_goals = am.pickup_steak_with_clean_plate_actions(grill_states_dict)
            elif subtask_index == 13:
                motion_goals = am.place_obj_on_counter_actions(state)
            elif subtask_index == 14:
                motion_goals = am.deliver_soup_actions()
            elif subtask_index == 15:
                motion_goals = Action.STAY
            else:
                raise ValueError(f"Index {subtask_index} not found in subtasks")
            
            new_xref.append(motion_goals)

        # return the cross reference and corresponding modified prompt references
        return cross_reference, references

    def get_relevant_subtasks(self, current_agent_state, world_state, grill_states, cutting_board_states):
        """obtain the relevant subtasks given the current agent state for the steakhouse env

        Return:
            1) list of subtask indices
            2) indices that cross reference to valid subtask indices
        """

        total_grills = 0
        for key, value_list in grill_states.items():
          total_grills += len(value_list)

        # get current held object
        held_object = current_agent_state["held_object"]

        if held_object is None:
            # remove subtask that need a held object
            cross_reference = [1, 2, 5] 
            task_list = [
                "Pick up the nearest onion.",
                "Pick up the nearest dirty plate.",
                "Pick up nearest meat.",
            ]

            grid_layout = self.grid_layout
            dish_found = False

            for i in range(len(grid_layout)):
                for j in range(len(grid_layout[i])):
                    if grid_layout[i][j] == "X":
                        for counter in world_state:
                            if counter["position"][0] == j and counter["position"][1] == i and counter["name"] == "steak":
                                cross_reference.insert(2,3)
                                task_list.insert(2,"Pick up the steak dish from the counter.")
                                dish_found = True 
                                break
                            elif counter["position"][0] == j and counter["position"][1] == i and counter["name"] == "steak_onion":
                                 cross_reference.insert(2,3)
                                 task_list.insert(2,"Pick up the steak onion dish from the counter.")
                                 dish_found = True 
                                 break
                    else: 
                        continue
                if dish_found: 
                    break

            if world_state is not None: 
                clean_plate_ready = any(item['name'] == 'clean_plate' and item.get('is_ready', False) for item in world_state)
                plate_exists = any(item['name'] == 'clean_plate' for item in world_state) 
                steak_dish_exists = any(item['name'] == 'steak' and item.get('is_idle',False) for item in world_state) 
                steak_onion_dish_exists = any(item['name'] == 'steak_onion_dish' for item in world_state) 
                onion_not_fully_cut = len(cutting_board_states['full']) > 0

                if clean_plate_ready: 
                    cross_reference.insert(5, 6)
                    task_list.insert(5, "Pick up nearest clean plate.")
                elif not clean_plate_ready and plate_exists: #if there is a plate in the sink and it is not ready (not rinsed)
                    cross_reference.append(9)
                    task_list.append("Go to nearest sink and clean the dirty dish that is already inside the sink")
                
                #if steak_dish_exists or steak_onion_dish_exists:
                #    cross_reference.insert(2,3)
                #    task_list.insert(2,"Pick up the closest completed steak (or steak_onion) dish")
                
                if onion_not_fully_cut:
                    cross_reference.insert(3,4)
                    task_list.insert(3,"Go to nearest cutting board and finish chopping the onion in the cutting board.")

                

        else:
            # construct task list based on held object, and add to cross reference with switch case
            task_list = [f"Place {held_object['name']} on the nearest counter"]

            # construct cross reference list
            cross_reference = [13]  # default cross reference

            if held_object["name"] == "onion":
                # remove subtask that cant do with an onion held
                if len(cutting_board_states['empty']) >= 1: #checks if there is more than 1 cutting board available 
                 task_list.append("Put the onion on cutting board")
                 cross_reference.append(7)

            elif held_object["name"] == "meat":
                if 'empty' in grill_states and len(grill_states['empty']) > 0:
                    task_list.append("Cook the meat on the grill")
                    cross_reference.append(8)

            elif held_object["name"] == "dirty_plate":
                #if there is no dirty plate
                if world_state is not None: 
                 clean_plate_ready = any(item['name'] == 'clean_plate' and item.get('is_ready', False) for item in world_state)
                 plate_exists = any(item['name'] == 'clean_plate' for item in world_state) 
                 if not plate_exists: #check this condition again later on
                     task_list.append("Place the dirty plate in the sink")
                     cross_reference.append(10)

            elif held_object["name"] == "clean_plate":
                if world_state is not None:
                    is_ready_steak_present = any(item['name'] == 'steak' and item.get('is_ready', False) for item in world_state)
                    if is_ready_steak_present:
                        cross_reference.append(12)
                        task_list.append("Pick up a steak from the grill with a clean dish")

            elif held_object["name"] == "steak":
                task_list.append("Add garnish to the steak")
                task_list.append("Deliver the steak dish")
                cross_reference.append(11)
                cross_reference.append(14)

            elif held_object["name"] == "steak_onion":
                # remove subtask that cant do with a dish held
                task_list.append("Deliver the garnished steak dish")
                cross_reference.append(14)

            # TODO: add back bottom once tomato is reintroduced
            # elif held_object['name'] == "tomato":
            # remove subtask that cant do with a tomato held
            # availlable_subtasks = "6. Place object on counter\n9. Put tomato in pot\n10. Do nothing"

        # add the do nothing action at the end (applies to every action subset)
        cross_reference.append(15)
        task_list.append("Do nothing")

        return task_list, cross_reference

    def get_grill_state_as_language(self, grill_state):
        """Construct the grill state as a string from a dictionary containing its contents"""

        # resolve if grill has no ingredients
        if grill_state is None:
            return "The grill is empty. It has 0 ingredients."

        # obtain the grill state
        number_of_ingredients = len(grill_state["_ingredients"])
        is_ready = grill_state["is_ready"]
        is_cooking = grill_state["is_cooking"]
        cook_time = grill_state["cook_time"]
        cooking_timer = grill_state["_cooking_tick"]

        grill_items = "The Grill is not empty."

        if not is_cooking:
            if is_ready:
                grill_items += f"The steak is finished cooking and ready for collection. Pick up it up with a clean dish. This grill is not ready for use. You cannot add meat to this grill."
            else:
                grill_items += f"The steak has not started cooking yet. This grill is ready for use. You can add meat to this grill."
        else:
            grill_items += f"The steak has already started cooking, but is not finished cooking. It is {cooking_timer} out of {cook_time} timesteps cooked. This grill is not ready for use.  You cannot add meat to this grill."


        return grill_items

    def get_cutting_board_state_as_language(self, cutting_board_state):
        # resolve if cutting board has no ingredients
        if cutting_board_state is None:
            return "The cutting board is empty. It has 0 ingredients."

        # obtain the cutting board state
        number_of_ingredients = len(cutting_board_state["_ingredients"])
        is_ready = cutting_board_state["is_ready"]
        is_cooking = cutting_board_state["is_cooking"]
        cook_time = cutting_board_state["cook_time"]
        num_chops = cutting_board_state["_cooking_tick"]

        cutting_board_items = ""

        if not is_cooking:
            cutting_board_items = "The Cutting Board is not empty."
        else:
            cutting_board_items += f"The garnish has been partially chopped. It is {num_chops} out of {2} chops complete. You cannot add another onion here before this one is fully cut. Please finish cutting this onion in order to prepare for the garnish."
        if is_ready:
            cutting_board_items += f"The garnish is finished being cut. You cannot add another onion here before this one is picked up. It is ready for you to add the garnish to your steak."

        return cutting_board_items

    def get_sink_state_as_language(self, sink_state):
        # resolve if sink has no ingredients
        if sink_state is None:
            return "The sink is empty."

        is_done = sink_state["is_ready"]
        rinse_count = sink_state["rinse_count"]
        rinse_total = sink_state["rinse_total"]

        sink_items = ""

        if is_done:
            sink_items = "The dish is clean and ready to be picked up for use for your order."
        else:
            sink_items += f"The dirty dish has been partially cleaned. It is {rinse_count} out of {rinse_total} cleans complete. You must go to nearest sink and clean the dish to rinse it again. You cannot add anymore dirty dishes until you clean this dish in the sink."

        return sink_items

# this is a prompt for a non player advisor in the steakhouse game. They delegate high level tasks to each agent.
class SteakhouseAdvisorPrompt(SteakhousePrompt):
    def __init__(self, grid_layout) -> None:
        super().__init__(grid_layout)

    def get_relevant_subtasks(self, current_agent_state):
        raise NotImplementedError("The advisor agent doesn't take any actions")
    
    def format_prompt_given_states(
        self,
        prompt: str,
        world_state,
        agent_1_state,
        agent_2_state,
        agent_one_last_advice,
        agent_two_last_advice,
        agent_one_actions,
        agent_two_actions,
        verbose=True,
    ):
        """format the prompt given world states

        does not format the task list

        Input:
            prompt: the layout read from the layout file

        Returns:
            formatted prompt
        """

        # format the prompt

        # agent 1 state
        chef_one = self.get_agent_state_as_language(
            agent_1_state, first_person=False
        )
        prompt = prompt.replace("{chef_one_state}", chef_one)

        # other chef state
        chef_two = self.get_agent_state_as_language(
            agent_2_state, first_person=False
        )
        prompt = prompt.replace("{chef_two_state}", chef_two)

        # update kitchen state
        kitchen_overview, kitchen_items = self.get_kitchen_as_language(
            world_state, agent_1_state, agent_2_state, verbose=verbose
        )

                # remove Nones from the array
        clean_action_names = [x for x in agent_one_actions if x is not None]
        other_agent_clean_names = [x for x in agent_two_actions if x is not None]

        #remove duplicates if they are next to each other
        clean_action_names = [clean_action_names[i] for i in range(len(clean_action_names)) if i == 0 or clean_action_names[i] != clean_action_names[i-1]]
        other_agent_clean_names = [other_agent_clean_names[i] for i in range(len(other_agent_clean_names)) if i == 0 or other_agent_clean_names[i] != other_agent_clean_names[i-1]]
        
        prompt = prompt.replace("{chef_one_last_actions}", ", ".join(clean_action_names))
        prompt = prompt.replace("{chef_two_last_actions}", ", ".join(other_agent_clean_names))

        prompt = prompt.replace("{chef_one_previous_advice}", agent_one_last_advice)
        prompt = prompt.replace("{chef_two_previous_advice}", agent_two_last_advice)

        prompt = prompt.replace("{kitchen_items}", kitchen_items)
        prompt = prompt.replace("{kitchen_overview}", kitchen_overview)

        return prompt


class OvercookedPrompt(Prompt):
    """Overcooked prompt that is 

    This version of overcooked assumes old dynamics, which is used to compare to RL
    agent implementations.
    """
    def __init__(self, grid_layout) -> None:
        super().__init__(grid_layout)

    def get_relevant_subtasks(self, current_agent_state):
        """obtain the relevant subtasks given the current agent state

        Return:
            1) list of subtask indices
            2) indices that cross reference to valid subtask indices
        """

        # get current held object
        held_object = current_agent_state["held_object"]

        if held_object is None:
            # remove subtask that need a held object
            # availlable_subtasks = "1. Pick up onion\n2. Pick up dish\n3. Pick up tomato\n5. Start cooking pot\n10. Do nothing"
            cross_reference = [1, 2]
            task_list = [
                "Pick up the nearest onion",
                "Pick up the nearest dish",
            ]
        else:
            # construct task list based on held object, and add to cross reference with switch case
            task_list = [f"Place the {clean_name(held_object['name'])} on the nearest counter"]

            # construct cross reference list

            if held_object["name"] == "onion":
                # remove subtask that cant do with an onion held
                task_list.append("Put the onion in the nearest pot")
                cross_reference = [6, 8]

            elif held_object["name"] == "dish":
                # remove subtask that cant do with a dish held
                task_list.append("Pick up soup from the nearest pot with a dish")
                cross_reference = [6, 4]

            elif held_object["name"] == "soup":
                # remove subtask that cant do with a soup held
                task_list.append(
                    "Deliver the finished soup you are holding to the delivery location"
                )
                cross_reference = [6, 7]

        # add the do nothing action at the end (applies to every action subset)
        cross_reference.append(10)
        task_list.append("Do nothing")

        return task_list, cross_reference


class SteakhousePrompt2(SteakhousePrompt):
    """Updated version of the steakhouse prompt that represents components according to the following 
    
    NOTE: get_relevant_subtasks is deprecated in this version. The build function returns the cross reference and references
    """

    def __init__(self, grid_layout) -> None:
        super().__init__(grid_layout)
    
    def get_grill_state_as_language(self, grill_state, grill_name="The grill"):
        """Construct the grill state as a string from a dictionary containing its contents"""

        # resolve if grill has no ingredients
        if grill_state is None:
            return f"{grill_name} is empty."

        # obtain the grill state
        number_of_ingredients = len(grill_state["_ingredients"])
        is_ready = grill_state["is_ready"]
        is_cooking = grill_state["is_cooking"]
        cook_time = grill_state["cook_time"]
        cooking_timer = grill_state["_cooking_tick"]

        # grill_items = f"{grill_name} is not empty."

        if not is_cooking:
            if is_ready:
                grill_items = f"{grill_name} has a cooked steak that is finished cooking and ready to be picked up with a clean dish. You cannot add meat to this grill."
            else:
                grill_items = f"{grill_name} has a raw meat that has not started cooking yet."
        else:
            grill_items = f"{grill_name} has a raw meat that has started cooking, but is not finished. It is {cooking_timer} out of {cook_time} timesteps cooked. You cannot add meat to this grill."

        return grill_items
    
    def get_pot_state_as_language(self, pot_state, pot_name="The pot"):
        """Construct the pot state as a string from a dictionary containing its contents"""

        # resolve if grill has no ingredients
        if pot_state is None:
            return f"{pot_name} is empty."

        # obtain the grill state
        number_of_ingredients = len(pot_state["_ingredients"])
        is_ready = pot_state["is_ready"]
        is_cooking = pot_state["is_cooking"]
        cook_time = pot_state["cook_time"]
        cooking_timer = pot_state["_cooking_tick"]


        if not is_cooking:
            if is_ready:
                pot_items = f"{pot_name} has a cooked chicken that is finished cooking and ready to be picked up with a clean dish. You cannot add chicken to this pot."
            else:
                pot_items = f"{pot_name} has a raw chicken that has not started cooking yet."
        else:
            pot_items = f"{pot_name} has a raw chicken that has started cooking, but is not finished. It is {cooking_timer} out of {cook_time} timesteps cooked. You cannot add chicken to this grill."

        return pot_items

    def get_cutting_board_state_as_language(self, cutting_board_state, chopping_board_name="The chopping board"):
        # resolve if chopping board has no ingredients
        if cutting_board_state is None:
            return f"{chopping_board_name} is empty. It has 0 ingredients."

        # obtain the chopping board state
        number_of_ingredients = len(cutting_board_state["_ingredients"])
        is_ready = cutting_board_state["is_ready"]
        is_cooking = cutting_board_state["is_cooking"]
        cook_time = cutting_board_state["cook_time"]
        num_chops = cutting_board_state["_cooking_tick"]

        cutting_board_items = ""

        if not is_cooking:
            cutting_board_items = f"{chopping_board_name} is not empty."
        else:
            cutting_board_items = f"{chopping_board_name} has an onion. It is {num_chops} out of {2} chops complete."

        if is_ready:
            cutting_board_items += f" The onion on {chopping_board_name} is fully chopped."

        return cutting_board_items

    def get_sink_state_as_language(self, sink_state, sink_name="The sink"):
        # resolve if sink has no ingredients
        if sink_state is None:
            return f"{sink_name} is empty."

        is_done = sink_state["is_ready"]
        rinse_count = sink_state["rinse_count"]
        rinse_total = sink_state["rinse_total"]

        if is_done:
            return f"There is a clean plate in {sink_name}. It is {rinse_count} out of {rinse_total} rinses complete."

        return f"The dirty plate in {sink_name} has been partially cleaned. It is {rinse_count} out of {rinse_total} rinses complete."

    def find_shortest_distance(self, mlam, p1, p_arr, agent_index, agent_states): # other_player_positions, other_agent_indices):
        """Credits to the original authors of this function, UCSC Eric AI Lab
        
        Reference:
        https://github.com/eric-ai-lab/llm_coordination/blob/master/src/llm_coordination_agents/overcooked_action_manager.py
        """
        other_player_positions = [agent_states[i]["position"] for i in range(len(agent_states)) if i != agent_index]
        all_player_positions = [agent_states[i]["position"] for i in range(len(agent_states))]

        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Up, Down, Right, Left

        # Given a starting point and an array of points, find the shortest distance to any point in the array
        shortest_distances = {p:(math.inf, -1) for p in p_arr} # (distance, is_blocked)
        for p2 in p_arr:
            queue = deque([(p1, 0, -1)]) # (position, distance, is_blocked)
            visited = set()

            while queue:
                current_pos, distance, is_blocked = queue.popleft()

                if current_pos == p2 and distance < shortest_distances[p2]:
                    shortest_distances[p2] = (distance, is_blocked)

                if current_pos in visited:
                    continue

                visited.add(current_pos)

                for dx, dy in directions:
                    new_x = current_pos[0] + dx
                    new_y = current_pos[1] + dy
                    new_pos = (new_x, new_y)

                    to_visit = mlam.mdp.get_terrain_type_at_pos(new_pos)

                    if (to_visit == " ") and new_pos not in visited:
                        if new_pos in other_player_positions:
                            # add to the queue, but it is blocked through this path
                            blocking_agent = all_player_positions.index(new_pos)
                            queue.append((new_pos, distance + 1, blocking_agent))
                        else:
                            queue.append((new_pos, distance + 1, is_blocked))
                    elif new_pos == p2:
                        # this is the new position; this basically means that we can get right up next to the object
                        shortest_distances[p2] = (distance, is_blocked)
        
        min_dist = math.inf
        min_dest = None
        point = None
        is_blocked = -1
        for p, (d, blocked) in shortest_distances.items():
            if d < min_dist:
                min_dist = d 
                point = p
                is_blocked = blocked

        if is_blocked != -1:
            name = NAMES[is_blocked]
            min_dest = f'{min_dist} units away, but blocked by {name}.'
        elif min_dist == math.inf:
            min_dist = 'infinite'
            min_dest = "unreachable."
        else:
            min_dest = f'{min_dist} units away.'

        if min_dest is None:
            min_dest = f'{min_dist} units away.'

        return min_dist, min_dest, point
    
    def get_grid_as_dict(self):
        """Creates a dict, something like {"Counter": [(0,0), (1,0)]}
        """
        grid_dict = {}
        grid_layout = self.grid_layout

        for j in range(len(grid_layout)):
            for i in range(len(grid_layout[j])):
                item = grid_layout[j][i]
                if item in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    item = " "

                if item in grid_dict:
                    grid_dict[item].append((i, j))
                else:
                    grid_dict[item] = [(i, j)]

        return grid_dict

    def get_kitchen_as_language(self, state, agent_index, mlam):
        """Construct the kitchen state as a string from a dictionary containing its contents

        Kitchen Example:
            <Environment Details>: c0 contains 1 out of 3 onions. c0 is off. soup in c0 is not cooking. c1 contains 0 ,→ out of 3 onions. c1 is off. soup in c1 is not cooking.

        Location Example:
            <My Location Information>: o0 is 0 units away. o1 is 1 units away. p0 is 3 units away. c0 is 6 units away , blocked by Bob. c1 is 7 units away. d0 is 4 units away. s0 is 1 units away. s1 is 0 units away. s2 ,→ is 1 units away. s3 in 2 units away. Closest empty kitchen counter k12 is 1 units away.
            <Bob's Location Information>: o0 is blocked by Alice. o1 is 7 units away. p0 is 3 units away. c0 is 0 units

        Actions Example:
            Available Actions: [place onion in c0, place onion in c1., place onion on s0., place onion on s1., place ,→ onion on s2, place onion on s3., place onion on k12., wait., move away.]"""

        current_agent_state = state.players[agent_index].to_dict()
        grill_states = mlam.mdp.get_grill_states(state)
        cutting_board_states = mlam.mdp.get_chopping_board_states(state)
        world_state = state.to_dict().pop("objects")


        location_info_array = []
        kitchen_items_array = []

        # counter_objects = mlam.mdp.get_counter_objects_dict(
        #     state, list(mlam.mdp.terrain_pos_dict["X"])
        # )
        # pot_states_dict = mlam.mdp.get_pot_states(state)
        # grill_states_dict = mlam.mdp.get_grill_states(state)
        # cutting_board_states_dict = mlam.mdp.get_chopping_board_states(state)

        grid_dict = self.get_grid_as_dict()

        # get all empty and non empty counters
        empty_counters = []
        non_empty_counters = []
        for pos in grid_dict["X"]:
            non_e = False
            for counter in world_state:
                if counter["position"] == pos:
                    non_empty_counters.append(counter)
                    non_e = True
                    break
            
            if not non_e:
                empty_counters.append(pos)

        # construct a location information array for each player
        for i, agent in enumerate(state.players):
            prefix = f"<{self.names[i]}'s Location Information>: "
            if i == agent_index:
                prefix = f"<Your ({self.names[i]}) Location Information>: "
                
            agent = agent.to_dict()
            other_agents_states = [agent.to_dict() for j, agent in enumerate(state.players) if j != i]

            agent_states = [agent.to_dict() for _, agent in enumerate(state.players)]

            positions = []
            for k, v in grid_dict.items():
                if k == " ":
                    continue

                # shortest distance to any of `v` relevant points
                if k == "X":
                    # only check empty counters
                    min_dist, min_dest = self.find_shortest_distance(mlam, agent["position"], empty_counters, i, agent_states)
                else:
                    min_dist, min_dest = self.find_shortest_distance(mlam, agent["position"], v, i, agent_states)

                if min_dist == "infinite":
                    distance = min_dest # "blocked by another agent."
                else:
                    distance = f"{min_dist} units away."

                # resolve all of the items
                if k == "M":
                    # meat dispenser

                    positions.append(f"The meat dispenser is {distance}")
                elif k == "D":
                    # dirty plate dispenser

                    positions.append(f"The dirty plate dispenser is {distance}")
                elif k == "O":
                    # onion dispenser

                    positions.append(f"The onion dispenser is {distance}")
                elif k == "X":
                    # counter

                    # find the nearest counter
                    positions.append(f"The nearest empty counter is {distance}")

                    # check all counters for an object
                    for counter in non_empty_counters:
                        new_min_dist, new_min_dest = self.find_shortest_distance(mlam, agent["position"], [counter["position"]], i, agent_states)
                        positions.append(f"The counter with {clean_name(counter['name'])} is {new_min_dist} units away.")

                        kitchen_items_array.append(f"There is a counter with a {clean_name(counter['name'])}.")

                elif k == "S":
                    # delivery location
                    positions.append(f"The nearest delivery location is {distance}")
                elif k == "G":
                    positions.append(f"The grill is {distance}")

                    # get the grill state
                    if i == agent_index:
                        for pos in v:
                            grill_state = None
                            for grill in world_state:
                                if (
                                    grill["name"] == "steak"
                                    and grill["position"] == pos
                                ):
                                    grill_state = grill
                                    break

                            # check on grill state
                            item_state = self.get_grill_state_as_language(grill_state)
                            kitchen_items_array.append(item_state)
                        
                elif k == "W":
                    # sink
                    positions.append(f"The sink is {distance}")

                    # get the sink state
                    if i == agent_index:
                        for pos in v:
                            sink_state = None
                            for sink in world_state:
                                if (
                                    sink["name"] == "clean_plate"
                                    and sink["position"] == pos
                                ):
                                    sink_state = sink
                                    break

                            # check on grill state
                            item_state = self.get_sink_state_as_language(sink_state)
                            kitchen_items_array.append(item_state)
                elif k == "B":
                    # chopping board
                    positions.append(f"The chopping board is {distance}")

                    # get the chopping board state
                    if i == agent_index:
                        for pos in v:
                            # find the chopping board at this position
                            cutting_board_state = None
                            for cutting_board in world_state:
                                if (
                                    cutting_board["name"] == "garnish"
                                    and cutting_board["position"] == pos
                                ):
                                    cutting_board_state = cutting_board
                                    break

                            # check on grill state
                            item_state = self.get_cutting_board_state_as_language(cutting_board_state)
                            kitchen_items_array.append(item_state)

            location_info_array.append(prefix + " ".join(positions))

        # calculate the cross reference and task list
        task_list = []
        cross_reference = []

        # get current held object
        held_object = current_agent_state["held_object"]

        # get all relevant states
        counter_objects = mlam.mdp.get_counter_objects_dict(
            state, list(mlam.mdp.terrain_pos_dict["X"])
        )
        pot_states_dict = mlam.mdp.get_pot_states(state)
        grill_states_dict = mlam.mdp.get_grill_states(state)
        cutting_board_states_dict = mlam.mdp.get_chopping_board_states(state)
        sink_states_dict = mlam.mdp.get_sink_states(state)

        if held_object is None:
            # add all the base tasks that the agent can do
            cross_reference = [
                mlam.pickup_onion_actions(counter_objects, only_use_dispensers=True),
                mlam.pickup_dirty_plate_actions(counter_objects, only_use_dispensers=True),
                mlam.pickup_meat_actions(counter_objects, only_use_dispensers=True),
            ] 
            task_list = [
                "Pick up an onion from the onion dispenser",
                "Pick up a dirty plate from the dirty plate dispenser.",
                "Pick up a meat from the meat dispenser.",
            ]

            # checl al
            for counter in non_empty_counters:
                # this is a special counter, so this must be added to the kitchen items
                task_list.append(f"Pick up {clean_name(counter['name'])} from counter.")
                cross_reference.append(mlam.pickup_item_specific_counter(counter['position']))

            # integrate these changes
            if world_state is not None: 
                for sink in sink_states_dict["ready"]:
                    # they can pick up the clean dish if wanted
                    cross_reference.append(
                        mlam.pickup_clean_plate_from_sink_actions(
                            counter_objects, state
                        )
                    )
                    task_list.append("Pick up the clean plate from the sink.")

                for sink in sink_states_dict["full"]:
                    # they can pick up the clean dish if wanted
                    cross_reference.append(
                        mlam.rinse_plate_in_sink_actions(state)
                    )
                    task_list.append("Do one rinse of the dirty dish in the sink.")

                for board in cutting_board_states_dict:
                    onion_not_fully_cut = len(cutting_board_states['full']) > 0

                    # check if the chopping board is ready
                    if onion_not_fully_cut:
                        cross_reference.append(
                            mlam.chop_onion_on_board_actions(state)
                        )
                        task_list.append("Do one chop of the onion on the chopping board.")

            # check the empty counter tasks and add it to the task list
        else:
            # construct task list based on held object, and add to cross reference with switch case
            task_list = [
                f"Place the {clean_name(held_object['name'])} in hand on the nearest counter."
            ]
            cross_reference = [
                mlam.place_obj_on_counter_actions(state)
            ]

            if held_object["name"] == "onion":
                # remove subtask that cant do with an onion held
                if len(cutting_board_states['empty']) >= 1:
                    task_list.append("Put onion in hand on the chopping board.")
                    cross_reference.append(
                        mlam.put_onion_on_board_actions(state)
                    )
            elif held_object["name"] == "meat":
                if 'empty' in grill_states and len(grill_states['empty']) > 0:
                    task_list.append("Put the meat in hand on the grill to cook.")
                    cross_reference.append(
                        mlam.put_meat_in_grill_actions(grill_states_dict)
                    )

            elif held_object["name"] == "dirty_plate":
                #if there is no dirty plate
                if world_state is not None:
                    # for each empty sink
                    for sink in sink_states_dict["empty"]:
                        task_list.append("Place dirty plate in hand in the sink.")
                        cross_reference.append(
                            mlam.put_dirty_plate_in_sink_actions(counter_objects, state)
                        )

            elif held_object["name"] == "clean_plate":
                # check all grills for a ready steak
                if world_state is not None:
                    for grill in grill_states_dict['ready']:
                        task_list.append("Use clean plate in hand to pick up steak from the grill.")
                        cross_reference.append(
                            mlam.pickup_steak_with_clean_plate_actions(grill_states_dict)
                        )

            elif held_object["name"] == "steak":
                # check if there is a garnish available
                if len(cutting_board_states_dict['ready']) > 0:
                    task_list.append("Add garnish to the steak dish in hand.")
                    cross_reference.append(
                        mlam.add_garnish_to_steak_actions(state)
                    )
                task_list.append("Deliver the steak dish in hand to delivery location.")
                cross_reference.append(
                    mlam.deliver_soup_actions()
                )

            elif held_object["name"] == "steak_onion":
                # remove subtask that cant do with a dish held
                task_list.append("Deliver the steak onion dish in hand to delivery location.")
                cross_reference.append(
                    mlam.deliver_soup_actions()
                )

        # add the do nothing action at the end (applies to every action subset)
        task_list.append("Wait for 5 timesteps.")
        cross_reference.append(
            Action.STAY
        )
        # TODO: add move away action

        # format and send them back
        location_info = "\n".join(location_info_array)
        kitchen_items = " ".join(kitchen_items_array)
        task_list_as_language = self.format_task_list(task_list)
        # cross reference is already in the correct format: List of MLA's

        return location_info, kitchen_items, task_list_as_language, cross_reference

    def build(
        self,
        state,
        mlam,
        agent_index: int,
        prompt: Dict[str, str],
        prompt_overrides: Dict[str, str] = {},
        ml_action_log: list = [],
        memory_depth: int = 0,
        **kwargs
    ):
        # obtain relevant state information
        current_agent_state = state.players[agent_index].to_dict()
        other_agents_states = [agent.to_dict() for i, agent in enumerate(state.players) if i != agent_index]
        grill_states = mlam.mdp.get_grill_states(state)
        cutting_board_states = mlam.mdp.get_chopping_board_states(state)
        world_state = state.to_dict().pop("objects")

        references = dict(prompt_overrides)

        # format the context
        chefs_description = self.format_other_agents_description(agent_index, kwargs['agent_types'])

        references['context'] = replace_all(
            prompt['context'],
            {
                "{agent_name}": self.names[agent_index],
                "{other_agents_description}": chefs_description,
                "{environment_description}": self.grid_config['description'],
            }
        )

        inventory_as_language = self.format_inventory_as_language(state, agent_index)

        location_info, kitchen_items, task_list_as_language, cross_reference = self.get_kitchen_as_language(state, agent_index, mlam)
        message_history_as_language = self.format_chat_history_as_language(agent_index, kwargs['chat_history'], state.timestep)

        # action history
        action_history_as_language = self.format_action_history_as_language(current_agent_state['action_history'], state.timestep, memory_depth=memory_depth)

        # get the current order and next order
        current_order = "nothing"
        next_order = "nothing"
        order_list = state.order_list

        if len(order_list) > 0:
            current_order = clean_name(order_list[0])
            if len(order_list) > 1:
                next_order = clean_name(order_list[1])

#         order_list = state.order_list
#         if len(order_list) > 0:
#             recipe = None 
#             if order_list[0] == "steak_dish":
#                 current_order = "steak"
#                 recipe = """
# #### Steak dish (1 ingredient: meat)
#     1. Pick up meat from the counter.
#     2. Cook meat on the grill.
#     3. Place a dirty plate in the sink.
#     4. Clean the plate by interacting with the sink 3 times.
#     5. Take the plate out of the sink and put cooked meat on your plate once you have the option to pick up steak from a grill.
#     6. Once you are carrying a steak, deliver it to the delivery location immediately for a big tip.

#     If you are holding a steak, you have a clean plate. So then deliver the steak to the delivery location. You do not have to prepare for another steak then.    
# """
#             elif order_list[0] == "steak_onion_dish": 
#                 current_order = "steak and onion"
#                 recipe = """
# #### Steak and onion dish (2 ingredients: meat and onion):
#     1. Pick up meat from the counter. Make sure to check if any grills are empty whenever you pick up a meat. If there are no empty grills available, wait for a grill to be free. Otherwise, move on to the next step.
#     2. Cook meat on the grill.
#     3. Clean a dirty plate in the sink.
#     4. Put cooked meat on the plate and place the plate on the counter if needed.
#     5. Pick up an onion from the counter.
#     6. Place the onion on the chopping board.
#     7. Interact and chop the onion 2 times.
#     8. After completing the steak dish (which includes a clean plate and steak), you will garnish it with chopped onions. To do so, take the plate with the steak (referred to as 'steak_dish') to the chopping board where the onions were previously chopped. Engage with the chopping board to add the onions as a garnish. This task should be performed by only one agent.
#     9. Deliver the dish (called steak_onion_dish) to the delivery location (this is very important).

# ### Order of Actions to chop an onion on the chopping board
#     1. Place the onion on the chopping board.
#     2. Interact with the chopping board 2 times to chop the onion. Keep note of it.
#     3. Once you are done, you must pick up the onion with a clean plate THAT ALREADY HAS A STEAK ON IT. THIS IS VERY IMPORTANT.
# """
        # format the goal
        references['goal'] = replace_all(
            prompt['goal'],
            {
                "{current_order}": current_order,
                "{next_order}": next_order,
            }
        )

        references['state'] = replace_all(
            prompt['state'],
            {
                "{inventory}": inventory_as_language,
                "{location_info}": location_info,
                "{kitchen_items}": kitchen_items,
                "{current_order}": current_order,
                "{next_order}": next_order,
                "{message_history}": message_history_as_language,
                # "{recipe}": recipe,
                "{task_list}": task_list_as_language,
                "{action_history}": action_history_as_language,
            }
        )

        return cross_reference, references
    

class SteakhousePrompt3(SteakhousePrompt2):
    def __init__(self, grid_layout) -> None:
        super().__init__(grid_layout)

    def format_timer_as_language(self, timestep, horizon):
        return f"{timestep} timesteps have passed, and {horizon - timestep} timesteps remain in the game."

    def xref_exact_location(self, agent_state):
        location_info = self.grid_config['location_info']
        mapping = self.grid_config['location_mapping']
        
        # get the agent's position
        p = agent_state['position']
        section = location_info[p[1]][p[0]]

        if section in mapping:
            return mapping[section]
        else:
            return "kitchen"


    def get_kitchen_as_language(self, state, agent_index, mlam, explicit_language=True):
        """Construct the kitchen state as a string from a dictionary containing its contents

        Kitchen Example:
            <Environment Details>: c0 contains 1 out of 3 onions. c0 is off. soup in c0 is not cooking. c1 contains 0 ,→ out of 3 onions. c1 is off. soup in c1 is not cooking.

        Location Example:
            <My Location Information>: o0 is 0 units away. o1 is 1 units away. p0 is 3 units away. c0 is 6 units away , blocked by Bob. c1 is 7 units away. d0 is 4 units away. s0 is 1 units away. s1 is 0 units away. s2 ,→ is 1 units away. s3 in 2 units away. Closest empty kitchen counter k12 is 1 units away.
            <Bob's Location Information>: o0 is blocked by Alice. o1 is 7 units away. p0 is 3 units away. c0 is 0 units

        Actions Example:
            Available Actions: [place onion in c0, place onion in c1., place onion on s0., place onion on s1., place ,→ onion on s2, place onion on s3., place onion on k12., wait., move away.]"""

        current_agent_state = state.players[agent_index].to_dict()
        grill_states = mlam.mdp.get_grill_states(state)
        cutting_board_states = mlam.mdp.get_chopping_board_states(state)
        world_state = state.to_dict().pop("objects")


        location_info_array = []
        kitchen_items_array = []

        grid_dict = self.get_grid_as_dict()

        # TODO: perhaps automate environment processing?
        shared_counter_locations = self.grid_config['shared_counter_locations'] # [(5, 3), (6, 5)]
        shared_counter_locations = [tuple(pt) for pt in shared_counter_locations]

        # get all empty and non empty counters
        empty_counters = []
        non_empty_counters = []

        empty_shared_counters = []
        non_empty_shared_counters = []
        for pos in grid_dict["X"]:
            non_e = False
            for counter in world_state:
                if counter["position"] == pos:
                    if pos in shared_counter_locations:
                        non_empty_shared_counters.append(counter)
                    else:
                        non_empty_counters.append(counter)
                    non_e = True
                    break
            
            if not non_e:
                if pos in shared_counter_locations:
                    empty_shared_counters.append(pos)
                else:
                    empty_counters.append(pos)


        # construct a location information array for each player
        for i, agent in enumerate(state.players):
            exact_location = self.xref_exact_location(agent.to_dict())

            prefix = f"<{self.names[i]}'s Current Location Information>: {self.names[i]} is in the {exact_location}. "
            if i == agent_index:
                prefix = f"<Your ({self.names[i]}) Current Location Information>: You are in the {exact_location}. "
                

            agent = agent.to_dict()

            agent_states = [agent.to_dict() for _, agent in enumerate(state.players)]

            positions = []
            for k, v in grid_dict.items():
                if k == " ":
                    continue

                # shortest distance to any of `v` relevant points
                if k == "X":
                    # only check empty counters
                    _, distance, point = self.find_shortest_distance(mlam, agent["position"], empty_counters, i, agent_states)
                else:
                    _, distance, point = self.find_shortest_distance(mlam, agent["position"], v, i, agent_states)

                # resolve all of the items
                if k == "M":
                    # meat dispenser

                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The meat dispenser is {distance}")
                elif k == "D":
                    # dirty plate dispenser

                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The dirty plate dispenser is {distance}")
                elif k == "O":
                    # onion dispenser

                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The onion dispenser is {distance}")

                elif k == "C":
                    # chicken dispenser

                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The chicken dispenser is {distance}")

                elif k == "X":
                    # counter

                    if len(empty_counters) == 0:
                        positions.append("There are no empty general counters.")
                    else:
                        # find the nearest counter
                        if explicit_language:
                            _, _, point = self.find_shortest_distance(mlam, agent["position"], empty_counters, i, agent_states)
                            if point != None:
                                name = self.grid_language_reference[point]
                                positions.append(
                                    f"The nearest empty general counter, {name}, is {distance}"
                                )
                            else:
                                positions.append(f"The nearest empty general counter is {distance}")

                    # check all counters for an object
                    for counter in non_empty_counters:
                        # new_min_dist, new_min_dest, _ = self.find_shortest_distance(mlam, agent["position"], [counter["position"]], i, agent_states)
                        _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [counter["position"]], i, agent_states)

                        # get language info
                        pre1 = "The counter"
                        pre2 = "There is a counter with a"
                        if explicit_language:
                            c_pos = counter["position"]
                            pre1 = self.grid_language_reference[c_pos]
                            pre2 = f"{pre1} has a"

                        positions.append(f"{pre1} with {clean_name(counter['name'])} is {distance}")

                        if i == agent_index:
                            kitchen_items_array.append(f"{pre2} {clean_name(counter['name'])}.")

                    # add all shared counters
                    for counter in non_empty_shared_counters:
                        # new_min_dist, new_min_dest, _ = self.find_shortest_distance(mlam, agent["position"], [counter["position"]], i, agent_states)
                        _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [counter["position"]], i, agent_states)

                        if explicit_language:
                            name = self.grid_language_reference[counter["position"]]

                            positions.append(f"{name} with {clean_name(counter['name'])} is {distance}")

                            if i == agent_index:
                                kitchen_items_array.append(f"{name} has a {clean_name(counter['name'])}.")
                        else:
                            sc = shared_counter_locations.index(counter["position"]) + 1

                            positions.append(f"Shared counter {sc} with {clean_name(counter['name'])} is {distance}")

                            if i == agent_index:
                                kitchen_items_array.append(f"Shared counter {sc} has a {clean_name(counter['name'])}.")

                    # add all empty shared counters (these are positions)
                    for counter in empty_shared_counters:
                        # new_min_dist, new_min_dest, _ = self.find_shortest_distance(mlam, agent["position"], [counter], i, agent_states)
                        _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [counter], i, agent_states)

                        if explicit_language:
                            name = self.grid_language_reference[counter]

                            positions.append(f"{name} is {distance}")
                        else:
                            sc = shared_counter_locations.index(counter) + 1
                            positions.append(f"Empty shared counter {sc} is {distance}")

                elif k == "S":
                    # delivery location

                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The nearest delivery location is {distance}")
                elif k == "G":

                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The grill is {distance}")

                    # get the grill state
                    if i == agent_index:
                        for pos in v:
                            grill_state = None
                            for grill in world_state:
                                if (
                                    grill["name"] == "steak"
                                    and grill["position"] == pos
                                ):
                                    grill_state = grill
                                    break

                            grill_name = None
                            if explicit_language:
                                grill_name = self.grid_language_reference[pos]

                            # check on grill state
                            item_state = self.get_grill_state_as_language(grill_state, grill_name=grill_name)
                            kitchen_items_array.append(item_state)
                        
                elif k == "W":
                    # sink
                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The sink is {distance}")

                    # get the sink state
                    if i == agent_index:
                        for pos in v:
                            sink_state = None
                            for sink in world_state:
                                if (
                                    sink["name"] == "clean_plate"
                                    and sink["position"] == pos
                                ):
                                    sink_state = sink
                                    break

                            sink_name = None
                            if explicit_language:
                                sink_name = self.grid_language_reference[pos]

                            # check on grill state
                            item_state = self.get_sink_state_as_language(sink_state, sink_name=sink_name)
                            kitchen_items_array.append(item_state)
                elif k == "B":
                    # chopping board
                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The chopping board is {distance}")

                    # get the chopping board state
                    if i == agent_index:
                        for pos in v:
                            # find the chopping board at this position
                            cutting_board_state = None
                            for cutting_board in world_state:
                                if (
                                    cutting_board["name"] == "garnish"
                                    and cutting_board["position"] == pos
                                ):
                                    cutting_board_state = cutting_board
                                    break

                            chopping_board_name = None
                            if explicit_language:
                                chopping_board_name = self.grid_language_reference[pos]

                            # check on grill state
                            item_state = self.get_cutting_board_state_as_language(cutting_board_state, chopping_board_name=chopping_board_name)
                            kitchen_items_array.append(item_state)

                elif k == "P":
                    # pot
                    if explicit_language:
                        for q in v:
                            _, distance, _ = self.find_shortest_distance(mlam, agent["position"], [q], i, agent_states)
                            name = self.grid_language_reference[q]
                            positions.append(
                                f"{name} is {distance}"
                            )
                    else:
                        positions.append(f"The pot is {distance}")

                    # get the pot state
                    if i == agent_index:
                        for pos in v:
                            # find the pot at this position
                            pot_state = None
                            for pot in world_state:
                                if (
                                    pot["name"] == "boiled_chicken"
                                    and pot["position"] == pos
                                ):
                                    pot_state = pot
                                    break

                            pot_name = None
                            if explicit_language:
                                pot_name = self.grid_language_reference[pos]

                            # check on grill state
                            item_state = self.get_pot_state_as_language(pot_state, pot_name=pot_name)
                            kitchen_items_array.append(item_state)


            location_info_array.append(prefix + " ".join(positions))

        # calculate the cross reference and task list
        task_list = []
        cross_reference = []

        # get current held object
        held_object = current_agent_state["held_object"]

        # get all relevant states
        counter_objects = mlam.mdp.get_counter_objects_dict(
            state, list(mlam.mdp.terrain_pos_dict["X"])
        )
        pot_states_dict = mlam.mdp.get_pot_states(state)
        grill_states_dict = mlam.mdp.get_grill_states(state)
        cutting_board_states_dict = mlam.mdp.get_chopping_board_states(state)
        sink_states_dict = mlam.mdp.get_sink_states(state)
        pot_states_dict = mlam.mdp.get_pot_states(state)

        if held_object is None:
            # TODO: add scalability/choosing for multiple dispensers
            if explicit_language:
                # add all the base tasks that the agent can do
                for onion_disp in grid_dict['O']:
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [onion_disp], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    task_list.append(f"Pick up an onion from {self.grid_language_reference[ onion_disp ]}.")
                    cross_reference.append((
                        mlam._get_ml_actions_for_positions([onion_disp]),
                        1
                    ))

                for dirty_plate_disp in grid_dict['D']:
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [dirty_plate_disp], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    task_list.append(f"Pick up a dirty plate from {self.grid_language_reference[ dirty_plate_disp ]}.")
                    cross_reference.append((
                        mlam._get_ml_actions_for_positions([dirty_plate_disp]),
                        2
                    ))
                
                if 'M' in grid_dict:
                    for meat_disp in grid_dict['M']:
                        d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [meat_disp], agent_index, agent_states)
                        if d == "infinite":
                            continue

                        task_list.append(f"Pick up a meat from {self.grid_language_reference[ meat_disp ]}.")
                        cross_reference.append((
                            mlam._get_ml_actions_for_positions([meat_disp]),
                            3
                        ))
                
                if 'C' in grid_dict:
                    for chicken_disp in grid_dict['C']:
                        d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [chicken_disp], agent_index, agent_states)
                        if d == "infinite":
                            continue

                        task_list.append(f"Pick up a chicken from {self.grid_language_reference[ chicken_disp ]}.")
                        cross_reference.append((
                            mlam._get_ml_actions_for_positions([chicken_disp]),
                            19
                        ))
            else:
                task_list = [
                    "Pick up an onion from the onion dispenser",
                    "Pick up a dirty plate from the dirty plate dispenser.",
                    "Pick up a meat from the meat dispenser.",
                ]

                # add all the base tasks that the agent can do
                cross_reference = [
                    (mlam.pickup_onion_actions(counter_objects, only_use_dispensers=True), 1),
                    (mlam.pickup_dirty_plate_actions(counter_objects, only_use_dispensers=True), 2),
                    (mlam.pickup_meat_actions(counter_objects, only_use_dispensers=True), 3),
                ]

            # check all non empty counters
            for i, counter in enumerate(non_empty_counters):
                d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [counter["position"]], agent_index, agent_states)
                if d == "infinite":
                    continue

                # this is a special counter, so this must be added to the kitchen items
                if explicit_language:
                    task_list.append(f"Pick up {clean_name(counter['name'])} from {self.grid_language_reference[counter['position']]}.")
                else:
                    task_list.append(f"Pick up {clean_name(counter['name'])} from general counter.")

                cross_reference.append((
                    mlam.pickup_item_specific_counter(counter['position']),
                    4 + 100 * i
                ))

            # add the shared counters
            for i, counter in enumerate(non_empty_shared_counters):
                d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [counter["position"]], agent_index, agent_states)
                if d == "infinite":
                    continue

                # this is a special counter, so this must be added to the kitchen items
                if explicit_language:
                    task_list.append(f"Pick up {clean_name(counter['name'])} from {self.grid_language_reference[counter['position']]}.")
                else:
                    sc = shared_counter_locations.index(counter["position"]) + 1
                    task_list.append(f"Pick up {clean_name(counter['name'])} from shared counter {sc}.")

                cross_reference.append((
                    mlam.pickup_item_specific_counter(counter['position']),
                    5 + 100 * i
                ))

            # integrate these changes
            if world_state is not None: 
                for sink in sink_states_dict["ready"]:
                    # they can pick up the clean dish if wanted

                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [sink], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (
                                mlam._get_ml_actions_for_positions(
                                    [sink],
                                ),
                                6
                            )
                        )
                        task_list.append(f"Pick up the clean plate from {self.grid_language_reference[sink]}.")
                    else:
                        cross_reference.append(
                            (
                                mlam.pickup_clean_plate_from_sink_actions(
                                    counter_objects, state
                                ),
                                6
                            )
                        )
                        task_list.append("Pick up the clean plate from the sink.")

                for sink in sink_states_dict["full"]:
                    # do a rinse of the clean dish

                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [sink], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([sink]), 7)
                        )
                        task_list.append(f"Do one rinse of the dirty dish in {self.grid_language_reference[sink]}.")
                    else:
                        cross_reference.append(
                            (mlam.rinse_plate_in_sink_actions(state, force=sink), 7)
                        )
                        task_list.append("Do one rinse of the dirty dish in the sink.")

                for board in cutting_board_states_dict['full']:
                    # do a chop of the onion

                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [board], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([board]), 8)
                        )
                        task_list.append(f"Do one chop of the onion on {self.grid_language_reference[board]}.")
                    else:
                        cross_reference.append(
                            (mlam.chop_onion_on_board_actions(state), 8)
                        )
                        task_list.append("Do one chop of the onion on the chopping board.")

            # check the empty counter tasks and add it to the task list
        else:
            # construct task list based on held object, and add to cross reference with switch case
            if len(empty_counters) > 0:
                if explicit_language:
                    d, _, point = self.find_shortest_distance(mlam, current_agent_state["position"], empty_counters, i, agent_states)

                    # place onto an empty counter if possible
                    if point != None and d != "infinite":
                        cross_reference = [
                            (mlam._get_ml_actions_for_positions([point]), 9)
                        ]

                        task_list = [
                            f"Place the {clean_name(held_object['name'])} in hand on the nearest general counter ({self.grid_language_reference[ point ]})."
                        ]
                else:
                    cross_reference = [
                        (mlam.place_obj_on_counter_actions(state), 9)
                    ]

                    task_list = [
                        f"Place the {clean_name(held_object['name'])} in hand on the nearest general counter."
                    ]

            # add all shared counters place actions
            for counter in empty_shared_counters:
                d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [counter], agent_index, agent_states)
                if d == "infinite":
                    continue

                if explicit_language:
                    task_list.append(
                        f"Place the {clean_name(held_object['name'])} in hand on {self.grid_language_reference[counter]}."
                    )
                else:
                    sc = shared_counter_locations.index(counter) + 1

                    task_list.append(
                        f"Place the {clean_name(held_object['name'])} in hand on shared counter {sc}."
                    )
                cross_reference.append(
                    (mlam.place_obj_on_specific_counter(counter), 10)
                )

            if held_object["name"] == "onion":
                # remove subtask that cant do with an onion held

                for cutting_board in cutting_board_states_dict['empty']:
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [cutting_board], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([cutting_board]), 11)
                        )

                        task_list.append(f"Put raw onion in hand on {self.grid_language_reference[cutting_board]}.")
                    else:
                        cross_reference.append(
                            (mlam.put_onion_on_board_actions(state), 11)
                        )

                        task_list.append("Put raw onion in hand on the chopping board.")

            elif held_object["name"] == "meat":
                # check all grills for an empty grill to place the meat

                for grill in grill_states['empty']:
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [grill], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([grill]), 12)
                        )

                        task_list.append(f"Put the raw meat in hand on {self.grid_language_reference[grill]} to cook.")
                    else:
                        cross_reference.append(
                            (mlam.put_meat_in_grill_actions(grill_states_dict), 12)
                        )

                        task_list.append("Put the raw meat in hand on the grill to cook.")

            elif held_object["name"] == "chicken":
                # check all pots for an empty pot to place the chicken

                for pot in pot_states_dict['empty']:
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [pot], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([pot]), 20)
                        )

                        task_list.append(f"Put the raw chicken in hand in {self.grid_language_reference[pot]} to cook.")
                    else:
                        cross_reference.append(
                            (mlam.put_chicken_in_pot_actions(pot_states_dict), 20)
                        )

                        task_list.append("Put the raw chicken in hand in the pot to cook.")

            elif held_object["name"] == "dirty_plate":
                #if there is no dirty plate
                if world_state is not None:
                    # for each empty sink
                    for sink in sink_states_dict["empty"]:
                        d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [sink], agent_index, agent_states)
                        if d == "infinite":
                            continue
                        
                        if explicit_language:
                            cross_reference.append(
                                (mlam._get_ml_actions_for_positions([sink]), 13)
                            )

                            task_list.append(f"Place dirty plate in hand in {self.grid_language_reference[sink]}.")
                        else:
                            cross_reference.append(
                                (mlam.put_dirty_plate_in_sink_actions(counter_objects, state), 13)
                            )

                            task_list.append("Place dirty plate in hand in the sink.")

            elif held_object["name"] == "clean_plate":
                # check all grills for a ready steak
                if world_state is not None:
                    for grill in grill_states_dict['ready']:
                        d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [grill], agent_index, agent_states)
                        if d == "infinite":
                            continue

                        if explicit_language:
                            cross_reference.append(
                                (mlam._get_ml_actions_for_positions([grill]), 14)
                            )

                            task_list.append(f"Use clean plate in hand to pick up steak from {self.grid_language_reference[grill]}.")
                        else:
                            cross_reference.append(
                                (mlam.pickup_steak_with_clean_plate_actions(grill_states_dict), 14)
                            )

                            task_list.append("Use clean plate in hand to pick up steak from the grill.")

                    for pot in pot_states_dict['ready']:
                        d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [pot], agent_index, agent_states)
                        if d == "infinite":
                            continue

                        if explicit_language:
                            cross_reference.append(
                                (mlam._get_ml_actions_for_positions([pot]), 24)
                            )

                            task_list.append(f"Use clean plate in hand to pick up boiled chicken from {self.grid_language_reference[pot]}.")
                        else:
                            cross_reference.append(
                                (mlam.pickup_chicken_with_clean_plate_actions(pot_states_dict), 24)
                            )

                            task_list.append("Use clean plate in hand to pick up boiled chicken from the pot.")

            elif held_object["name"] == "steak":
                # check if there is a garnish available
                for chopping_board in cutting_board_states_dict['ready']:
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [chopping_board], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([chopping_board]), 15)
                        )
                        
                        task_list.append(f"Add garnish from {self.grid_language_reference[chopping_board]} to the steak dish in hand.")
                    else:
                        cross_reference.append(
                            (mlam.add_garnish_to_steak_actions(state), 15)
                        )

                        task_list.append("Add garnish from chopping board to the steak dish in hand.")

                # also allow for the steak to be delivered
                for delivery in mlam.mdp.get_serving_locations():
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [delivery], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([delivery]), 16)
                        )

                        task_list.append(f"Deliver the steak dish in hand to {self.grid_language_reference[delivery]}.")
                    else:
                        cross_reference.append(
                            (mlam.deliver_soup_actions(), 16)
                        )

                        task_list.append("Deliver the steak dish in hand to delivery location.")

            elif held_object["name"] == "steak_onion":
                # remove subtask that cant do with a dish held
                for delivery in mlam.mdp.get_serving_locations():
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [delivery], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([delivery]), 17)
                        )

                        task_list.append(f"Deliver the steak onion dish in hand to {self.grid_language_reference[delivery]}.")
                    else:
                        cross_reference.append(
                            (mlam.deliver_soup_actions(), 17)
                        )

                        task_list.append("Deliver the steak onion dish in hand to delivery location.")
            
            elif held_object["name"] == "boiled_chicken":
                # check if there is a garnish available
                for chopping_board in cutting_board_states_dict['ready']:
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [chopping_board], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([chopping_board]), 21)
                        )
                        
                        task_list.append(f"Add garnish from {self.grid_language_reference[chopping_board]} to the boiled chicken dish in hand.")
                    else:
                        cross_reference.append(
                            (mlam.add_garnish_to_chicken_actions(state), 21)
                        )

                        task_list.append("Add garnish from chopping board to the boiled chicken dish in hand.")

                # also allow for the chicken to be delivered
                for delivery in mlam.mdp.get_serving_locations():
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [delivery], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([delivery]), 22)
                        )

                        task_list.append(f"Deliver the boiled chicken dish in hand to {self.grid_language_reference[delivery]}.")
                    else:
                        cross_reference.append(
                            (mlam.deliver_soup_actions(), 22)
                        )

                        task_list.append("Deliver the boiled chicken dish in hand to delivery location.")

            elif held_object["name"] == "boiled_chicken_onion":
                # remove subtask that cant do with a dish held
                for delivery in mlam.mdp.get_serving_locations():
                    d, _, _ = self.find_shortest_distance(mlam, current_agent_state["position"], [delivery], agent_index, agent_states)
                    if d == "infinite":
                        continue

                    if explicit_language:
                        cross_reference.append(
                            (mlam._get_ml_actions_for_positions([delivery]), 23)
                        )

                        task_list.append(f"Deliver the boiled chicken onion dish in hand to {self.grid_language_reference[delivery]}.")
                    else:
                        cross_reference.append(
                            (mlam.deliver_soup_actions(), 23)
                        )

                        task_list.append("Deliver the boiled chicken onion dish in hand to delivery location.")

        # add the do nothing action at the end (applies to every action subset)
        task_list.append("Wait for 5 timesteps.")
        cross_reference.append(
            (Action.STAY, 18)
        )
        # TODO: add move away action

        # format and send them back
        location_info = "\n\n".join(location_info_array)
        kitchen_items = "\n".join(kitchen_items_array)
        task_list_as_language = self.format_task_list(task_list)
        # cross reference is already in the correct format: List of MLA's

        return location_info, kitchen_items, task_list_as_language, cross_reference, task_list

    def build(
        self,
        state,
        mlam,
        agent_index: int,
        prompt: Dict[str, str],
        prompt_overrides: Dict[str, str] = {},
        ml_action_log: list = [],
        memory_depth: int = 0,
        communication_enabled: bool = True,
        explicit_language: bool = True,
        **kwargs
    ):
        """NOTE:
            If both communication and radio are enabled, then the agent can only either take an action or communicate with the other agent. They cannot do both in the same turn.
        """
        # obtain relevant state information
        current_agent_state = state.players[agent_index].to_dict()
        other_agents_states = [agent.to_dict() for i, agent in enumerate(state.players) if i != agent_index]
        grill_states = mlam.mdp.get_grill_states(state)
        cutting_board_states = mlam.mdp.get_chopping_board_states(state)
        world_state = state.to_dict().pop("objects")

        references = dict(prompt_overrides)

        # format the context
        chefs_description = self.format_other_agents_description(agent_index, kwargs['agent_types'])

        references['context'] = replace_all(
            prompt['context'],
            {
                "{agent_name}": self.names[agent_index],
                "{other_agents_description}": chefs_description,
                "{environment_description}": self.grid_config['description_advanced'],
            }
        )

        inventory_as_language = self.format_inventory_as_language(state, agent_index)

        location_info, kitchen_items, task_list_as_language, cross_reference, task_list = self.get_kitchen_as_language(state, agent_index, mlam, explicit_language=explicit_language)
        message_history_as_language = self.format_chat_history_as_language(agent_index, kwargs['chat_history'], state.timestep)

        # action history
        action_history_as_language = self.format_action_history_as_language(current_agent_state['action_history'], state.timestep, memory_depth=memory_depth)

        # get the current order and next order
        current_order = "nothing"
        next_order = "nothing"
        order_list = state.order_list

        order_list_as_language = ", ".join([f"Order {str(i+1)}: {clean_name(order)}" for i, order in enumerate(order_list)])

        if len(order_list) > 0:
            current_order = clean_name(order_list[0])
            if len(order_list) > 1:
                next_order = clean_name(order_list[1])

        # format the goal
        references['goal'] = replace_all(
            prompt['goal'],
            {
                "{current_order}": current_order,
                "{next_order}": next_order,
                "{order_list}": order_list_as_language
            }
        )

        communication_addition = ""
        if communication_enabled:
            communication_addition = """\n\n<Message History (Last is most recent)>:
{message_history}"""

        references['state'] = replace_all(
            prompt['state'] + communication_addition,
            {
                "{inventory}": inventory_as_language,
                "{location_info}": location_info,
                "{kitchen_items}": kitchen_items,
                "{current_order}": current_order,
                "{next_order}": next_order,
                "{order_list}": order_list_as_language,
                "{message_history}": message_history_as_language,
                "{task_list}": task_list_as_language,
                "{action_history}": action_history_as_language,
            }
        )

        steak_cook_time = mlam.mdp.steak_cook_time
        chicken_cook_time = mlam.mdp.steak_cook_time

        references['rules'] = replace_all(
            prompt['rules'],
            {
                "{steak_cook_time}": str(steak_cook_time),
                "{chicken_cook_time}": str(chicken_cook_time)
            }
        )

        # goal
        references['goal'] = prompt['goal']
        
        # if not 'P' in self.grid_config['objects']:
        if communication_enabled:
            references['goal'] += "\n" + "I'll provide your action history, current state, state of your teammates, chat history with other agents, and possible actions. Select the best action from the list. Got it?"
        else:
            references['goal'] += "\n" + "I'll provide your action history, current state, state of your teammates, and possible actions. Select the best action from the list. Got it?"

        # timer
        references['timer'] = self.format_timer_as_language(state.timestep, kwargs['horizon'])


        # actions

        references['actions'] = f"""
<Available Actions>:
{task_list_as_language}"""


        # format
        references['format'] = "The other agent does not know what you are doing, they only have information about your current location.\n\n" + prompt['format']
        

        if communication_enabled:
            # with communication
            references['format'] = f"""{references['format']}

Using your conversation history with {NAMES[1 - agent_index]}, Select the best action from the list to deliver the current order (Order 1) as quickly as possible. Do not deliver an order other than Order 1. When you choose an action, format it as follows:
    
    I have chosen Option [YOUR OPTION NUMBER HERE] (WHAT YOUR OPTION IS)

After selecting an action, you have the option to include a clear and informative message to the other agents in brackets []. Do not break the game rules in your message. If no message is necessary, do not provide one. Below is how you should format your message:
    
    My message: [INSERT YOUR MESSAGE HERE]
"""
        else:
            # without communication
            references['format'] = f"""{references['format']}

Select the best action from the list to deliver the current order (Order 1) as quickly as possible. Do not deliver an order other than Order 1. When you choose an action, format it as follows:
    
    I have chosen Option [YOUR OPTION NUMBER HERE] (WHAT YOUR OPTION IS)"""

        return cross_reference, references, task_list