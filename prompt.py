from abc import abstractmethod

from pyftg.models.frame_data import FrameData, CharacterData, AttackData
import json
import pandas as pd
import os
import yaml
from loguru import logger




class BasePrompt:
    def __init__(self):
        pass

    @abstractmethod
    def generate_prompt(
        self, frame_data: FrameData, player: bool, reward: float
    ) -> str:
        pass

# 在 prompt.py 中添加这个类 #修改这里
class CustomPromptGenerator(BasePrompt):
    def __init__(self):
        super().__init__()
        
    def frame_data_to_json(
        self, frame_data: FrameData, player: bool, reward: float = 0
    ) -> dict:
        json_data = {}
        self_player = frame_data.get_character(player)
        self_player_json = self.character_data_to_json(self_player)
        opp_player = frame_data.get_character(not player)
        opp_player_json = self.character_data_to_json(opp_player)
        self_projectiles = [
            self.attack_data_to_json(a)
            for a in frame_data.get_projectiles_by_player(player)
        ]
        opp_projectiles = [
            self.attack_data_to_json(a)
            for a in frame_data.get_projectiles_by_player(not player)
        ]
        self_player_json["projectiles"] = self_projectiles
        opp_player_json["projectiles"] = opp_projectiles
        json_data = {
            "opponent": opp_player_json,
            "self": self_player_json,
            "frame num": frame_data.current_frame_number,
        }
        return json_data

    def character_data_to_json(self, character_data: CharacterData) -> dict:
        data_json = {
            "remaining frames": character_data.remaining_frame,
            "hp": character_data.hp,
            "action": character_data.action.name,
            "state": character_data.state.name,
            "position": {
                "left": character_data.left,
                "right": character_data.right,
                "top": character_data.top,
                "bottom": character_data.bottom,
            },
            "speed": {"x": character_data.speed_x, "y": character_data.speed_y},
            "energy": character_data.energy,
        }
        return data_json

    def attack_data_to_json(self, attack_data: AttackData) -> dict:
        attack_data_json = {
            "damage": attack_data.hit_damage,
            "left": attack_data.current_hit_area.left,
            "right": attack_data.current_hit_area.right,
            "bottom": attack_data.current_hit_area.bottom,
            "top": attack_data.current_hit_area.top,
            "speed": {"x": attack_data.speed_x, "y": attack_data.speed_y},
        }
        return attack_data_json

    def generate_prompt(
        self, frame_data: FrameData, player: bool, reward: float
    ) -> str:
        game_state = self.frame_data_to_json(frame_data, player, reward)
#         prompt = f"""You are the best in a 2D fighting game.


# {json.dumps(game_state, indent=4)}
# ----
# Reply with only one move. Follow RUSHDOWN fighting style. And explain why you choose it."""
#         return prompt
#         prompt = f"""You are the best in a 2D fighting game.


# {json.dumps(game_state, indent=4)}
# ----
# Reply with only one move. And explain why you choose it."""
#         return prompt

# change prompt here
        prompt = f"""You are the best in a 2D fighting game.

{json.dumps(game_state, indent=4)}
----
Reply with only one move. Follow ZONING fighting style. And explain why you choose it."""
        return prompt

class PromptGenerator(BasePrompt):
    def __init__(self, n_shots=0):
        super().__init__()
        self.n_shots = n_shots
        if self.n_shots > 0:
            prompt_file = "prompt1.txt"
        else:
            prompt_file = "prompt2.txt"
        with open(prompt_file, "r") as f:
            self.prompt = "".join([a for a in f.readlines()])
        self.examples = self.read_examples("examples.txt", self.n_shots)
        self.llm_actions = pd.read_csv("merged_actions.csv")
        self.llm_actions = self.llm_actions.to_dict(orient="records")

    def frame_data_to_json(
        self, frame_data: FrameData, player: bool, reward: float = 0
    ) -> dict:
        json_data = {}
        self_player = frame_data.get_character(player)
        self_player_json = self.character_data_to_json(self_player)
        opp_player = frame_data.get_character(not player)
        opp_player_json = self.character_data_to_json(opp_player)
        self_projectiles = [
            self.attack_data_to_json(a)
            for a in frame_data.get_projectiles_by_player(player)
        ]
        opp_projectiles = [
            self.attack_data_to_json(a)
            for a in frame_data.get_projectiles_by_player(not player)
        ]
        self_player_json["projectiles"] = self_projectiles
        opp_player_json["projectiles"] = opp_projectiles
        json_data = {
            "self": self_player_json,
            "opponent": opp_player_json,
            "frame num": frame_data.current_frame_number,
        }
        return json_data

    def character_data_to_json(self, character_data: CharacterData) -> dict:
        data_json = {
            "hp": character_data.hp,
            "energy": character_data.energy,
            "position": {
                "left": character_data.left,
                "right": character_data.right,
                "top": character_data.top,
                "bottom": character_data.bottom,
            },
            "speed": {"x": character_data.speed_x, "y": character_data.speed_y},
            # "direction": character_data,
            "action": character_data.action.name,
            "state": character_data.state.name,
            # "projectiles": [
            #     attack_data_to_json(a) for a in character_data.projectile_attack
            # ],
            "remaining frames": character_data.remaining_frame,
        }
        return data_json

    def attack_data_to_json(self, attack_data: AttackData) -> dict:
        attack_data_json = {
            "damage": attack_data.hit_damage,
            "left": attack_data.current_hit_area.left,
            "right": attack_data.current_hit_area.right,
            "bottom": attack_data.current_hit_area.bottom,
            "top": attack_data.current_hit_area.top,
            "speed": {"x": attack_data.speed_x, "y": attack_data.speed_y},
        }
        return attack_data_json

    def read_examples(self, file_path, shots):
        example_texts = []
        text_template = "State:\n {state}\nAction: {action}"
        with open(file_path, "r") as f:
            for line in f.readlines():
                t = eval(line)
                action = t["action"]
                temp = {}
                for key in ["self", "opponent", "frame num"]:
                    temp[key] = t[key]
                example_texts.append(
                    text_template.format(
                        state=json.dumps(temp, indent=4), action=action
                    )
                )

        return "\n".join(example_texts[:shots])

    def generate_prompt(
        self, frame_data: FrameData, player: bool, reward: float = 0
    ) -> str:
        json_data = self.frame_data_to_json(frame_data, player, reward)
        if self.n_shots > 0:
            prompt = self.prompt.format(
                game_state=json.dumps(json_data, indent=4),
                actions=json.dumps(self.llm_actions, indent=4),
                examples=self.examples,
            )
        else:
            prompt = self.prompt.format(
                game_state=json.dumps(json_data, indent=4),
                actions=json.dumps(self.llm_actions, indent=4),
            )
        return prompt


class OpenGenerativeAIGenerator(BasePrompt):
    def __init__(self):
        super().__init__()
        with open("prompt3.txt", "r") as f:
            self.prompt_template = "".join([a for a in f.readlines()])
        self.llm_actions = pd.read_csv("merged_actions.csv")
        self.llm_actions = self.llm_actions.to_dict(orient="records")

    def context_prompt(self, frame_data: FrameData, player: bool, reward: float) -> str:
        my = frame_data.get_character(player)
        opp = frame_data.get_character(not player)
        my_hp = my.hp
        my_energy = my.energy
        my_x = (my.left + my.right) / 2 / 960
        my_y = (my.bottom + my.top) / 2 / 640
        my_action = my.action.name

        opp_hp = opp.hp
        opp_x = (opp.left + opp.right) / 2 / 960
        opp_y = (opp.bottom + opp.top) / 2 / 640
        opp_action = opp.action.name

        normalized_relative_position = abs(my_x - opp_x)
        position_prompt = ""
        if normalized_relative_position > 0.1:
            position_prompt += (
                "You are very far from the opponent. Move closer to the opponent."
            )
        else:
            position_prompt += "You are close to the opponent. You should attack him."
        power_prompt = ""
        if my_energy >= 5:
            power_prompt = "You can now use a light attack move. The names of the light attack moves are: THROW_A, STAND_D_DF_FA, AIR_D_DF_FA"
        if my_energy >= 10:
            power_prompt = "You can now use a light attack move. The names of the hard attack moves are: THROW_B, AIR_F_D_DFA, AIR_D_DB_BA"
        if my_energy >= 20:
            power_prompt = "You can now use a heavy attack move. The names of the hard attack moves are: AIR_D_DF_FB, STAND_D_DF_FB"
        if my_energy >= 40:
            power_prompt = "You can now use a light kick move. The names of the hard attack moves are: AIR_F_D_DFB"
        if my_energy >= 50:
            power_prompt = "You can now use a light kick move. The names of the hard attack moves are: AIR_D_DB_BB, STAND_D_DB_BB"
        if my_energy >= 55:
            power_prompt = "You can now use a hard uppercut move. The names of the hard attack moves are: STAND_F_D_DFB"
        if my_energy >= 150:
            power_prompt = "You can now throw a special fireball. The names of the hard attack moves are: STAND_D_DF_FC"

        last_action_prompt = f"Your last action was {my_action}. The opponent's last action was {opp_action}."
        score_prompt = ""
        if reward > 0:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif reward < 0:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )
        opponent_projectiles = frame_data.get_projectiles_by_player(player=not player)
        if len(opponent_projectiles) == 0 or opponent_projectiles is None:
            projectiles_prompt = "There is no projectile attack by the opponent."
        else:
            projectiles_prompt = f"There are {len(opponent_projectiles)} projectile attacks, you should jump to avoid them."
        # Assemble everything
        context = f"""{position_prompt}
            {projectiles_prompt}
            {power_prompt}
            {last_action_prompt}
            Your current score is {reward}. {score_prompt}
            To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
            """

        return context

    def generate_prompt(
        self, frame_data: FrameData, player: bool, reward: float
    ) -> str:
        context_prompt = self.context_prompt(frame_data, player, reward)
        move_list = "- " + "\n - ".join(
            [f"{u['Action']}: {u['Description']}" for u in self.llm_actions]
        )
        return self.prompt_template.format(
            context_prompt=context_prompt, move_list=move_list
        )


class OpenGenerativeAIGenerator1(OpenGenerativeAIGenerator):
    def __init__(self):
        super().__init__()
        with open("prompt4.txt", "r") as f:
            self.prompt_template = "".join([a for a in f.readlines()])
        self.llm_actions = pd.read_csv("merged_actions.csv")
        self.llm_actions = self.llm_actions.to_dict(orient="records")


class OpenGenerativeAIGenerator2(OpenGenerativeAIGenerator1):
    def generate_prompt(
        self, frame_data: FrameData, player: bool, reward: float
    ) -> str:
        context_prompt = self.context_prompt(frame_data, player, reward)
        move_list = json.dumps(self.llm_actions, indent=4)
        return self.prompt_template.format(
            context_prompt=context_prompt, move_list=move_list
        )


class OpenGenerativeAIGenerator3(OpenGenerativeAIGenerator2):
    def context_prompt(self, frame_data: FrameData, player: bool, reward: float) -> str:
        my = frame_data.get_character(player)
        opp = frame_data.get_character(not player)
        my_hp = my.hp
        my_energy = my.energy
        my_x = (my.left + my.right) / 2 / 960
        my_y = (my.bottom + my.top) / 2 / 640
        my_action = my.action.name

        opp_hp = opp.hp
        opp_x = (opp.left + opp.right) / 2 / 960
        opp_y = (opp.bottom + opp.top) / 2 / 640
        opp_action = opp.action.name

        normalized_relative_position = abs(my_x - opp_x)
        position_prompt = ""
        if normalized_relative_position > 0.1:
            position_prompt += (
                "You are very far from the opponent. Move closer to the opponent."
            )
        else:
            position_prompt += "You are close to the opponent. You should attack him."
        power_prompt = ""
        if my_energy >= 5:
            power_prompt += "You can now use a light attack move. The names of the light attack moves are: THROW_A, STAND_D_DF_FA, AIR_D_DF_FA\n"
        if my_energy >= 10:
            power_prompt += "You can now use a light attack move. The names of the hard attack moves are: THROW_B, AIR_F_D_DFA, AIR_D_DB_BA\n"
        if my_energy >= 20:
            power_prompt += "You can now use a heavy attack move. The names of the hard attack moves are: AIR_D_DF_FB, STAND_D_DF_FB\n"
        if my_energy >= 40:
            power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_F_D_DFB\n"
        if my_energy >= 50:
            power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_D_DB_BB, STAND_D_DB_BB\n"
        if my_energy >= 55:
            power_prompt += "You can now use a hard uppercut move. The names of the hard attack moves are: STAND_F_D_DFB\n"
        if my_energy >= 150:
            power_prompt += "You can now throw a special fireball. The names of the hard attack moves are: STAND_D_DF_FC\n"

        last_action_prompt = f"Your last action was {my_action}. The opponent's last action was {opp_action}."
        score_prompt = ""
        if reward > 0:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif reward < 0:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )
        # Assemble everything
        context = f"""{position_prompt}
            {power_prompt}
            {last_action_prompt}
            Your current score is {reward}. {score_prompt}
            To increase your score, move toward the opponent and attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
            """

        return context


class OpenGenerativeAIGeneratorPersona(OpenGenerativeAIGenerator3):
    def __init__(self, persona: str):
        super().__init__()
        with open("prompt5.txt", "r") as f:
            self.prompt_template = "".join([a for a in f.readlines()])
        self.llm_actions = pd.read_csv("merged_actions.csv")
        self.llm_actions = self.llm_actions.to_dict(orient="records")
        self.persona = persona
        if self.persona is not None and self.persona != "":
            persona_file = os.path.join("personas", f"{self.persona}.txt")
            if not os.path.exists(persona_file):
                raise Exception("Persona invalid")
            else:
                logger.info(f"Playing in {self.persona} persona")
            with open(persona_file, "r") as f:
                self.persona_actions = "".join(f.readlines())
            self.persona_description = f"You are playing in {self.persona} style, below are the list of actions that a {self.persona} style should use:\n{self.persona_actions}"
        else:
            self.persona_description = ""

    def generate_prompt(
        self, frame_data: FrameData, player: bool, reward: float
    ) -> str:
        context_prompt = self.context_prompt(frame_data, player, reward)
        move_list = json.dumps(self.llm_actions, indent=4)
        return self.prompt_template.format(
            context_prompt=context_prompt,
            move_list=move_list,
            persona=self.persona_description,
        )

    def context_prompt(self, frame_data: FrameData, player: bool, reward: float) -> str:
        my = frame_data.get_character(player)
        opp = frame_data.get_character(not player)
        my_hp = my.hp
        my_energy = my.energy
        my_x = (my.left + my.right) / 2
        my_y = (my.bottom + my.top) / 2
        my_action = my.action.name

        opp_hp = opp.hp
        opp_x = (opp.left + opp.right) / 2
        opp_y = (opp.bottom + opp.top) / 2
        opp_action = opp.action.name

        normalized_relative_position = abs(my_x - opp_x)
        position_prompt = f"The distance of you and the opponent is {normalized_relative_position}/960"
        power_prompt = ""
        if my_energy >= 5:
            power_prompt += "You can now use a light attack move. The names of the light attack moves are: THROW_A, STAND_D_DF_FA, AIR_D_DF_FA\n"
        if my_energy >= 10:
            power_prompt += "You can now use a light attack move. The names of the hard attack moves are: THROW_B, AIR_F_D_DFA, AIR_D_DB_BA\n"
        if my_energy >= 20:
            power_prompt += "You can now use a heavy attack move. The names of the hard attack moves are: AIR_D_DF_FB, STAND_D_DF_FB\n"
        if my_energy >= 40:
            power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_F_D_DFB\n"
        if my_energy >= 50:
            power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_D_DB_BB, STAND_D_DB_BB\n"
        if my_energy >= 55:
            power_prompt += "You can now use a hard uppercut move. The names of the hard attack moves are: STAND_F_D_DFB\n"
        if my_energy >= 150:
            power_prompt += "You can now throw a special fireball. The names of the hard attack moves are: STAND_D_DF_FC\n"

        last_action_prompt = f"Your last action was {my_action}. The opponent's last action was {opp_action}."
        score_prompt = ""
        if reward > 0:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif reward < 0:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )
        # Assemble everything
        context = f"""
            {position_prompt}
            {power_prompt}
            {last_action_prompt}
            Your current score is {reward}. {score_prompt}
            To increase your score, try to attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
            """

        return context


class OpenGenerativeAIGeneratorPersona2(OpenGenerativeAIGeneratorPersona):
    def frame_data_to_json(
        self, frame_data: FrameData, player: bool, reward: float = 0
    ) -> dict:
        json_data = {}
        self_player = frame_data.get_character(player)
        self_player_json = self.character_data_to_json(self_player)
        opp_player = frame_data.get_character(not player)
        opp_player_json = self.character_data_to_json(opp_player)
        self_projectiles = [
            self.attack_data_to_json(a)
            for a in frame_data.get_projectiles_by_player(player)
        ]
        opp_projectiles = [
            self.attack_data_to_json(a)
            for a in frame_data.get_projectiles_by_player(not player)
        ]
        self_player_json["projectiles"] = self_projectiles
        opp_player_json["projectiles"] = opp_projectiles
        json_data = {
            "self": self_player_json,
            "opponent": opp_player_json,
            # "frame num": frame_data.current_frame_number,
        }
        return json_data

    def character_data_to_json(self, character_data: CharacterData) -> dict:
        data_json = {
            "hp": character_data.hp,
            "energy": character_data.energy,
            "position": {
                "left": character_data.left,
                "right": character_data.right,
                "top": character_data.top,
                "bottom": character_data.bottom,
            },
            "speed": {"x": character_data.speed_x, "y": character_data.speed_y},
            # "direction": character_data,
            "action": character_data.action.name,
            "state": character_data.state.name,
            # "projectiles": [
            #     attack_data_to_json(a) for a in character_data.projectile_attack
            # ],
            # "remaining frames": character_data.remaining_frame,
        }
        return data_json

    def attack_data_to_json(self, attack_data: AttackData) -> dict:
        attack_data_json = {
            "damage": attack_data.hit_damage,
            "left": attack_data.current_hit_area.left,
            "right": attack_data.current_hit_area.right,
            "bottom": attack_data.current_hit_area.bottom,
            "top": attack_data.current_hit_area.top,
            "speed": {"x": attack_data.speed_x, "y": attack_data.speed_y},
        }
        return attack_data_json

    def context_prompt(self, frame_data: FrameData, player: bool, reward: float) -> str:
        current_status = self.frame_data_to_json(frame_data, player, reward)
        my = frame_data.get_character(player)
        opp = frame_data.get_character(not player)
        my_hp = my.hp
        my_energy = my.energy
        my_x = (my.left + my.right) / 2
        my_y = (my.bottom + my.top) / 2
        my_action = my.action.name

        opp_hp = opp.hp
        opp_x = (opp.left + opp.right) / 2
        opp_y = (opp.bottom + opp.top) / 2
        opp_action = opp.action.name

        normalized_relative_position = abs(my_x - opp_x)
        position_prompt = f"The distance of you and the opponent is {normalized_relative_position}/960."
        if normalized_relative_position <= 10:
            position_prompt += "You are already close the opponent, there is no need to move forward, you can use shot-range attacks."
        elif normalized_relative_position >= 200:
            position_prompt += "You are far away from the opponent, you can move closer to the opponent or use long-range attacks. However, make sure that your energy is enough to use those attacks."
        power_prompt = ""
        if my_energy >= 5:
            power_prompt += "You can now use a light attack move. The names of the light attack moves are: THROW_A, STAND_D_DF_FA, AIR_D_DF_FA\n"
        if my_energy >= 10:
            power_prompt += "You can now use a light attack move. The names of the hard attack moves are: THROW_B, AIR_F_D_DFA, AIR_D_DB_BA\n"
        if my_energy >= 20:
            power_prompt += "You can now use a heavy attack move. The names of the hard attack moves are: AIR_D_DF_FB, STAND_D_DF_FB\n"
        if my_energy >= 40:
            power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_F_D_DFB\n"
        if my_energy >= 50:
            power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_D_DB_BB, STAND_D_DB_BB\n"
        if my_energy >= 55:
            power_prompt += "You can now use a hard uppercut move. The names of the hard attack moves are: STAND_F_D_DFB\n"
        if my_energy >= 150:
            power_prompt += "You can now throw a special fireball. The names of the hard attack moves are: STAND_D_DF_FC\n"

        last_action_prompt = f"Your last action was {my_action}. The opponent's last action was {opp_action}."
        score_prompt = ""
        if my_hp > opp_hp:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif my_hp < opp_hp:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )
        # Assemble everything
        context = f"""
The current game state is fomatted as below:
Player info:
- self: the player you are controlling
- opponent: the opponent
You and opponent info contain the following fields:
- hp: player's current hp
- energy: player's current energy
- position: player's current position in left, right, top, bottom
- speed: player's current speed in x and y axis
- action: player's current action
- state: player's current state
- projectiles: player's projectiles (if any), it has damage, left, right, top, bottom and speed (x, y)
---
{yaml.dump(current_status)}
---
{position_prompt}
{power_prompt}
{last_action_prompt}

Your current score is {reward}. {score_prompt}
To increase your score, try to attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
            """

        return context

    def generate_prompt(
        self, frame_data: FrameData, player: bool, reward: float
    ) -> str:
        context_prompt = self.context_prompt(frame_data, player, reward)
        move_list = yaml.dump(self.llm_actions)
        return self.prompt_template.format(
            context_prompt=context_prompt,
            move_list=move_list,
            persona=self.persona_description,
        )


class OpenGenerativeAIGeneratorPersona3(OpenGenerativeAIGeneratorPersona2):
    def context_prompt(self, frame_data: FrameData, player: bool, reward: float) -> str:
        current_status = self.frame_data_to_json(frame_data, player, reward)
        my = frame_data.get_character(player)
        opp = frame_data.get_character(not player)
        my_hp = my.hp
        my_energy = my.energy
        my_x = (my.left + my.right) / 2
        my_y = (my.bottom + my.top) / 2
        my_action = my.action.name

        opp_hp = opp.hp
        opp_x = (opp.left + opp.right) / 2
        opp_y = (opp.bottom + opp.top) / 2
        opp_action = opp.action.name

        normalized_relative_position = abs(my_x - opp_x)
        position_prompt = f"The distance of you and the opponent is {normalized_relative_position}/960."
        if normalized_relative_position <= 10:
            position_prompt += "You are already close the opponent, there is no need to move forward, you can use shot-range attacks."
        elif normalized_relative_position >= 200:
            position_prompt += "You are far away from the opponent, you can move closer to the opponent or use long-range attacks. However, make sure that your energy is enough to use those attacks."
        power_prompt = ""
        # if my_energy >= 5:
        #     power_prompt += "You can now use a light attack move. The names of the light attack moves are: THROW_A, STAND_D_DF_FA, AIR_D_DF_FA\n"
        # if my_energy >= 10:
        #     power_prompt += "You can now use a light attack move. The names of the hard attack moves are: THROW_B, AIR_F_D_DFA, AIR_D_DB_BA\n"
        # if my_energy >= 20:
        #     power_prompt += "You can now use a heavy attack move. The names of the hard attack moves are: AIR_D_DF_FB, STAND_D_DF_FB\n"
        # if my_energy >= 40:
        #     power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_F_D_DFB\n"
        # if my_energy >= 50:
        #     power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_D_DB_BB, STAND_D_DB_BB\n"
        # if my_energy >= 55:
        #     power_prompt += "You can now use a hard uppercut move. The names of the hard attack moves are: STAND_F_D_DFB\n"
        # if my_energy >= 150:
        #     power_prompt += "You can now throw a special fireball. The names of the hard attack moves are: STAND_D_DF_FC\n"

        last_action_prompt = f"Your last action was {my_action}. The opponent's last action was {opp_action}."
        score_prompt = ""
        if my_hp > opp_hp:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif my_hp < opp_hp:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )
        # Assemble everything
        context = f"""
The current game state is fomatted as below:
Player info:
- self: the player you are controlling
- opponent: the opponent
You and opponent info contain the following fields:
- hp: player's current hp
- energy: player's current energy
- position: player's current position in left, right, top, bottom
- speed: player's current speed in x and y axis
- action: player's current action
- state: player's current state
- projectiles: player's projectiles (if any), it has damage, left, right, top, bottom and speed (x, y)
---
{yaml.dump(current_status)}
---
{position_prompt}
{power_prompt}
{last_action_prompt}

Your current score is {reward}. {score_prompt}
To increase your score, try to attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
            """

        return context


class OpenGenerativeAIGeneratorPersona4(OpenGenerativeAIGeneratorPersona3):
    def __init__(self, persona: str):
        # super().__init__()
        with open("prompt6.txt", "r") as f:
            self.prompt_template = "".join([a for a in f.readlines()])
        self.llm_actions = pd.read_csv("merged_actions.csv")
        self.llm_actions = self.llm_actions.to_dict(orient="records")
        self.persona = persona
        if self.persona is not None and self.persona != "":
            persona_file = os.path.join("personas", f"{self.persona}.txt")
            if not os.path.exists(persona_file):
                raise Exception("Persona invalid")
            else:
                logger.info(f"Playing in {self.persona} persona")
            with open(persona_file, "r") as f:
                self.persona_actions = "".join(f.readlines())
            self.persona_description = f"You are playing in {self.persona} style, below are the list of actions that a {self.persona} style should use:\n{self.persona_actions}"
        else:
            self.persona_description = ""


class OpenGenerativeAIGeneratorPersona5(OpenGenerativeAIGeneratorPersona4):
    def __init__(self, persona: str):
        # super().__init__()
        with open("prompt7.txt", "r") as f:
            self.prompt_template = "".join([a for a in f.readlines()])
        self.llm_actions = pd.read_csv("merged_actions.csv")
        self.llm_actions = self.llm_actions.to_dict(orient="records")
        self.persona = persona
        if self.persona is not None and self.persona != "":
            persona_file = os.path.join("personas", f"{self.persona}.txt")
            if not os.path.exists(persona_file):
                raise Exception("Persona invalid")
            else:
                logger.info(f"Playing in {self.persona} persona")
            with open(persona_file, "r") as f:
                self.persona_actions = "".join(f.readlines())
            self.persona_description = f"You are playing in {self.persona} style, below are the list of actions that a {self.persona} style should use:\n{self.persona_actions}"
        else:
            self.persona_description = ""


class OpenGenerativeAIGeneratorPersona6(OpenGenerativeAIGeneratorPersona5):
    def context_prompt(self, frame_data: FrameData, player: bool, reward: float) -> str:
        current_status = self.frame_data_to_json(frame_data, player, reward)
        my = frame_data.get_character(player)
        opp = frame_data.get_character(not player)
        my_hp = my.hp
        my_energy = my.energy
        my_x = (my.left + my.right) / 2
        my_y = (my.bottom + my.top) / 2
        my_action = my.action.name

        opp_hp = opp.hp
        opp_x = (opp.left + opp.right) / 2
        opp_y = (opp.bottom + opp.top) / 2
        opp_action = opp.action.name

        normalized_relative_position = abs(my_x - opp_x)
        position_prompt = f"The distance of you and the opponent is {normalized_relative_position}/960."
        if normalized_relative_position <= 10:
            position_prompt += "You are already close the opponent, there is no need to move forward, you can use shot-range attacks."
        elif normalized_relative_position >= 200:
            position_prompt += "You are far away from the opponent, you can move closer to the opponent or use long-range attacks. However, make sure that your energy is enough to use those attacks."
        power_prompt = ""
        if my_energy >= 5:
            power_prompt += "You can now use a light attack move. The names of the light attack moves are: THROW_A, STAND_D_DF_FA, AIR_D_DF_FA\n"
        if my_energy >= 10:
            power_prompt += "You can now use a light attack move. The names of the hard attack moves are: THROW_B, AIR_F_D_DFA, AIR_D_DB_BA\n"
        if my_energy >= 20:
            power_prompt += "You can now use a heavy attack move. The names of the hard attack moves are: AIR_D_DF_FB, STAND_D_DF_FB\n"
        if my_energy >= 40:
            power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_F_D_DFB\n"
        if my_energy >= 50:
            power_prompt += "You can now use a light kick move. The names of the hard attack moves are: AIR_D_DB_BB, STAND_D_DB_BB\n"
        if my_energy >= 55:
            power_prompt += "You can now use a hard uppercut move. The names of the hard attack moves are: STAND_F_D_DFB\n"
        if my_energy >= 150:
            power_prompt += "You can now throw a special fireball. The names of the hard attack moves are: STAND_D_DF_FC\n"

        last_action_prompt = f"Your last action was {my_action}. The opponent's last action was {opp_action}."
        score_prompt = ""
        if my_hp > opp_hp:
            score_prompt += "You are winning. Keep attacking the opponent."
        elif my_hp < opp_hp:
            score_prompt += (
                "You are losing. Continue to attack the opponent but don't get hit."
            )
        # Assemble everything
        context = f"""
The current game state is fomatted as below:
Player info:
- self: the player you are controlling
- opponent: the opponent
You and opponent info contain the following fields:
- hp: player's current hp
- energy: player's current energy
- position: player's current position in left, right, top, bottom
- speed: player's current speed in x and y axis
- action: player's current action
- state: player's current state
- projectiles: player's projectiles (if any), it has damage, left, right, top, bottom and speed (x, y)
---
{yaml.dump(current_status)}
---
{position_prompt}
{power_prompt}
{last_action_prompt}

Your current score is {reward}. {score_prompt}
To increase your score, try to attack the opponent. To prevent your score from decreasing, don't get hit by the opponent.
            """

        return context


if __name__ == "__main__":
    data = PromptGenerator().read_examples("examples.txt")
    print("\n".join(data))
