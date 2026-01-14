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
# change prompt here
        prompt = f"""You are the best in a 2D fighting game.

{json.dumps(game_state, indent=4)}
----
Reply with only one move. Follow ZONING fighting style. And explain why you choose it."""
        return prompt
