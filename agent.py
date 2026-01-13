import sys
import time

from pyftg.models.screen_data import ScreenData

sys.path.append("./")

import numpy as np
import pandas as pd
from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult

from prompt import CustomPromptGenerator
from llm import LLamaLLM
from loguru import logger

import logging

from model_manager import model_manager

logging.basicConfig(level=logging.INFO)
STATE_DIM = {
    1: {"conv1d": 160, "fft": 512, "mel": 2560},
    4: {"conv1d": 64, "fft": 512, "mel": 1280},
}
DEBUG = False


class LLMAgent(AIInterface):
    def __init__(self, **kwargs):
        self.actions = (
            "AIR_A",
            "AIR_B",
            "AIR_D_DB_BA",
            "AIR_D_DB_BB",
            "AIR_D_DF_FA",
            "AIR_D_DF_FB",
            "AIR_DA",
            "AIR_DB",
            "AIR_F_D_DFA",
            "AIR_F_D_DFB",
            "AIR_FA",
            "AIR_FB",
            "AIR_UA",
            "AIR_UB",
            "BACK_JUMP",
            "BACK_STEP",
            "CROUCH_A",
            "CROUCH_B",
            "CROUCH_FA",
            "CROUCH_FB",
            "CROUCH_GUARD",
            "DASH",
            "FOR_JUMP",
            "FORWARD_WALK",
            "JUMP",
            "NEUTRAL",
            "STAND_A",
            "STAND_B",
            "STAND_D_DB_BA",
            "STAND_D_DB_BB",
            "STAND_D_DF_FA",
            "STAND_D_DF_FB",
            "STAND_D_DF_FC",
            "STAND_F_D_DFA",
            "STAND_F_D_DFB",
            "STAND_FA",
            "STAND_FB",
            "STAND_GUARD",
            "THROW_A",
            "THROW_B",
        )
        self.audio_data = None
        self.raw_audio_memory = None
        self.just_inited = True
        self.n_frame = 1
        self.round_count = 0

        model = kwargs.get("llm_model")
        tmp = model.find(":")
        self.llm_model = model[tmp + 1 :]
        model_type = model[:tmp]
        
        # Record model information for debugging purposes.
        logger.info(f"Initialize LLMAgent - Model Type: {model_type}, Model Path: {self.llm_model}")
        
        # Select different LLM implementations based on model type
        if model_type == "local":
            # Use the global model manager to obtain a LocalLLaMA instance (supports reuse).
            logger.info("Use local models by obtaining instances through the global model manager....")
            
            # Retrieve or create a LocalLLaMA instance (the model is only loaded the first time)
            self.llm = model_manager.get_model(
                model_path=self.llm_model,
                device="cuda",
                use_quantization="auto",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.75,  # Reduce memory usage
                max_model_len=2048,           # Maximum length
                enable_prefix_caching=False,
                enforce_eager=True,
                attention_backend="FLASH_ATTN",
                output_file=f"actions-vllm-{id(self)}.txt",  # Each agent instance has a different output file.
                save_prompts=False,
            )
            
            logger.info(f"Local model instance retrieval complete. Current number of loaded models: {model_manager.get_model_count()}")
            
        else:
            logger.info(f"Use external model service: {model_type}")
            self.llm = LLamaLLM(model_type=model_type, model_name=self.llm_model)
    
        self._initialize_prompt_generator()
        
        logger.info(f"Prompt generator: {type(self.prompt_generator)}")
        
        # Other attribute initialization
        self.example = None
        self.current_actions = []
        self.pre_frame_data = None

    def _initialize_prompt_generator(self):
        self.prompt_generator = CustomPromptGenerator()

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return False

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = Key()
        # self.frameData = FrameData.get_default_instance()
        self.frameData = FrameData()
        self.cc = CommandCenter()
        self.player = player  # p1 == True, p2 == False
        self.gameData = gameData
        self.isGameJustStarted = True
        return 0

    def close(self):
        """Close the agent - but do not clean up shared models"""
        # Note: We do not clean up the LLM model here, as it may be in use by other agent instances.
        # Model cleanup is managed centrally by the global model manager.
        logger.info("LLMAgent关闭")

    def get_non_delay_frame_data(self, non_delay: FrameData):
        pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        # Load the frame data every time getInformation gets called
        self.pre_frame_data = self.frameData
        self.frameData = frame_data
        self.cc.set_frame_data(self.frameData, self.player)
        self.isControl = is_control

    def round_end(self, round_result: RoundResult):
        logger.info(round_result.remaining_hps[0])
        logger.info(round_result.remaining_hps[1])
        logger.info(round_result.elapsed_frame)
        self.just_inited = True
        self.raw_audio_memory = None
        self.round_count += 1
        self.frameData = None
        self.pre_frame_data = None
        self.current_actions = []
        logger.info("Finished {} round".format(self.round_count))

    def game_end(self):
        """Game Over - Clear game-related data but do not clear models"""
        logger.info("Game over. Clear game data.")
        self.current_actions = []
        self.frameData = None
        self.pre_frame_data = None

    def input(self):
        return self.inputKey

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def processing(self):
        if (
            self.frameData.empty_flag
            or (3600 - self.frameData.current_frame_number) <= 0
        ):
            self.isGameJustStarted = True
            return
        if self.cc.get_skill_flag():
            self.inputKey = self.cc.get_skill_key()
            return
        self.inputKey.empty()
        self.cc.skill_cancel()
        action_idx = 0
        if self.just_inited:
            self.just_inited = False
            action_idx = np.random.choice(40, 1, replace=False)[0]
            action = self.actions[int(action_idx)]
        else:
            # give action
            if len(self.current_actions) == 0:
                try:
                    current_prompt = self.prompt_generator.generate_prompt(
                        self.frameData, self.player, self.get_reward()
                    )
                    if DEBUG:
                        logger.info(f"Prompt sent to llm: \n{current_prompt}")
                    
                    # Acquire actions using an LLM instance
                    actions = self.llm.get_actions(current_prompt)
                    if DEBUG:
                        logger.info(f"Actions retrieved by llm: {actions}")
                    if len(actions) == 0:
                        return
                    action = actions[0]
                    if len(actions) > 1:
                        self.current_actions = actions[1:]
                except Exception as e:
                    logger.error(f"LLM inference error: {e}")
                    # Use random actions when an error occurs
                    action_idx = np.random.choice(40, 1, replace=False)[0]
                    action = self.actions[int(action_idx)]
            else:
                # logger.info("Continue previous action sequence")
                action = self.current_actions[0]
                if len(self.current_actions) > 1:
                    self.current_actions = self.current_actions[1:]
                else:
                    self.current_actions = []
        if action != "":
            self.cc.command_call(action)
        self.inputKey = self.cc.get_skill_key()

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data

    def get_reward(self):
        if self.pre_frame_data is None:
            return 0
        offence_reward = (
            self.pre_frame_data.get_character(not self.player).hp
            - self.frameData.get_character(not self.player).hp
        )
        defence_reward = (
            self.frameData.get_character(self.player).hp
            - self.pre_frame_data.get_character(self.player).hp
        )
        return offence_reward + defence_reward

    @staticmethod
    def get_agent_name(generator_type):
        """Static method to retrieve agent name based on prompt generator type"""
        name_mapping = {
            1: "CustomPromptGenerator",
        }
        
        if generator_type in name_mapping:
            return name_mapping[generator_type]
        else:
            raise Exception(f"Invalid prompt generator: {generator_type}")
    
    def get_model_info(self):
        """Retrieve model information for debugging purposes"""
        return {
            "model_path": self.llm_model,
            "model_type": type(self.llm).__name__,
            "total_loaded_models": model_manager.get_model_count(),
        }