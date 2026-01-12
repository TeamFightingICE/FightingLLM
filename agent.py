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

from prompt import (
    PromptGenerator,
    OpenGenerativeAIGenerator,
    OpenGenerativeAIGenerator1,
    OpenGenerativeAIGenerator2,
    OpenGenerativeAIGenerator3,
    OpenGenerativeAIGeneratorPersona,
    OpenGenerativeAIGeneratorPersona2,
    OpenGenerativeAIGeneratorPersona3,
    OpenGenerativeAIGeneratorPersona4,
    OpenGenerativeAIGeneratorPersona5,
    OpenGenerativeAIGeneratorPersona6,
    CustomPromptGenerator,  # 添加这一行
)
from llm import LLamaLLM
from loguru import logger

import logging

# 导入全局模型管理器
from model_manager import model_manager

# 在文件开头添加导入
from llm_local import LocalLLaMA

logging.basicConfig(level=logging.INFO)
STATE_DIM = {
    1: {"conv1d": 160, "fft": 512, "mel": 2560},
    4: {"conv1d": 64, "fft": 512, "mel": 1280},
}


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
        self.llm_actions = pd.read_csv("merged_actions.csv")
        self.llm_actions = self.llm_actions.to_dict(orient="records")

        # 解析模型信息
        model = kwargs.get("llm_model")
        tmp = model.find(":")
        self.llm_model = model[tmp + 1 :]
        model_type = model[:tmp]
        
        # 记录模型信息用于调试
        logger.info(f"初始化SoundAgent - 模型类型: {model_type}, 模型路径: {self.llm_model}")
        
        # 根据模型类型选择不同的 LLM 实现
        if model_type == "local":
            # 使用全局模型管理器获取LocalLLaMA实例（支持复用）
            logger.info("使用本地模型，通过全局模型管理器获取实例...")
            
            # 获取或创建LocalLLaMA实例（只在第一次时真正加载模型）
            self.llm = model_manager.get_model(
                model_path=self.llm_model,
                device="cuda",
                use_quantization="auto",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.75,  # 降低内存使用
                max_model_len=2048,           # 限制最大长度
                enable_prefix_caching=False,
                enforce_eager=True,
                attention_backend="FLASH_ATTN",
                output_file=f"actions-vllm-{id(self)}.txt",  # 每个agent实例不同的输出文件
                save_prompts=False,
            )
            
            logger.info(f"本地模型实例获取完成，当前已加载模型数: {model_manager.get_model_count()}")
            
        else:
            # 非本地模型仍使用原来的方式
            logger.info(f"使用外部模型服务: {model_type}")
            self.llm = LLamaLLM(model_type=model_type, model_name=self.llm_model)
        
        # 其他初始化参数
        self.shots = kwargs.get("shots")
        
        # 初始化提示生成器
        generator = kwargs.get("prompt_generator", 1)
        self._initialize_prompt_generator(generator, kwargs)
        
        logger.info(f"Prompt generator: {type(self.prompt_generator)}")
        
        # 其他属性初始化
        self.example = None
        self.current_actions = []
        self.pre_frame_data = None

    def _initialize_prompt_generator(self, generator, kwargs):
        """初始化提示生成器"""
        if generator == 0:
            self.prompt_generator = PromptGenerator(n_shots=self.shots)
        elif generator == 1:
            self.prompt_generator = CustomPromptGenerator()  # 使用我们的新提示生成器
        elif generator == 2:
            self.prompt_generator = OpenGenerativeAIGenerator1()
        elif generator == 3:
            self.prompt_generator = OpenGenerativeAIGenerator2()
        elif generator == 4:
            self.prompt_generator = OpenGenerativeAIGenerator3()
        elif generator == 5:
            self.prompt_generator = OpenGenerativeAIGeneratorPersona(
                persona=kwargs.get("persona")
            )
        elif generator == 6:
            self.prompt_generator = OpenGenerativeAIGeneratorPersona2(
                persona=kwargs.get("persona")
            )
        elif generator == 7:
            self.prompt_generator = OpenGenerativeAIGeneratorPersona3(
                persona=kwargs.get("persona")
            )
        elif generator == 8:
            self.prompt_generator = OpenGenerativeAIGeneratorPersona4(
                persona=kwargs.get("persona")
            )
        elif generator == 9:
            self.prompt_generator = OpenGenerativeAIGeneratorPersona5(
                persona=kwargs.get("persona")
            )
        elif generator == 10:
            self.prompt_generator = OpenGenerativeAIGeneratorPersona6(
                persona=kwargs.get("persona")
            )
        else:
            raise Exception(f"Invalid prompt generator: {generator}")

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
        """关闭agent - 但不清理共享的模型"""
        # 注意：我们不在这里清理llm模型，因为它可能被其他agent实例使用
        # 模型的清理由全局模型管理器统一管理
        logger.info("SoundAgent关闭")

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
        """游戏结束 - 清理游戏相关数据但不清理模型"""
        logger.info("游戏结束，清理游戏数据")
        self.current_actions = []
        self.frameData = None
        self.pre_frame_data = None

    def input(self):
        return self.inputKey

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def processing(self):
        start = time.perf_counter_ns()
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
        # obs = self.raw_audio_memory
        action_idx = 0
        if self.just_inited:
            self.just_inited = False
            action_idx = np.random.choice(40, 1, replace=False)[0]
            action = self.actions[int(action_idx)]
        else:
            # give action
            if len(self.current_actions) == 0:
                # logger.info("Get actions by llm")
                try:
                    current_prompt = self.prompt_generator.generate_prompt(
                        self.frameData, self.player, self.get_reward()
                    )
                    logger.info(f"Prompt sent to llm: \n{current_prompt}")
                    
                    # 使用复用的LLM实例获取动作
                    actions = self.llm.get_actions(current_prompt)
                    
                    logger.info(f"Actions retrieved by llm: {actions}")
                    if len(actions) == 0:
                        return
                    action = actions[0]
                    if len(actions) > 1:
                        self.current_actions = actions[1:]
                except Exception as e:
                    logger.error(f"LLM推理出错: {e}")
                    # 出错时使用随机动作
                    action_idx = np.random.choice(40, 1, replace=False)[0]
                    action = self.actions[int(action_idx)]
            else:
                # logger.info("Continue previous action sequence")
                action = self.current_actions[0]
                if len(self.current_actions) > 1:
                    self.current_actions = self.current_actions[1:]
                else:
                    self.current_actions = []
        end = time.perf_counter_ns()
        used_time = (end - start) / 1e6
        # logger.info(f"Executed action:{action}")
        if action != "":
            self.cc.command_call(action)
        self.inputKey = self.cc.get_skill_key()
        # with open("time.txt", "a") as f:
        #     f.write(str(used_time) + "\n")

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
        """静态方法，获取agent名称"""
        name_mapping = {
            0: PromptGenerator.__name__,
            1: "CustomPromptGenerator",  # 修正这里
            2: OpenGenerativeAIGenerator1.__name__,
            3: OpenGenerativeAIGenerator2.__name__,
            4: OpenGenerativeAIGenerator3.__name__,
            5: OpenGenerativeAIGeneratorPersona.__name__,
            6: OpenGenerativeAIGeneratorPersona2.__name__,
            7: OpenGenerativeAIGeneratorPersona3.__name__,
            8: OpenGenerativeAIGeneratorPersona4.__name__,
            9: OpenGenerativeAIGeneratorPersona5.__name__,
            10: OpenGenerativeAIGeneratorPersona6.__name__,
        }
        
        if generator_type in name_mapping:
            return name_mapping[generator_type]
        else:
            raise Exception(f"Invalid prompt generator: {generator_type}")
    
    def get_model_info(self):
        """获取模型信息用于调试"""
        return {
            "model_path": self.llm_model,
            "model_type": type(self.llm).__name__,
            "total_loaded_models": model_manager.get_model_count(),
        }