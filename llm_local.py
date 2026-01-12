# ============================================================================
# 全局配置开关
# ============================================================================
ENABLE_DATA_COLLECTION = False # 设置为 False 可以完全禁用数据收集和保存功能
MAX_TOKENS = 1 # 控制生成的最大token数量，单token动作输出设为1

# ============================================================================
# 采样参数配置 - 在这里快速调整推理参数
# ============================================================================
# 核心参数
TEMPERATURE = 0.2      # 温度参数：0.1=保守策略, 0.2-0.3=平衡, 0.5+=创意策略
TOP_K = 10          # 候选动作数量：5=精确, 10=平衡, 15+=多样
TOP_P = 0.9            # 累积概率：0.8=保守, 0.9=平衡, 0.95=宽松
REPETITION_PENALTY = 1.2  # 重复惩罚：1.0=无惩罚, 1.2=适中, 1.5=强力避免重复

# 高级参数
DO_SAMPLE = False     # 是否启用采样：True=随机性, False=确定性
SKIP_SPECIAL_TOKENS = False  # 保持False以确保动作token正常输出
SPACES_BETWEEN_SPECIAL_TOKENS = False  # 保持False避免token间空格

# 快速预设配置（可选择使用）
PRESET_CONFIGS = {
    "competitive": {      # 竞技模式 - 追求最优决策
        "temperature": 0.1,
        "top_k": 5,
        "top_p": 0.8,
        "repetition_penalty": 1.05
    },
    "balanced": {         # 平衡模式 - 当前默认
        "temperature": 0.2,
        "top_k": 10,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    },
    "creative": {         # 创意模式 - 更多变化
        "temperature": 0.4,
        "top_k": 15,
        "top_p": 0.95,
        "repetition_penalty": 1.3
    },
    "defensive": {        # 防守模式 - 保守策略
        "temperature": 0.05,
        "top_k": 3,
        "top_p": 0.7,
        "repetition_penalty": 1.1
    }
}

# 选择预设配置（设为None使用上面的单独参数）
CURRENT_PRESET = None # 可选: "competitive", "balanced", "creative", "defensive"
# ============================================================================

from abc import abstractmethod
import re
import os
import json
import datetime
import atexit
import signal
from loguru import logger
from vllm import LLM as vLLM
from vllm import SamplingParams
from vllm import __version__ as vllm_version
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
from difflib import SequenceMatcher
from llm import LLM
from typing import List, Optional, Dict

# 设置环境变量以解决Triton兼容性问题
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # 使用xformers backend
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"  # 确保PTXAS路径正确
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 明确指定GPU

class LocalLLaMA(LLM):
    # 定义所有可执行的动作 - 这些应该在微调时作为特殊token添加到tokenizer中
    COMMANDABLE_ACTIONS = [
        "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP",
        "BACK_JUMP", "STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD", "STAND_A", "STAND_B", "THROW_A",
        "THROW_B", "CROUCH_A", "CROUCH_B", "STAND_FA", "STAND_FB", "CROUCH_FA",
        "CROUCH_FB", "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_D_DB_BA", "STAND_D_DB_BB",
        "STAND_D_DF_FA", "STAND_D_DF_FB", "STAND_D_DF_FC", "AIR_A",
        "AIR_B", "AIR_DA", "AIR_DB", "AIR_FA",
        "AIR_FB", "AIR_UA", "AIR_UB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB",
        "AIR_D_DF_FA", "AIR_D_DF_FB"
    ]
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda",
        use_quantization: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        enable_prefix_caching: bool = True,
        enforce_eager: Optional[bool] = None,
        attention_backend: Optional[str] = None,
        output_file: str = "actions-vllm.txt",
        save_prompts: bool = False,
        fallback_to_fuzzy: bool = True,  # 是否在token匹配失败时使用模糊匹配
        **kwargs
    ) -> None:
        """
        初始化本地模型 (使用vLLM) - 针对单token动作输出优化
        
        参数:
            model_path (str): 模型目录路径
            device (str): 加载模型的设备('cuda'或'cpu')，vLLM仅支持CUDA
            use_quantization (str): 量化选项 ('auto', 'awq', 'gptq', 'squeezellm', 'fp8', 'none')
            tensor_parallel_size (int): 张量并行大小，用于多GPU
            gpu_memory_utilization (float): GPU内存利用率 (0-1)
            max_model_len (int): 最大模型长度，None表示自动
            enable_prefix_caching (bool): 是否启用前缀缓存
            enforce_eager (bool): 是否强制使用eager模式（V100推荐True）
            attention_backend (str): 注意力后端 ('XFORMERS', 'FLASH_ATTENTION', 'FLASHINFER')
            output_file (str): 输出文件名，用于保存原始动作输出
            save_prompts (bool): 是否保存prompt到输出文件
            fallback_to_fuzzy (bool): 当token匹配失败时是否使用模糊匹配
        """
        super().__init__()
        self.device = device
        self.model = None
        self.tensor_parallel_size = tensor_parallel_size
        self.save_prompts = save_prompts
        self.fallback_to_fuzzy = fallback_to_fuzzy
        
        # 初始化动作token相关
        self.action_token_ids = {}  # 存储动作到token_id的映射
        self.id_to_action = {}      # 存储token_id到动作的映射
        
        # 训练时使用的prompt模板（与您的训练代码完全一致）
        self.training_prompt_template = (
            "Instruction:{instruction}\n"
            "Input:{input}\n"
            "Output:\n"
        )
        
        # 默认指令（与训练数据匹配）
        self.default_instruction = (
            "You are the best and most aggressive player in a "
            "2D fighting video game. You will receive the "
            "game state and must choose the best action."
        )
        
        # 初始化输出收集器（仅在启用数据收集时）
        if ENABLE_DATA_COLLECTION:
            self.output_file = output_file
            self.raw_outputs = []
            self.output_counter = 0
            self._initialize_output_file()
        else:
            self.output_file = None
            self.raw_outputs = []
            self.output_counter = 0
            logger.info("数据收集功能已禁用")
        
        # 注册清理函数
        self._register_cleanup_handlers()
        
        # 设置attention backend
        if attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
            logger.info(f"设置attention backend: {attention_backend}")
        
        logger.info(f"vLLM版本: {vllm_version}")
        
        if device != "cuda":
            logger.warning("vLLM仅支持CUDA设备，强制使用cuda")
            self.device = "cuda"
        
        logger.info(f"设备: {self.device}")
        logger.info(f"正在从以下路径加载模型: {model_path}")
        
        try:
            # 路径处理
            model_path = self._resolve_model_path(model_path)
            
            # 初始化tokenizer（用于chat template和动作token验证）
            logger.info("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=True,
            )
            
            # 确保pad token存在
            if self.tokenizer.pad_token is None:
                logger.info("分词器没有pad_token，将其设置为eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 验证和建立动作token映射
            self._build_action_token_mapping()
            
            # 确定量化配置
            quantization = self._determine_quantization(model_path, use_quantization)
            
            # 检测GPU类型
            gpu_name = ""
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                logger.info(f"检测到GPU: {gpu_name}")
            
            # vLLM配置
            vllm_kwargs = {
                "model": model_path,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": True,
                "dtype": "half",
                "tokenizer_mode": "auto",
                "disable_log_stats": True,
            }
            
            # V100特定优化
            if "v100" in gpu_name or "gv100" in gpu_name:
                logger.info("检测到V100 GPU，应用特定优化...")
                if enforce_eager is None:
                    enforce_eager = True
                vllm_kwargs.update({
                    "enable_prefix_caching": False,
                    "enable_chunked_prefill": False,
                    "enforce_eager": enforce_eager,
                    "disable_custom_all_reduce": True,
                })
            else:
                if enforce_eager is not None:
                    vllm_kwargs["enforce_eager"] = enforce_eager
                vllm_kwargs.update({
                    "enable_prefix_caching": enable_prefix_caching,
                    "enable_chunked_prefill": True,
                    "max_num_batched_tokens": 512,
                    "max_num_seqs": 1,
                })
            
            # 添加量化配置
            if quantization and quantization != "none":
                vllm_kwargs["quantization"] = quantization
                logger.info(f"使用 {quantization} 量化")
            
            # 设置最大模型长度
            vllm_kwargs["max_model_len"] = 10000
            
            # 添加其他用户自定义参数
            vllm_kwargs.update(kwargs)
            
            # 初始化vLLM
            logger.info("正在初始化vLLM引擎...")
            logger.info(f"vLLM配置: {vllm_kwargs}")
            
            try:
                self.model = vLLM(**vllm_kwargs)
            except Exception as e:
                if "triton" in str(e).lower() or "passmanager" in str(e).lower():
                    logger.warning(f"初始化失败，可能是Triton兼容性问题: {e}")
                    logger.info("尝试使用更保守的配置...")
                    
                    vllm_kwargs.update({
                        "enable_prefix_caching": False,
                        "enable_chunked_prefill": False,
                        "enforce_eager": True,
                        "disable_custom_all_reduce": True,
                        "block_size": 16,
                    })
                    
                    self.model = vLLM(**vllm_kwargs)
                else:
                    raise
            
            # 设置针对单token动作输出优化的采样参数
            self.default_sampling_params = self._create_action_sampling_params()
            
            logger.info("vLLM模型加载成功")
            
        except Exception as e:
            logger.error(f"加载模型出错: {str(e)}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise
    
    def _build_action_token_mapping(self):
        """建立动作与token_id的映射关系"""
        logger.info("正在建立动作token映射...")
        
        missing_actions = []
        found_actions = []
        
        for action in self.COMMANDABLE_ACTIONS:
            # 尝试获取token_id
            token_ids = self.tokenizer.encode(action, add_special_tokens=False)
            
            # 检查是否为单token
            if len(token_ids) == 1:
                token_id = token_ids[0]
                # 验证反向解码是否匹配
                decoded = self.tokenizer.decode([token_id])
                if decoded.strip() == action:
                    self.action_token_ids[action] = token_id
                    self.id_to_action[token_id] = action
                    found_actions.append(action)
                else:
                    logger.warning(f"动作 '{action}' 解码不匹配: '{decoded.strip()}'")
                    missing_actions.append(action)
            else:
                logger.warning(f"动作 '{action}' 不是单token: {token_ids}")
                missing_actions.append(action)
        
        logger.info(f"成功找到 {len(found_actions)} 个动作token")
        logger.info(f"动作token示例: {list(found_actions)[:5]}")
        
        if missing_actions:
            logger.warning(f"以下 {len(missing_actions)} 个动作不是单token或不在词汇表中:")
            logger.warning(f"缺失动作: {missing_actions}")
            
            if not self.fallback_to_fuzzy:
                raise ValueError(f"有 {len(missing_actions)} 个动作token缺失，且未启用模糊匹配")
        
        if not found_actions:
            raise ValueError("没有找到任何有效的动作token！请检查模型是否正确包含动作token")
        
        return len(found_actions) == len(self.COMMANDABLE_ACTIONS)
    
    def format_game_prompt(self, game_state_json: str, instruction: str = None) -> str:
        """
        使用与训练时完全相同的格式来格式化prompt
        
        Args:
            game_state_json: 游戏状态的JSON字符串
            instruction: 可选的指令，如果不提供则使用默认值
        
        Returns:
            格式化后的prompt
        """
        if instruction is None:
            instruction = self.default_instruction
        
        # 使用训练时的确切格式
        formatted_prompt = self.training_prompt_template.format(
            instruction=instruction,
            input=game_state_json
        )
        
        return formatted_prompt
    
    def _create_action_sampling_params(self) -> SamplingParams:
        """创建专门用于单token动作输出的采样参数"""
        try:
            # 首先检查是否使用预设配置
            if CURRENT_PRESET and CURRENT_PRESET in PRESET_CONFIGS:
                preset = PRESET_CONFIGS[CURRENT_PRESET]
                logger.info(f"使用预设配置: {CURRENT_PRESET}")
                
                params = {
                    'temperature': preset.get('temperature', TEMPERATURE),
                    'max_tokens': MAX_TOKENS,
                    'top_p': preset.get('top_p', TOP_P),
                    'top_k': preset.get('top_k', TOP_K),
                    'repetition_penalty': preset.get('repetition_penalty', REPETITION_PENALTY),
                    'do_sample': DO_SAMPLE,
                    'skip_special_tokens': SKIP_SPECIAL_TOKENS,
                    'spaces_between_special_tokens': SPACES_BETWEEN_SPECIAL_TOKENS,
                }
            else:
                # 使用全局配置参数（原来是硬编码的值）
                logger.info("使用全局配置参数")
                params = {
                    'temperature': TEMPERATURE,        # 原来硬编码为 0.01
                    'max_tokens': MAX_TOKENS,
                    'top_p': TOP_P,                   # 原来硬编码为 0.95
                    'top_k': TOP_K,                   # 原来硬编码为 10
                    'repetition_penalty': REPETITION_PENALTY,  # 原来硬编码为 1.0
                    'do_sample': DO_SAMPLE,           # 原来硬编码为 True
                    'skip_special_tokens': SKIP_SPECIAL_TOKENS,  # 原来硬编码为 False
                    'spaces_between_special_tokens': SPACES_BETWEEN_SPECIAL_TOKENS,  # 原来硬编码为 False
                }
            
            # 打印当前使用的参数（方便调试）
            logger.info(f"采样参数: temperature={params['temperature']}, top_k={params['top_k']}, "
                       f"top_p={params['top_p']}, do_sample={params['do_sample']}, "
                       f"repetition_penalty={params['repetition_penalty']}")
            
            return self._create_safe_sampling_params(**params)
            
        except Exception as e:
            logger.error(f"创建动作采样参数失败: {e}")
            # 使用最基本的参数作为备用
            return SamplingParams(
                temperature=TEMPERATURE,  # 使用全局配置而不是硬编码
                max_tokens=MAX_TOKENS,
            )
    
    def _create_safe_sampling_params(self, **kwargs) -> SamplingParams:
        """安全地创建SamplingParams，兼容不同版本的vLLM"""
        # 基本参数（所有版本都支持）
        safe_params = {
            'temperature': kwargs.get('temperature', TEMPERATURE),
            'max_tokens': kwargs.get('max_tokens', MAX_TOKENS),
        }
        
        # 尝试添加其他参数
        optional_params = {
            'top_p': kwargs.get('top_p', TOP_P),
            'top_k': kwargs.get('top_k', TOP_K),
            'repetition_penalty': kwargs.get('repetition_penalty', REPETITION_PENALTY),
            'stop': kwargs.get('stop', None),
            'skip_special_tokens': kwargs.get('skip_special_tokens', SKIP_SPECIAL_TOKENS),
            'spaces_between_special_tokens': kwargs.get('spaces_between_special_tokens', SPACES_BETWEEN_SPECIAL_TOKENS),
        }
        
        # 逐个尝试添加参数
        for param, value in optional_params.items():
            try:
                test_params = {**safe_params, param: value}
                _ = SamplingParams(**test_params)
                safe_params[param] = value
            except TypeError:
                logger.debug(f"参数 '{param}' 在当前vLLM版本中不支持")
        
        return SamplingParams(**safe_params)
    
    def get_actions(self, prompt: str, sampling_params: Optional[SamplingParams] = None, use_training_format: bool = True) -> List[str]:
        """
        使用vLLM生成动作 - 支持训练格式
        
        Args:
            prompt: 输入提示（可以是游戏状态JSON或已格式化的prompt）
            sampling_params: 可选的采样参数
            use_training_format: 是否使用训练时的格式（推荐True）
        
        Returns:
            动作列表
        """
        try:
            if use_training_format:
                # 检查prompt是否已经是格式化的
                if "### Instruction:" not in prompt:
                    # 假设这是游戏状态JSON，需要格式化
                    formatted_prompt = self.format_game_prompt(prompt)
                else:
                    # 已经格式化过了
                    formatted_prompt = prompt
            else:
                # 使用chat template（不推荐，因为与训练格式不匹配）
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            
            logger.info(f"使用的prompt格式: {'训练格式' if use_training_format else 'Chat格式'}")
            #logger.debug(f"完整prompt: {formatted_prompt}")
            
            # 使用提供的采样参数或默认参数
            params = sampling_params or self.default_sampling_params
            
            # 使用vLLM生成
            outputs = self.model.generate(
                prompts=[formatted_prompt],
                sampling_params=params,
                use_tqdm=False
            )
            
            # 提取生成的文本
            if outputs and len(outputs) > 0:
                response = outputs[0].outputs[0].text.strip()
            else:
                response = ""
            
            logger.info(f"vLLM生成的原始响应: '{response}'")
            
            # 提取动作
            extracted_actions = self._extract_actions_from_response(response)
            
            # 保存原始输出
            self._save_raw_output(prompt, response, extracted_actions)
            
            return extracted_actions
            
        except Exception as e:
            logger.error(f"在get_actions中出错: {str(e)}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 即使出错也要保存
            self._save_raw_output(prompt, f"ERROR: {str(e)}", [""])
            return [""]
    
    def get_actions_from_game_state(self, game_state_json: str, instruction: str = None) -> List[str]:
        """
        直接从游戏状态JSON生成动作（推荐使用）
        
        Args:
            game_state_json: 游戏状态的JSON字符串
            instruction: 可选的指令
        
        Returns:
            动作列表
        """
        formatted_prompt = self.format_game_prompt(game_state_json, instruction)
        return self.get_actions(formatted_prompt, use_training_format=True)
    
    def _extract_actions_from_response(self, response: str) -> List[str]:
        """从响应中提取动作 - 针对训练格式优化"""
        if not response or not response.strip():
            logger.warning("收到空响应")
            return ["DASH"]  # 返回默认动作
        
        response_clean = response.strip()
        logger.info(f"原始响应: '{response_clean}'")
        
        # 移除可能的多余字符和空格
        response_clean = response_clean.replace('\n', '').replace('\r', '').strip()
        
        # 1. 直接检查是否是有效的动作token
        if response_clean in self.COMMANDABLE_ACTIONS:
            logger.info(f"直接匹配到动作: {response_clean}")
            return [response_clean]
        
        # 2. 检查是否在动作token映射中
        if hasattr(self, 'action_token_ids') and response_clean in self.action_token_ids:
            logger.info(f"在动作token映射中找到: {response_clean}")
            return [response_clean]
        
        # 3. 处理可能的大小写问题
        response_upper = response_clean.upper()
        if response_upper in self.COMMANDABLE_ACTIONS:
            logger.info(f"大写匹配到动作: {response_upper}")
            return [response_upper]
        
        # 4. 如果响应包含多个token，尝试提取第一个有效动作
        words = response_clean.split()
        for word in words:
            word_clean = word.strip().upper()
            if word_clean in self.COMMANDABLE_ACTIONS:
                logger.info(f"从多词响应中提取到动作: {word_clean}")
                return [word_clean]
        
        # 5. 模糊匹配（如果启用）
        if self.fallback_to_fuzzy:
            logger.info(f"尝试模糊匹配: '{response_clean}'")
            matched_action = self._fuzzy_match_action(response_clean)
            if matched_action:
                logger.info(f"模糊匹配成功: '{response_clean}' -> '{matched_action}'")
                return [matched_action]
        
        # 6. 所有方法都失败
        logger.warning(f"无法识别的动作输出: '{response_clean}'")
        logger.warning(f"可用的动作: {self.COMMANDABLE_ACTIONS[:10]}...")
        return ["DASH"]  # 返回默认动作
    
    def _fuzzy_match_action(self, generated_action: str) -> str:
        """模糊匹配动作（作为token匹配失败时的后备方案）"""
        if not generated_action or not generated_action.strip():
            return ""
        
        generated_action = generated_action.strip().upper()
        
        # 对所有可执行动作进行相似度比较
        best_match = ""
        best_ratio = 0.0
        
        for action in self.COMMANDABLE_ACTIONS:
            ratio = SequenceMatcher(None, generated_action, action).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = action
        
        # 只有相似度足够高才返回匹配结果
        if best_ratio > 0.6:
            logger.info(f"模糊匹配: '{generated_action}' -> '{best_match}' (相似度: {best_ratio:.3f})")
            return best_match
        
        logger.warning(f"模糊匹配失败: '{generated_action}', 最佳匹配: '{best_match}' (相似度: {best_ratio:.3f})")
        return ""
    
    def batch_get_actions(self, prompts: List[str], sampling_params: Optional[SamplingParams] = None, use_training_format: bool = True) -> List[List[str]]:
        """批量生成动作（vLLM的优势所在）"""
        try:
            # 批量格式化prompts
            formatted_prompts = []
            for prompt in prompts:
                if use_training_format:
                    if "### Instruction:" not in prompt:
                        formatted_prompt = self.format_game_prompt(prompt)
                    else:
                        formatted_prompt = prompt
                else:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                formatted_prompts.append(formatted_prompt)
            
            # 使用提供的采样参数或默认参数
            params = sampling_params or self.default_sampling_params
            
            # 批量生成
            outputs = self.model.generate(
                prompts=formatted_prompts,
                sampling_params=params,
                use_tqdm=False
            )
            
            # 提取每个输出的动作
            all_actions = []
            for i, output in enumerate(outputs):
                if output and output.outputs:
                    response = output.outputs[0].text.strip()
                    actions = self._extract_actions_from_response(response)
                    all_actions.append(actions)
                    
                    # 保存原始输出（批量模式）
                    self._save_raw_output(prompts[i], response, actions, is_batch=True, batch_index=i)
                else:
                    all_actions.append(["DASH"])
                    # 保存错误输出
                    self._save_raw_output(prompts[i], "NO_OUTPUT", ["DASH"], is_batch=True, batch_index=i)
            
            return all_actions
            
        except Exception as e:
            logger.error(f"在batch_get_actions中出错: {str(e)}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            
            # 保存错误信息
            for i, prompt in enumerate(prompts):
                self._save_raw_output(prompt, f"BATCH_ERROR: {str(e)}", ["DASH"], is_batch=True, batch_index=i)
            
            return [["DASH"] for _ in prompts]
    
    def batch_get_actions_from_game_states(self, game_states: List[str], instruction: str = None) -> List[List[str]]:
        """
        批量从游戏状态生成动作
        
        Args:
            game_states: 游戏状态JSON字符串列表
            instruction: 可选的指令
        
        Returns:
            每个游戏状态对应的动作列表
        """
        formatted_prompts = [self.format_game_prompt(state, instruction) for state in game_states]
        return self.batch_get_actions(formatted_prompts, use_training_format=True)
    
    def get_action_token_info(self) -> Dict[str, any]:
        """获取动作token的详细信息"""
        return {
            "total_actions": len(self.COMMANDABLE_ACTIONS),
            "tokenized_actions": len(self.action_token_ids),
            "missing_actions": len(self.COMMANDABLE_ACTIONS) - len(self.action_token_ids),
            "action_token_mapping": self.action_token_ids.copy(),
            "fallback_enabled": self.fallback_to_fuzzy,
            "coverage_percentage": len(self.action_token_ids) / len(self.COMMANDABLE_ACTIONS) * 100
        }
    
    def test_action_tokens(self) -> Dict[str, any]:
        """测试动作token的编码和解码"""
        test_results = {
            "encoding_tests": {},
            "decoding_tests": {},
            "round_trip_tests": {}
        }
        
        for action in list(self.action_token_ids.keys())[:5]:  # 测试前5个
            token_id = self.action_token_ids[action]
            
            # 编码测试
            encoded = self.tokenizer.encode(action, add_special_tokens=False)
            test_results["encoding_tests"][action] = {
                "expected_id": token_id,
                "encoded_ids": encoded,
                "is_single_token": len(encoded) == 1,
                "matches": encoded == [token_id] if len(encoded) == 1 else False
            }
            
            # 解码测试
            decoded = self.tokenizer.decode([token_id])
            test_results["decoding_tests"][action] = {
                "token_id": token_id,
                "decoded": decoded,
                "matches": decoded.strip() == action
            }
            
            # 往返测试
            round_trip = self.tokenizer.decode(self.tokenizer.encode(action, add_special_tokens=False))
            test_results["round_trip_tests"][action] = {
                "original": action,
                "round_trip": round_trip.strip(),
                "matches": round_trip.strip() == action
            }
        
        return test_results
    
    def test_training_format(self, sample_game_state: str = None):
        """测试训练格式的推理"""
        print("\n=== 测试训练格式推理 ===")
        
        # 使用示例游戏状态（如果没有提供）
        if sample_game_state is None:
            sample_game_state = '''{"opponent": {"remaining_frames": 48, "hp": 400, "action": "STAND", "state": "STAND", "position": {"top": 435, "left": 700, "bottom": 640, "right": 740}, "speed": {"x": 0, "y": 0}, "projectiles": [], "energy": 0}, "self": {"remaining_frames": 48, "hp": 400, "state": "STAND", "position": {"top": 435, "left": 220, "bottom": 640, "right": 260}, "speed": {"x": 0, "y": 0}, "projectiles": [], "energy": 0}, "Reply with only one move."}'''
        
        print(f"测试游戏状态: {sample_game_state[:100]}...")
        
        # 测试1: 使用训练格式
        print("\n1. 使用训练格式:")
        formatted_prompt = self.format_game_prompt(sample_game_state)
        print(f"格式化的prompt:\n{formatted_prompt}")
        
        actions = self.get_actions(formatted_prompt, use_training_format=True)
        print(f"生成的动作: {actions}")
        
        # 测试2: 使用便捷方法
        print("\n2. 使用便捷方法:")
        actions2 = self.get_actions_from_game_state(sample_game_state)
        print(f"生成的动作: {actions2}")
        
        # 测试3: 对比chat格式（为了看差异）
        print("\n3. 使用Chat格式（对比）:")
        actions3 = self.get_actions(sample_game_state, use_training_format=False)
        print(f"生成的动作: {actions3}")
        
        return actions, actions2, actions3
    
    def comprehensive_debug(self):
        """全面调试动作token和生成问题"""
        print("=" * 60)
        print("开始全面调试...")
        print("=" * 60)
        
        # 1. 检查tokenizer基本信息
        print("\n1. Tokenizer信息:")
        print(f"  词汇表大小: {len(self.tokenizer.get_vocab())}")
        print(f"  特殊token: {self.tokenizer.special_tokens_map}")
        print(f"  EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        
        # 2. 检查动作token
        print("\n2. 动作Token检查:")
        valid_actions = 0
        invalid_actions = []
        
        for action in self.COMMANDABLE_ACTIONS[:10]:  # 检查前10个
            tokens = self.tokenizer.encode(action, add_special_tokens=False)
            is_single = len(tokens) == 1
            
            if is_single:
                token_id = tokens[0]
                decoded = self.tokenizer.decode([token_id])
                is_exact_match = decoded.strip() == action
                
                if is_exact_match:
                    valid_actions += 1
                    print(f"  ✓ {action}: ID={token_id}")
                else:
                    invalid_actions.append(f"{action} -> '{decoded.strip()}'")
                    print(f"  ✗ {action}: 解码不匹配 '{decoded.strip()}'")
            else:
                invalid_actions.append(f"{action} -> {len(tokens)} tokens")
                print(f"  ✗ {action}: 不是单token ({len(tokens)} tokens: {tokens})")
        
        print(f"\n  总结: {valid_actions}/{len(self.COMMANDABLE_ACTIONS[:10])} 动作有效")
        
        # 3. 测试训练格式生成
        print("\n3. 测试训练格式生成:")
        sample_state = '{"opponent": {"hp": 400}, "self": {"hp": 400}}'
        
        formatted_prompt = self.format_game_prompt(sample_state)
        print(f"  格式化prompt预览:\n{formatted_prompt[:200]}...")
        
        try:
            outputs = self.model.generate(
                prompts=[formatted_prompt],
                sampling_params=SamplingParams(
                    temperature=0.01,
                    max_tokens=1,
                    skip_special_tokens=False
                ),
                use_tqdm=False
            )
            
            if outputs and outputs[0].outputs:
                result = outputs[0].outputs[0].text
                token_ids = getattr(outputs[0].outputs[0], 'token_ids', [])
                print(f"  生成结果: '{result}' (tokens: {token_ids})")
                
                # 检查是否是有效动作
                if result.strip() in self.COMMANDABLE_ACTIONS:
                    print(f"  ✓ 生成了有效动作!")
                else:
                    print(f"  ✗ 生成的不是有效动作")
            else:
                print(f"  ✗ 没有生成结果")
                
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
        
        # 4. 建议
        print("\n4. 建议:")
        if valid_actions == 0:
            print("  ❌ 没有发现有效的动作token!")
            print("  建议:")
            print("    1. 检查模型路径是否正确")
            print("    2. 确认使用的是微调后的模型")
            print("    3. 检查tokenizer是否包含动作token")
        elif valid_actions < len(self.COMMANDABLE_ACTIONS) * 0.8:
            print("  ⚠️  只有部分动作token有效")
            print("  建议检查微调过程")
        else:
            print("  ✅ 动作token基本正确")
            print("  如果生成结果仍有问题，可能需要调整采样参数")
        
        print("\n" + "=" * 60)
    
    # ============================================================================
    # 保持原有的其他方法（清理、文件处理等）
    # ============================================================================
    
    def _register_cleanup_handlers(self):
        """注册清理处理器"""
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("清理处理器已注册")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，正在清理...")
        self._cleanup()
        exit(0)
    
    def _cleanup(self):
        """清理资源"""
        try:
            logger.info("开始清理vLLM资源...")
            
            if hasattr(self, 'model') and self.model is not None:
                try:
                    del self.model
                    self.model = None
                    logger.info("vLLM模型实例已删除")
                except Exception as e:
                    logger.warning(f"删除vLLM模型时出错: {e}")
            
            if hasattr(self, 'tensor_parallel_size') and self.tensor_parallel_size > 1:
                try:
                    if dist.is_initialized():
                        logger.info("正在销毁分布式进程组...")
                        dist.destroy_process_group()
                        logger.info("分布式进程组已销毁")
                except Exception as e:
                    logger.warning(f"销毁分布式进程组时出错: {e}")
            
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("CUDA缓存已清理")
                except Exception as e:
                    logger.warning(f"清理CUDA缓存时出错: {e}")
                    
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"清理过程中出错: {e}")
    
    def __del__(self):
        """析构函数"""
        self._cleanup()
    
    def shutdown(self):
        """手动关闭方法"""
        logger.info("手动关闭LocalLLaMA...")
        self._cleanup()
    
    def _initialize_output_file(self):
        """初始化输出文件"""
        if not ENABLE_DATA_COLLECTION:
            return
            
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"vLLM Single-Token Actions Output Log (Training Format)\n")
                f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Save prompts: {self.save_prompts}\n")
                f.write("=" * 50 + "\n\n")
            logger.info(f"输出文件初始化完成: {self.output_file}")
        except Exception as e:
            logger.error(f"初始化输出文件失败: {e}")
    
    def _save_raw_output(self, prompt: str, raw_response: str, extracted_actions: List[str], is_batch: bool = False, batch_index: int = None):
        """保存原始输出到文件"""
        if not ENABLE_DATA_COLLECTION:
            return
            
        try:
            self.output_counter += 1
            
            output_entry = {
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "counter": self.output_counter,
                "is_batch": is_batch,
                "batch_index": batch_index,
                "raw_response": raw_response,
                "extracted_actions": extracted_actions
            }
            
            if self.save_prompts:
                output_entry["prompt"] = prompt
            
            self.raw_outputs.append(output_entry)
            
            # 立即写入文件
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"Entry #{self.output_counter}\n")
                f.write(f"Timestamp: {output_entry['timestamp']}\n")
                
                if is_batch:
                    f.write(f"Batch Index: {batch_index}\n")
                
                if self.save_prompts:
                    f.write(f"Prompt: {prompt}\n")
                
                f.write(f"Raw Response: {raw_response}\n")
                f.write(f"Extracted Actions: {extracted_actions}\n")
                f.write("-" * 40 + "\n\n")
            
            logger.debug(f"原始输出已保存到文件: Entry #{self.output_counter}")
            
        except Exception as e:
            logger.error(f"保存原始输出失败: {e}")
    
    def get_raw_outputs(self) -> List[Dict]:
        """获取所有收集的原始输出"""
        if not ENABLE_DATA_COLLECTION:
            logger.warning("数据收集功能已禁用，返回空列表")
            return []
        return self.raw_outputs.copy()
    
    def export_raw_outputs_json(self, filename: str = "actions-training-format.json"):
        """将原始输出导出为JSON格式"""
        if not ENABLE_DATA_COLLECTION:
            logger.warning("数据收集功能已禁用，无法导出JSON")
            return
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_entries": len(self.raw_outputs),
                    "save_prompts": self.save_prompts,
                    "action_token_info": self.get_action_token_info(),
                    "training_format_used": True,
                    "outputs": self.raw_outputs
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"原始输出已导出为JSON: {filename}")
        except Exception as e:
            logger.error(f"导出JSON失败: {e}")
    
    def clear_raw_outputs(self):
        """清空收集的原始输出"""
        if not ENABLE_DATA_COLLECTION:
            logger.warning("数据收集功能已禁用，无需清空")
            return
            
        self.raw_outputs.clear()
        self.output_counter = 0
        logger.info("原始输出已清空")
    
    def _resolve_model_path(self, model_path: str) -> str:
        """解析并验证模型路径"""
        if not os.path.exists(model_path):
            logger.warning(f"路径未找到: {model_path}")
            alternate_path = model_path.lower()
            if os.path.exists(alternate_path):
                logger.info(f"使用替代路径: {alternate_path}")
                model_path = alternate_path
            elif os.path.exists(model_path.replace('/Home/', '/home/')):
                alternate_path = model_path.replace('/Home/', '/home/')
                logger.info(f"使用替代路径: {alternate_path}")
                model_path = alternate_path
        
        logger.info(f"最终模型路径: {model_path}")
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            logger.warning(f"配置文件不存在: {config_path}")
        else:
            logger.info(f"找到配置文件: {config_path}")
        
        return model_path
    
    def _determine_quantization(self, model_path: str, use_quantization: str) -> Optional[str]:
        """确定使用的量化方法"""
        if use_quantization.lower() in ['false', 'none']:
            logger.info("不使用量化")
            return None
        
        if use_quantization.lower() in ['awq', 'gptq', 'squeezellm', 'fp8']:
            logger.info(f"使用指定的量化方法: {use_quantization}")
            return use_quantization.lower()
        
        if use_quantization.lower() == 'auto':
            model_files = os.listdir(model_path) if os.path.exists(model_path) else []
            
            if any('awq' in f.lower() for f in model_files):
                logger.info("检测到AWQ量化模型")
                return 'awq'
            
            if any('gptq' in f.lower() or 'quantize_config.json' in f for f in model_files):
                logger.info("检测到GPTQ量化模型")
                return 'gptq'
            
            logger.info("自动模式：不使用量化（使用FP16）")
            return None
        
        logger.warning(f"未知的量化设置: {use_quantization}，不使用量化")
        return None
    
    def diagnose_environment(self) -> Dict[str, any]:
        """诊断当前环境，返回有助于调试的信息"""
        import subprocess
        
        info = {
            "vllm_version": vllm_version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "action_token_info": self.get_action_token_info(),
            "training_format_enabled": True
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["cuda_version"] = torch.version.cuda
        
        try:
            import triton
            info["triton_version"] = triton.__version__
        except:
            info["triton_version"] = "Not installed"
        
        info["env_vars"] = {
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND", "Not set"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        }
        
        try:
            nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            info["nvcc_available"] = nvcc_result.returncode == 0
        except:
            info["nvcc_available"] = False
        
        return info


# 使用示例和测试
if __name__ == "__main__":
    try:
        # 初始化模型（使用您微调后的模型路径）
        llm = LocalLLaMA(
            model_path="fightingice-llm-ai/Llama3.2/3b_bvb_1token_fix_merged",  # 或者您的模型路径
            tensor_parallel_size=1,
            output_file="actions-training-format.txt",
            save_prompts=True,
            fallback_to_fuzzy=True,
            gpu_memory_utilization=0.9
        )
        
        # 全面调试
        llm.comprehensive_debug()
        
        # 检查动作token状态
        token_info = llm.get_action_token_info()
        print(f"\n动作token信息: {token_info}")
        
        # 测试训练格式推理
        llm.test_training_format()
        
        # 测试实际游戏状态
        game_state = '''{"opponent": {"remaining_frames": 47, "hp": 400, "action": "STAND", "state": "STAND", "position": {"top": 435, "left": 700, "bottom": 640, "right": 740}, "speed": {"x": 0, "y": 0}, "projectiles": [], "energy": 0}, "self": {"remaining_frames": 47, "hp": 400, "state": "STAND", "position": {"top": 435, "left": 220, "bottom": 640, "right": 260}, "speed": {"x": 0, "y": 0}, "projectiles": [], "energy": 0}}'''
        
        print("\n=== 实际游戏测试 ===")
        actions = llm.get_actions_from_game_state(game_state)
        print(f"为游戏状态生成的动作: {actions}")
        
        # 测试批量生成
        print("\n=== 批量生成测试 ===")
        game_states = [game_state, game_state.replace('"hp": 400', '"hp": 350')]
        batch_actions = llm.batch_get_actions_from_game_states(game_states)
        print(f"批量生成的动作: {batch_actions}")
        
        # 导出结果
        llm.export_raw_outputs_json("training-format-test.json")
        
        # 环境诊断
        env_info = llm.diagnose_environment()
        print(f"\n=== 环境诊断 ===")
        for key, value in env_info.items():
            print(f"{key}: {value}")
        
        # 清理资源
        llm.shutdown()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        if 'llm' in locals():
            llm.shutdown()