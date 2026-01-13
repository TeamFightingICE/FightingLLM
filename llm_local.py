# ============================================================================
# Global configuration for local LLM model (vLLM) usage
# ============================================================================
ENABLE_DATA_COLLECTION = False # Set to False to completely disable data collection and saving
MAX_TOKENS = 1 # Control the maximum number of tokens generated, set to 1 for single-token action output

# ============================================================================
# Sampling parameters configuration - quickly adjust inference parameters here
# ============================================================================
# Core parameters
TEMPERATURE = 0.2      # Temperature parameter: 0.1=conservative, 0.2-0.3=balanced, 0.5+=creative
TOP_K = 10          # Number of candidates: 5=precise, 10=balanced, 15+=diverse
TOP_P = 0.9            # Cumulative probability: 0.8=conservative, 0.9=balanced, 0.95=lenient
REPETITION_PENALTY = 1.2  # Repetition penalty: 1.0=no penalty, 1.2=moderate, 1.5=strong avoidance

# Advanced parameters
DO_SAMPLE = False     # Whether to enable sampling: True=randomness, False=deterministic
SKIP_SPECIAL_TOKENS = False  # Keep False to ensure action tokens are output correctly
SPACES_BETWEEN_SPECIAL_TOKENS = False  # Keep False to avoid spaces between tokens

# Quick preset configuration (optional)
PRESET_CONFIGS = {
    "competitive": {      # Competitive mode – pursue optimal decisions.
        "temperature": 0.1,
        "top_k": 5,
        "top_p": 0.8,
        "repetition_penalty": 1.05
    },
    "balanced": {         # Balanced mode - current default
        "temperature": 0.2,
        "top_k": 10,
        "top_p": 0.9,
        "repetition_penalty": 1.2
    },
    "creative": {         # Creative mode - more variation
        "temperature": 0.4,
        "top_k": 15,
        "top_p": 0.95,
        "repetition_penalty": 1.3
    },
    "defensive": {        # Defensive mode - conservative strategy
        "temperature": 0.05,
        "top_k": 3,
        "top_p": 0.7,
        "repetition_penalty": 1.1
    }
}

# Select a preset configuration (set to None to use the individual parameters above)
CURRENT_PRESET = None # Optional: "competitive", "balanced", "creative", "defensive"
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

# Set environment variables to resolve Triton compatibility issues
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # FLASH_ATTN backend
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"  # Ensure the PTXAS path is correct.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Explicitly specify the GPU.

DEBUG = False  # Set to True to enable debug logs

class LocalLLaMA(LLM):
    # Define all executable actions — these should be added to the tokenizer as special tokens during fine-tuning.
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
        fallback_to_fuzzy: bool = True,  
        **kwargs
    ) -> None:
        """
        Initialize a local model (using vLLM) — optimized for single-token action output.

        Parameters:
            model_path (str): Path to the model directory.
            device (str): Device to load the model on ('cuda' or 'cpu'); vLLM only supports CUDA.
            use_quantization (str): Quantization option ('auto', 'awq', 'gptq', 'squeezellm', 'fp8', 'none').
            tensor_parallel_size (int): Tensor parallel size, for multi-GPU setups.
            gpu_memory_utilization (float): GPU memory utilization (0–1).
            max_model_len (int): Maximum model length; None means auto.
            enable_prefix_caching (bool): Whether to enable prefix caching.
            enforce_eager (bool): Whether to force eager mode (True is recommended for V100).
            attention_backend (str): Attention backend ('XFORMERS', 'FLASH_ATTENTION', 'FLASHINFER').
            output_file (str): Output filename, used to save raw action outputs.
            save_prompts (bool): Whether to save prompts to the output file.
            fallback_to_fuzzy (bool): Whether to use fuzzy matching when token matching fails.
        """
        super().__init__()
        self.device = device
        self.model = None
        self.tensor_parallel_size = tensor_parallel_size
        self.save_prompts = save_prompts
        self.fallback_to_fuzzy = fallback_to_fuzzy
        
        # Initialize action token–related components.
        self.action_token_ids = {}  # Store mapping from actions to token IDs
        self.id_to_action = {}      # Store mapping from token IDs to actions

        # Training prompt template (consistent with your training code)
        self.training_prompt_template = (
            "Instruction:{instruction}\n"
            "Input:{input}\n"
            "Output:\n"
        )

        # Default instruction (consistent with training data)
        self.default_instruction = (
            "You are the best and most aggressive player in a "
            "2D fighting video game. You will receive the "
            "game state and must choose the best action."
        )

        # Initialize output collector (only if data collection is enabled)
        if ENABLE_DATA_COLLECTION:
            self.output_file = output_file
            self.raw_outputs = []
            self.output_counter = 0
            self._initialize_output_file()
        else:
            self.output_file = None
            self.raw_outputs = []
            self.output_counter = 0
            logger.info("Data collection is disabled.")
        
        # Register a cleanup function.
        self._register_cleanup_handlers()
        
        # Set the attention backend.
        if attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
            logger.info(f"Set the attention backend: {attention_backend}")
        
        logger.info(f"vLLM version: {vllm_version}")
        
        if device != "cuda":
            logger.warning("vLLM only supports CUDA devices; force the use of CUDA.")
            self.device = "cuda"
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Loading the model from the following path: {model_path}")
        
        try:
            model_path = self._resolve_model_path(model_path)
            
            # Initialize the tokenizer (for chat templates and action-token validation)
            logger.info("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                trust_remote_code=True,
            )
            
            # Ensure the pad token exists.
            if self.tokenizer.pad_token is None:
                logger.info("The tokenizer has no pad_token; set it to eos_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Validate and build the action-token mapping
            self._build_action_token_mapping()
            
            # Determine the quantization configuration
            quantization = self._determine_quantization(model_path, use_quantization)
            
            # Detect GPU type
            gpu_name = ""
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                logger.info(f"Detected GPU: {gpu_name}")
            
            # vLLM configuration
            vllm_kwargs = {
                "model": model_path,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": True,
                "dtype": "half",
                "tokenizer_mode": "auto",
                "disable_log_stats": True,
            }
            
            # V100 specific optimization
            if "v100" in gpu_name or "gv100" in gpu_name:
                logger.info("Detected V100 GPU, applying specific optimizations...")
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
            
            # Add the quantization configuration
            if quantization and quantization != "none":
                vllm_kwargs["quantization"] = quantization
                logger.info(f"Using {quantization} quantization.")
            
            # Set the maximum model length
            vllm_kwargs["max_model_len"] = 10000
            
            # Add other user-defined parameters
            vllm_kwargs.update(kwargs)
            
            # Initialize vLLM
            logger.info("Initializing the vLLM engine....")
            logger.info(f"vLLM configuration: {vllm_kwargs}")
            
            try:
                self.model = vLLM(**vllm_kwargs)
            except Exception as e:
                if "triton" in str(e).lower() or "passmanager" in str(e).lower():
                    logger.warning(f"Initialization failed, possibly due to Triton compatibility issues: {e}")
                    logger.info("Attempting to use a more conservative configuration...")
                    
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
            
            # Set sampling parameters optimized for single-token action output
            self.default_sampling_params = self._create_action_sampling_params()
            
            logger.info("vLLM model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading the model.: {str(e)}")
            import traceback
            logger.error(f"Error details: {traceback.format_exc()}")
            raise e
    
    def _build_action_token_mapping(self):
        """“Establish the mapping between actions and token IDs, ensuring each action corresponds to a single token."""
        logger.info("Building the action-token mapping...")
        
        missing_actions = []
        found_actions = []
        
        for action in self.COMMANDABLE_ACTIONS:
            # Attempt to obtain the token ID
            token_ids = self.tokenizer.encode(action, add_special_tokens=False)
            
            # Check whether it is a single token
            if len(token_ids) == 1:
                token_id = token_ids[0]
                # Verify whether the reverse decoding matches
                decoded = self.tokenizer.decode([token_id])
                if decoded.strip() == action:
                    self.action_token_ids[action] = token_id
                    self.id_to_action[token_id] = action
                    found_actions.append(action)
                else:
                    logger.warning(f"Action '{action}' decoding mismatch: '{decoded.strip()}'")
                    missing_actions.append(action)
            else:
                logger.warning(f"Action '{action}' is not a single token: {token_ids}")
                missing_actions.append(action)
        
        logger.info(f"Successfully found {len(found_actions)} action tokens.")
        logger.info(f"Example action tokens: {list(found_actions)[:5]}")
        
        if missing_actions:
            logger.warning(f"The following {len(missing_actions)} actions are not single tokens or are not in the vocabulary:")
            logger.warning(f"Missing actions: {missing_actions}")
            
            if not self.fallback_to_fuzzy:
                raise ValueError(f"There are {len(missing_actions)} missing action tokens, and fuzzy matching is not enabled.")
        
        if not found_actions:
            raise ValueError("No valid action tokens were found! Please check whether the model correctly includes the action tokens.")
        
        return len(found_actions) == len(self.COMMANDABLE_ACTIONS)
    
    def format_game_prompt(self, game_state_json: str, instruction: str = None) -> str:
        """
        Format the prompt using exactly the same format as during training.

        Args:
            game_state_json: JSON string of the game state.
            instruction: Optional instruction; if not provided, the default value is used.

        Returns:
            The formatted prompt.
        """
        if instruction is None:
            instruction = self.default_instruction
        
        # Use the exact format from training
        formatted_prompt = self.training_prompt_template.format(
            instruction=instruction,
            input=game_state_json
        )
        
        return formatted_prompt
    
    def _create_action_sampling_params(self) -> SamplingParams:
        """Create sampling parameters specifically for single-token action outputs."""
        try:
            # 首先检查是否使用预设配置
            if CURRENT_PRESET and CURRENT_PRESET in PRESET_CONFIGS:
                preset = PRESET_CONFIGS[CURRENT_PRESET]
                logger.info(f"Using preset configuration: {CURRENT_PRESET}")
                
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
                # Use global configuration parameters (previously hard-coded values).
                logger.info("Using global configuration parameters")
                params = {
                    'temperature': TEMPERATURE,       
                    'max_tokens': MAX_TOKENS,
                    'top_p': TOP_P,                   
                    'top_k': TOP_K,                   
                    'repetition_penalty': REPETITION_PENALTY,  
                    'do_sample': DO_SAMPLE,           
                    'skip_special_tokens': SKIP_SPECIAL_TOKENS,  
                    'spaces_between_special_tokens': SPACES_BETWEEN_SPECIAL_TOKENS, 
                }
            
            # Print the currently used parameters (for easier debugging)
            logger.info(f"Sampling parameters: temperature={params['temperature']}, top_k={params['top_k']}, "
                       f"top_p={params['top_p']}, do_sample={params['do_sample']}, "
                       f"repetition_penalty={params['repetition_penalty']}")
            
            return self._create_safe_sampling_params(**params)
            
        except Exception as e:
            logger.error(f"Failed to create action sampling parameters: {e}")
            # Use the most basic parameters as a fallback
            return SamplingParams(
                temperature=TEMPERATURE,  # Use global configuration instead of hardcoding
                max_tokens=MAX_TOKENS,
            )
    
    def _create_safe_sampling_params(self, **kwargs) -> SamplingParams:
        """Safely create SamplingParams, compatible with different vLLM versions"""
        # Basic parameters (supported by all versions)
        safe_params = {
            'temperature': kwargs.get('temperature', TEMPERATURE),
            'max_tokens': kwargs.get('max_tokens', MAX_TOKENS),
        }
        
        # Try adding other parameters
        optional_params = {
            'top_p': kwargs.get('top_p', TOP_P),
            'top_k': kwargs.get('top_k', TOP_K),
            'repetition_penalty': kwargs.get('repetition_penalty', REPETITION_PENALTY),
            'stop': kwargs.get('stop', None),
            'skip_special_tokens': kwargs.get('skip_special_tokens', SKIP_SPECIAL_TOKENS),
            'spaces_between_special_tokens': kwargs.get('spaces_between_special_tokens', SPACES_BETWEEN_SPECIAL_TOKENS),
        }
        
        # Try adding parameters one by one
        for param, value in optional_params.items():
            try:
                test_params = {**safe_params, param: value}
                _ = SamplingParams(**test_params)
                safe_params[param] = value
            except TypeError:
                logger.debug(f"Parameter '{param}' is not supported in the current vLLM version")
        
        return SamplingParams(**safe_params)
    
    def get_actions(self, prompt: str, sampling_params: Optional[SamplingParams] = None, use_training_format: bool = True) -> List[str]:
        """
        Generate actions using vLLM — supports the training format.

        Args:
            prompt: Input prompt (can be a game-state JSON or a preformatted prompt).
            sampling_params: Optional sampling parameters.
            use_training_format: Whether to use the training-time format (recommended: True).

        Returns:
            A list of actions.

        """
        try:
            if use_training_format:
                # Check whether the prompt is already formatted
                if "### Instruction:" not in prompt:
                    # Assume this is a game-state JSON and needs to be formatted.
                    formatted_prompt = self.format_game_prompt(prompt)
                else:
                    # Already formatted.
                    formatted_prompt = prompt
            else:
                # Use chat template (not recommended, as it doesn't match the training format)
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            
            if DEBUG:logger.info(f"Prompt format in use: {'Training format' if use_training_format else 'Chat format'}")
            
            # Use the provided sampling parameters or the default parameters.
            params = sampling_params or self.default_sampling_params
            
            # Generate using vLLM.
            outputs = self.model.generate(
                prompts=[formatted_prompt],
                sampling_params=params,
                use_tqdm=False
            )
            
            # Extract the generated text.
            if outputs and len(outputs) > 0:
                response = outputs[0].outputs[0].text.strip()
            else:
                response = ""
            
            # logger.info(f"vLLM generated raw response: '{response}'")
            
            # Extract actions
            extracted_actions = self._extract_actions_from_response(response)
            
            # Save raw output
            self._save_raw_output(prompt, response, extracted_actions)
            
            return extracted_actions
            
        except Exception as e:
            logger.error(f"Error in get_actions: {str(e)}")
            import traceback
            logger.error(f"Error details: {traceback.format_exc()}")
            # Even if there's an error, save the raw output
            self._save_raw_output(prompt, f"ERROR: {str(e)}", [""])
            return [""]
    
    def get_actions_from_game_state(self, game_state_json: str, instruction: str = None) -> List[str]:
        """
        Directly generate actions from a game-state JSON (recommended)
        
        Args:
            game_state_json: Game state as a JSON string.
            instruction: Optional instruction.

        Returns:
            List of actions.
        """
        formatted_prompt = self.format_game_prompt(game_state_json, instruction)
        return self.get_actions(formatted_prompt, use_training_format=True)
    
    def _extract_actions_from_response(self, response: str) -> List[str]:
        """Extract actions from the response — optimized for the training format."""
        if not response or not response.strip():
            if DEBUG: logger.warning("Received an empty response.")
            return ["DASH"]  # Return the default action
        
        response_clean = response.strip()
        if DEBUG:
            logger.info(f"Raw response: '{response_clean}'")
        
        # Remove any possible extra characters and whitespace
        response_clean = response_clean.replace('\n', '').replace('\r', '').strip()
        
        # 1. Directly check whether it is a valid action token.
        if response_clean in self.COMMANDABLE_ACTIONS:
            if DEBUG: logger.info(f"Directly matched action: {response_clean}")
            return [response_clean]
        
        # 2. Check if it is in the action token mapping.
        if hasattr(self, 'action_token_ids') and response_clean in self.action_token_ids:
            if DEBUG: logger.info(f"Found in action token mapping: {response_clean}")
            return [response_clean]
        
        # 3. Handle potential case-sensitivity issues.
        response_upper = response_clean.upper()
        if response_upper in self.COMMANDABLE_ACTIONS:
            if DEBUG: logger.info(f"Uppercase matched action: {response_upper}")
            return [response_upper]
        
        # 4. If the response contains multiple tokens, try to extract the first valid action
        words = response_clean.split()
        for word in words:
            word_clean = word.strip().upper()
            if word_clean in self.COMMANDABLE_ACTIONS:
                if DEBUG: logger.info(f"Extracted an action from a multi-word response: {word_clean}")
                return [word_clean]
        
        # 5. Fuzzy matching (if enabled).
        if self.fallback_to_fuzzy:
            if DEBUG: logger.info(f"Attempting fuzzy matching: '{response_clean}'")
            matched_action = self._fuzzy_match_action(response_clean)
            if matched_action:
                if DEBUG: logger.info(f"Fuzzy match successful: '{response_clean}' -> '{matched_action}'")
                return [matched_action]
        
        # 6. All methods failed.
        if DEBUG: logger.warning(f"Unrecognized action output: '{response_clean}'")
        if DEBUG: logger.warning(f"Available actions: {self.COMMANDABLE_ACTIONS[:10]}...")
        return ["DASH"]  # Return default action
    
    def _fuzzy_match_action(self, generated_action: str) -> str:
        """Fuzzy match actions (as a fallback when token matching fails)"""
        if not generated_action or not generated_action.strip():
            return ""
        
        generated_action = generated_action.strip().upper()
        
        # Compare similarity against all executable actions.
        best_match = ""
        best_ratio = 0.0
        
        for action in self.COMMANDABLE_ACTIONS:
            ratio = SequenceMatcher(None, generated_action, action).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = action
        
        # Only return the match if the similarity is high enough
        if best_ratio > 0.6:
            if DEBUG: logger.info(f"Fuzzy match: '{generated_action}' -> '{best_match}' (similarity: {best_ratio:.3f})")
            return best_match

        if DEBUG: logger.warning(f"Fuzzy match failed: '{generated_action}', best match: '{best_match}' (similarity: {best_ratio:.3f})")
        return ""
    
    def batch_get_actions(self, prompts: List[str], sampling_params: Optional[SamplingParams] = None, use_training_format: bool = True) -> List[List[str]]:
        """Batch action generation (where vLLM excels)"""
        try:
            # Batch format prompts
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
            
            # Use the provided sampling parameters or the default parameters.
            params = sampling_params or self.default_sampling_params
            
            # Batch generate
            outputs = self.model.generate(
                prompts=formatted_prompts,
                sampling_params=params,
                use_tqdm=False
            )
            
            # Extract the action from each output.
            all_actions = []
            for i, output in enumerate(outputs):
                if output and output.outputs:
                    response = output.outputs[0].text.strip()
                    actions = self._extract_actions_from_response(response)
                    all_actions.append(actions)
                    
                    # Save raw outputs (batch mode)
                    self._save_raw_output(prompts[i], response, actions, is_batch=True, batch_index=i)
                else:
                    all_actions.append(["DASH"])
                    # Save error output
                    self._save_raw_output(prompts[i], "NO_OUTPUT", ["DASH"], is_batch=True, batch_index=i)
            
            return all_actions
            
        except Exception as e:
            logger.error(f"Error in batch_get_actions: {str(e)}")
            import traceback
            logger.error(f"Error details: {traceback.format_exc()}")
            
            # Save error information
            for i, prompt in enumerate(prompts):
                self._save_raw_output(prompt, f"BATCH_ERROR: {str(e)}", ["DASH"], is_batch=True, batch_index=i)
            
            return [["DASH"] for _ in prompts]
    
    def batch_get_actions_from_game_states(self, game_states: List[str], instruction: str = None) -> List[List[str]]:
        """
            Batch-generate actions from game states.

            Args:
                game_states: A list of JSON strings representing game states.
                instruction: Optional instruction.

            Returns:
                A list of actions corresponding to each game state.
        """
        formatted_prompts = [self.format_game_prompt(state, instruction) for state in game_states]
        return self.batch_get_actions(formatted_prompts, use_training_format=True)
    
    def get_action_token_info(self) -> Dict[str, any]:
        """Get detailed information about action tokens."""
        return {
            "total_actions": len(self.COMMANDABLE_ACTIONS),
            "tokenized_actions": len(self.action_token_ids),
            "missing_actions": len(self.COMMANDABLE_ACTIONS) - len(self.action_token_ids),
            "action_token_mapping": self.action_token_ids.copy(),
            "fallback_enabled": self.fallback_to_fuzzy,
            "coverage_percentage": len(self.action_token_ids) / len(self.COMMANDABLE_ACTIONS) * 100
        }
    
    def test_action_tokens(self) -> Dict[str, any]:
        """Get detailed information about action tokens."""
        test_results = {
            "encoding_tests": {},
            "decoding_tests": {},
            "round_trip_tests": {}
        }
        
        for action in list(self.action_token_ids.keys())[:5]: # Test the first 5 actions
            token_id = self.action_token_ids[action]
            
            # Encoding test.
            encoded = self.tokenizer.encode(action, add_special_tokens=False)
            test_results["encoding_tests"][action] = {
                "expected_id": token_id,
                "encoded_ids": encoded,
                "is_single_token": len(encoded) == 1,
                "matches": encoded == [token_id] if len(encoded) == 1 else False
            }
            
            # Decoding test
            decoded = self.tokenizer.decode([token_id])
            test_results["decoding_tests"][action] = {
                "token_id": token_id,
                "decoded": decoded,
                "matches": decoded.strip() == action
            }
            
            # Round-trip test
            round_trip = self.tokenizer.decode(self.tokenizer.encode(action, add_special_tokens=False))
            test_results["round_trip_tests"][action] = {
                "original": action,
                "round_trip": round_trip.strip(),
                "matches": round_trip.strip() == action
            }
        
        return test_results
    
    def test_training_format(self, sample_game_state: str = None):
        """Test training format reasoning"""
        print("\n=== Test inference with the training format===")
        
        # Use an example game state (if none is provided).
        if sample_game_state is None:
            sample_game_state = '''{"opponent": {"remaining_frames": 48, "hp": 400, "action": "STAND", "state": "STAND", "position": {"top": 435, "left": 700, "bottom": 640, "right": 740}, "speed": {"x": 0, "y": 0}, "projectiles": [], "energy": 0}, "self": {"remaining_frames": 48, "hp": 400, "state": "STAND", "position": {"top": 435, "left": 220, "bottom": 640, "right": 260}, "speed": {"x": 0, "y": 0}, "projectiles": [], "energy": 0}, "Reply with only one move."}'''
        
        print(f"Test game state: {sample_game_state[:100]}...")
        
        # Test 1: Using the training format.
        print("\n1. Using the training format:")
        formatted_prompt = self.format_game_prompt(sample_game_state)
        print(f"Formatted prompt:\n{formatted_prompt}")
        
        actions = self.get_actions(formatted_prompt, use_training_format=True)
        print(f"Generated actions: {actions}")
        
        # Test 2: Using the convenience method
        print("\n2. Using the convenience method:")
        actions2 = self.get_actions_from_game_state(sample_game_state)
        print(f"Generated actions: {actions2}")
        
        # Test 3: Compare with chat format (to see differences)
        print("\n3. Using Chat format (comparison):")
        actions3 = self.get_actions(sample_game_state, use_training_format=False)
        print(f"Generated actions: {actions3}")
        
        return actions, actions2, actions3
    
    def comprehensive_debug(self):
        """Comprehensively debug action tokens and generation issues."""
        print("=" * 60)
        print("Start comprehensive debugging...")
        print("=" * 60)
        
        # 1. Check basic tokenizer information.
        print("\n1. Tokenizer information:")
        print(f"  Vocabulary size: {len(self.tokenizer.get_vocab())}")
        print(f"  Special tokens: {self.tokenizer.special_tokens_map}")
        print(f"  EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        
        # 2. Check action tokens
        print("\n2. Action token check:")
        valid_actions = 0
        invalid_actions = []
        
        for action in self.COMMANDABLE_ACTIONS[:10]:  
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
                    print(f"  ✗ {action}: Decoding mismatch '{decoded.strip()}'")
            else:
                invalid_actions.append(f"{action} -> {len(tokens)} tokens")
                print(f"  ✗ {action}: not a single token ({len(tokens)} tokens: {tokens})")
        
        print(f"\n  Summary: {valid_actions}/{len(self.COMMANDABLE_ACTIONS[:10])} actions are valid")
        
        # 3. Test training format generation
        print("\n3. Test training format generation:")
        sample_state = '{"opponent": {"hp": 400}, "self": {"hp": 400}}'
        
        formatted_prompt = self.format_game_prompt(sample_state)
        print(f"  Formatted prompt preview:\n{formatted_prompt[:200]}...")
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
                print(f"  Generated result: '{result}' (tokens: {token_ids})")
                
                # Check whether it is a valid action.
                if result.strip() in self.COMMANDABLE_ACTIONS:
                    print(f"  ✓ Generated a valid action!")
                else:
                    print(f"  ✗ Generated an invalid action")
            else:
                print(f"  ✗ No generation result")
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")

        # 4. Suggestions
        print("\n4. Suggestions:")
        if valid_actions == 0:
            print("  ❌ No valid action tokens found!")
            print("  Suggestions:")
            print("    1. Check if the model path is correct")
            print("    2. Confirm that you are using a fine-tuned model")
            print("    3. Check if the tokenizer contains action tokens")
        elif valid_actions < len(self.COMMANDABLE_ACTIONS) * 0.8:
            print("  ⚠️  Only a portion of action tokens are valid")
            print("  Suggestion: Check the fine-tuning process")
        else:
            print("  ✅ Action tokens are basically correct")
            print("  If the generation results still have issues, you may need to adjust the sampling parameters")

        print("\n" + "=" * 60)
    
    # ============================================================================
    # Keep the existing other methods (cleanup, file handling, etc.).
    # ============================================================================
    
    def _register_cleanup_handlers(self):
        """Register the cleanup handler"""
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if DEBUG: logger.info("Cleanup handler registered.")
    
    def _signal_handler(self, signum, frame):
        """Signal handler"""
        logger.info(f"Received signal {signum}, cleaning up...")
        self._cleanup()
        exit(0)
    
    def _cleanup(self):
        """Clean up resources."""
        try:
            if DEBUG: logger.info("Start cleaning up vLLM resources....")
            
            if hasattr(self, 'model') and self.model is not None:
                try:
                    del self.model
                    self.model = None
                    if DEBUG: logger.info("vLLM model instance has been deleted.")
                except Exception as e:
                    logger.warning(f"Error while deleting the vLLM model: {e}")
            
            if hasattr(self, 'tensor_parallel_size') and self.tensor_parallel_size > 1:
                try:
                    if dist.is_initialized():
                        if DEBUG: logger.info("Destroying the distributed process group...")
                        dist.destroy_process_group()
                        if DEBUG: logger.info("Distributed process group destroyed")
                except Exception as e:
                    logger.warning(f"Error while destroying the distributed process group: {e}")

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("CUDA cache has been cleared")
                except Exception as e:
                    logger.warning(f"Error while clearing CUDA cache: {e}")
                    
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor"""
        self._cleanup()
    
    def shutdown(self):
        """Manual shutdown method"""
        logger.info("Manually shut down LocalLLaMA...")
        self._cleanup()
    
    def _initialize_output_file(self):
        """Initialize the output file"""
        if not ENABLE_DATA_COLLECTION:
            return
            
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"vLLM Single-Token Actions Output Log (Training Format)\n")
                f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Save prompts: {self.save_prompts}\n")
                f.write("=" * 50 + "\n\n")
            logger.info(f"Output file initialized: {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to initialize output file: {e}")
    
    def _save_raw_output(self, prompt: str, raw_response: str, extracted_actions: List[str], is_batch: bool = False, batch_index: int = None):
        """Save raw output to file"""
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
            
            # Write to file immediately
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
            
            logger.debug(f"Raw output saved to file: Entry #{self.output_counter}")
            
        except Exception as e:
            logger.error(f"Failed to save raw output: {e}")
    
    def get_raw_outputs(self) -> List[Dict]:
        """Get all collected raw outputs"""
        if not ENABLE_DATA_COLLECTION:
            logger.warning("Data collection is disabled; return an empty list.")
            return []
        return self.raw_outputs.copy()
    
    def export_raw_outputs_json(self, filename: str = "actions-training-format.json"):
        """Export raw outputs in JSON format"""
        if not ENABLE_DATA_COLLECTION:
            logger.warning("Data collection is disabled; cannot export JSON")
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
            logger.info(f"Raw outputs exported as JSON: {filename}")
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
    
    def clear_raw_outputs(self):
        """Clear the collected raw outputs"""
        if not ENABLE_DATA_COLLECTION:
            logger.warning("Data collection is disabled; no need to clear")
            return
            
        self.raw_outputs.clear()
        self.output_counter = 0
        logger.info("Raw outputs cleared")
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Parse and validate the model path"""
        if not os.path.exists(model_path):
            logger.warning(f"Path not found: {model_path}")
            alternate_path = model_path.lower()
            if os.path.exists(alternate_path):
                logger.info(f"Use an alternative path: {alternate_path}")
                model_path = alternate_path
            elif os.path.exists(model_path.replace('/Home/', '/home/')):
                alternate_path = model_path.replace('/Home/', '/home/')
                logger.info(f"Use an alternative path: {alternate_path}")
                model_path = alternate_path
        
        logger.info(f"Final model path: {model_path}")
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file does not exist.: {config_path}")
        else:
            logger.info(f"Found configuration file: {config_path}")
        
        return model_path
    
    def _determine_quantization(self, model_path: str, use_quantization: str) -> Optional[str]:
        """Determine the quantization method to use"""
        if use_quantization.lower() in ['false', 'none']:
            logger.info("Do not use quantization")
            return None
        
        if use_quantization.lower() in ['awq', 'gptq', 'squeezellm', 'fp8']:
            logger.info(f"Use the specified quantization method: {use_quantization}")
            return use_quantization.lower()
        
        if use_quantization.lower() == 'auto':
            model_files = os.listdir(model_path) if os.path.exists(model_path) else []
            
            if any('awq' in f.lower() for f in model_files):
                logger.info("Detected AWQ quantized model")
                return 'awq'
            
            if any('gptq' in f.lower() or 'quantize_config.json' in f for f in model_files):
                logger.info("Detected GPTQ quantized model")
                return 'gptq'
            
            logger.info("Auto mode: no quantization (using FP16)")
            return None
        
        logger.warning(f"Unknown quantization setting: {use_quantization}, not using quantization")
        return None
    
    def diagnose_environment(self) -> Dict[str, any]:
        """Diagnose the current environment and return information helpful for debugging"""
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


# Use examples and tests
if __name__ == "__main__":
    try:
        # Initialize the model (using your fine-tuned model path)
        llm = LocalLLaMA(
            model_path="fightingice-llm-ai/Llama3.2/3b_bvb_1token_fix_merged", 
            tensor_parallel_size=1,
            output_file="actions-training-format.txt",
            save_prompts=True,
            fallback_to_fuzzy=True,
            gpu_memory_utilization=0.9
        )
        DEBUG = True  # Enable debug logging
        # Comprehensive debugging
        llm.comprehensive_debug()
        
        # Check action token status.
        token_info = llm.get_action_token_info()
        print(f"\nAction token information: {token_info}")
        
        # Test training-format inference
        llm.test_training_format()
        
        # Test with an actual game state
        game_state = '''{"opponent": {"remaining_frames": 47, "hp": 400, "action": "STAND", "state": "STAND", "position": {"top": 435, "left": 700, "bottom": 640, "right": 740}, "speed": {"x": 0, "y": 0}, "projectiles": [], "energy": 0}, "self": {"remaining_frames": 47, "hp": 400, "state": "STAND", "position": {"top": 435, "left": 220, "bottom": 640, "right": 260}, "speed": {"x": 0, "y": 0}, "projectiles": [], "energy": 0}}'''
        
        print("\n=== Actual game test ===")
        actions = llm.get_actions_from_game_state(game_state)
        print(f"Action generated for the game state: {actions}")
        
        # Test batch generation
        print("\n=== Batch generation test ===")
        game_states = [game_state, game_state.replace('"hp": 400', '"hp": 350')]
        batch_actions = llm.batch_get_actions_from_game_states(game_states)
        print(f"Batch generated actions: {batch_actions}")
        
        # Export results
        llm.export_raw_outputs_json("training-format-test.json")
        
        # Environment diagnosis
        env_info = llm.diagnose_environment()
        print(f"\n=== Environment Diagnosis ===")
        for key, value in env_info.items():
            print(f"{key}: {value}")
        
        # Clean up resources.
        llm.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        if 'llm' in locals():
            llm.shutdown()