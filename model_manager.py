# model_manager.py - 创建这个新文件
import threading
import torch
import gc
from typing import Dict, Optional, Any
from loguru import logger

class GlobalModelManager:
    """Global model manager — ensure the model is loaded only once."""
    
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _model_configs: Dict[str, Dict] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            logger.info("Global model manager initialized")
    
    def get_model(self, model_path: str, **kwargs):
        """Get the model instance; create it if it does not exist."""
        
        # Create a unique identifier for the configuration.
        config_key = self._create_config_key(model_path, **kwargs)
        
        if config_key not in self._models:
            logger.info(f"First-time model loading: {model_path}")
            logger.info(f"Configuration: {kwargs}")
            
            # Check GPU memory
            self._check_gpu_memory()
            
            # Import LocalLLaMA
            from llm_local import LocalLLaMA
            
            # Set memory optimization parameters
            optimized_kwargs = self._optimize_memory_config(**kwargs)
            
            try:
                # Create model instance
                model = LocalLLaMA(
                    model_path=model_path,
                    **optimized_kwargs
                )
                
                self._models[config_key] = model
                self._model_configs[config_key] = {
                    'model_path': model_path,
                    'config': optimized_kwargs
                }

                logger.info(f"Model loading completed: {config_key}")
                self._check_gpu_memory()
                
            except torch.OutOfMemoryError as e:
                logger.error(f"Insufficient GPU memory: {e}")
                self._suggest_memory_optimizations()
                raise
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                raise
        else:
            logger.info(f"Reusing already loaded model: {config_key}")
        
        return self._models[config_key]
    
    def _create_config_key(self, model_path: str, **kwargs) -> str:
        """Create a unique identifier for the configuration."""
        # Only include parameters that affect model loading
        key_params = {
            'model_path': model_path,
            'tensor_parallel_size': kwargs.get('tensor_parallel_size', 1),
            'gpu_memory_utilization': kwargs.get('gpu_memory_utilization', 0.9),
            'use_quantization': kwargs.get('use_quantization', 'auto'),
            'max_model_len': kwargs.get('max_model_len', None),
        }
        return str(hash(str(sorted(key_params.items()))))
    
    def _optimize_memory_config(self, **kwargs) -> dict:
        """Optimize the configuration based on GPU memory"""
        
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        logger.info(f"GPU Total memory: {gpu_memory_gb:.2f} GB")
        
        # Default configuration
        optimized_config = {
            'device': 'cuda',
            'use_quantization': 'auto',
            'tensor_parallel_size': 1,
            'enable_prefix_caching': False,
            'enforce_eager': True,
            'attention_backend': 'XFORMERS',
            'output_file': 'actions-vllm.txt',
            'save_prompts': False,
        }
        
        # Adjust parameters based on GPU memory
        if gpu_memory_gb >= 32:  # Large memory GPU
            optimized_config.update({
                'gpu_memory_utilization': 0.85,
                'max_model_len': 4096,
                'enable_prefix_caching': True,
                'enforce_eager': False,
            })
        elif gpu_memory_gb >= 24:  # Medium-memory GPU
            optimized_config.update({
                'gpu_memory_utilization': 0.80,
                'max_model_len': 2048,
                'enable_prefix_caching': False,
                'enforce_eager': True,
            })
        elif gpu_memory_gb >= 16:  # Small memory GPU
            optimized_config.update({
                'gpu_memory_utilization': 0.75,
                'max_model_len': 1024,
                'enable_prefix_caching': False,
                'enforce_eager': True,
            })
        else:  # Very small GPU
            optimized_config.update({
                'gpu_memory_utilization': 0.70,
                'max_model_len': 512,
                'enable_prefix_caching': False,
                'enforce_eager': True,
            })
        
        # Apply user-defined parameters
        optimized_config.update(kwargs)
        
        return optimized_config
    
    def _check_gpu_memory(self):
        """Check GPU memory status"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - cached_memory
            
            logger.info(f"GPU memory status: Total {total_memory:.2f}GB, Allocated {allocated_memory:.2f}GB, Cached {cached_memory:.2f}GB, Free {free_memory:.2f}GB")
    
    def _suggest_memory_optimizations(self):
        """Suggest memory optimization strategies"""
        logger.error("Memory optimization suggestions:")
        logger.error("1. Use quantized models (AWQ/GPTQ)")
        logger.error("2. Reduce gpu_memory_utilization (e.g., 0.6)")
        logger.error("3. Reduce max_model_len (e.g., 1024)")
        logger.error("4. Use a smaller model")
        logger.error("5. Check if other processes are using GPU memory")
    
    def cleanup_all_models(self):
        """Clear all models from memory"""
        logger.info("Start cleaning up all models...")
        
        for config_key, model in self._models.items():
            try:
                if hasattr(model, 'shutdown'):
                    model.shutdown()
                logger.info(f"Model cleaned up: {config_key}")
            except Exception as e:
                logger.error(f"Error cleaning up model {config_key}: {e}")
        
        self._models.clear()
        self._model_configs.clear()
        
        # Force clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        logger.info("All models have been cleaned up")
    
    def get_model_count(self) -> int:
        """Get the number of loaded models"""
        return len(self._models)
    
    def list_models(self):
        """List all loaded models"""
        for config_key, config in self._model_configs.items():
            logger.info(f"Model {config_key}: {config['model_path']}")

# Global instance
model_manager = GlobalModelManager()