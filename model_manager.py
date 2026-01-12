# model_manager.py - 创建这个新文件
import threading
import torch
import gc
from typing import Dict, Optional, Any
from loguru import logger

class GlobalModelManager:
    """全局模型管理器 - 确保模型只加载一次"""
    
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
            logger.info("全局模型管理器已初始化")
    
    def get_model(self, model_path: str, **kwargs):
        """获取模型实例，如果不存在则创建"""
        
        # 创建配置的唯一标识
        config_key = self._create_config_key(model_path, **kwargs)
        
        if config_key not in self._models:
            logger.info(f"首次加载模型: {model_path}")
            logger.info(f"配置: {kwargs}")
            
            # 检查GPU内存
            self._check_gpu_memory()
            
            # 导入LocalLLaMA
            from llm_local import LocalLLaMA
            
            # 设置内存优化参数
            optimized_kwargs = self._optimize_memory_config(**kwargs)
            
            try:
                # 创建模型实例
                model = LocalLLaMA(
                    model_path=model_path,
                    **optimized_kwargs
                )
                
                self._models[config_key] = model
                self._model_configs[config_key] = {
                    'model_path': model_path,
                    'config': optimized_kwargs
                }
                
                logger.info(f"模型加载完成: {config_key}")
                self._check_gpu_memory()
                
            except torch.OutOfMemoryError as e:
                logger.error(f"GPU内存不足: {e}")
                self._suggest_memory_optimizations()
                raise
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise
        else:
            logger.info(f"复用已加载的模型: {config_key}")
        
        return self._models[config_key]
    
    def _create_config_key(self, model_path: str, **kwargs) -> str:
        """创建配置的唯一标识"""
        # 只包含影响模型加载的关键参数
        key_params = {
            'model_path': model_path,
            'tensor_parallel_size': kwargs.get('tensor_parallel_size', 1),
            'gpu_memory_utilization': kwargs.get('gpu_memory_utilization', 0.9),
            'use_quantization': kwargs.get('use_quantization', 'auto'),
            'max_model_len': kwargs.get('max_model_len', None),
        }
        return str(hash(str(sorted(key_params.items()))))
    
    def _optimize_memory_config(self, **kwargs) -> dict:
        """根据GPU内存优化配置"""
        
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        logger.info(f"GPU总内存: {gpu_memory_gb:.2f} GB")
        
        # 默认配置
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
        
        # 根据GPU内存调整参数
        if gpu_memory_gb >= 32:  # 大内存GPU
            optimized_config.update({
                'gpu_memory_utilization': 0.85,
                'max_model_len': 4096,
                'enable_prefix_caching': True,
                'enforce_eager': False,
            })
        elif gpu_memory_gb >= 24:  # 中等内存GPU
            optimized_config.update({
                'gpu_memory_utilization': 0.80,
                'max_model_len': 2048,
                'enable_prefix_caching': False,
                'enforce_eager': True,
            })
        elif gpu_memory_gb >= 16:  # 较小内存GPU
            optimized_config.update({
                'gpu_memory_utilization': 0.75,
                'max_model_len': 1024,
                'enable_prefix_caching': False,
                'enforce_eager': True,
            })
        else:  # 很小的GPU
            optimized_config.update({
                'gpu_memory_utilization': 0.70,
                'max_model_len': 512,
                'enable_prefix_caching': False,
                'enforce_eager': True,
            })
        
        # 应用用户自定义参数
        optimized_config.update(kwargs)
        
        return optimized_config
    
    def _check_gpu_memory(self):
        """检查GPU内存状态"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - cached_memory
            
            logger.info(f"GPU内存状态: 总计{total_memory:.2f}GB, 已分配{allocated_memory:.2f}GB, 缓存{cached_memory:.2f}GB, 可用{free_memory:.2f}GB")
    
    def _suggest_memory_optimizations(self):
        """建议内存优化方案"""
        logger.error("内存优化建议:")
        logger.error("1. 使用量化模型 (AWQ/GPTQ)")
        logger.error("2. 降低 gpu_memory_utilization (例如: 0.6)")
        logger.error("3. 减小 max_model_len (例如: 1024)")
        logger.error("4. 使用更小的模型")
        logger.error("5. 检查其他进程是否占用GPU内存")
    
    def cleanup_all_models(self):
        """清理所有模型"""
        logger.info("开始清理所有模型...")
        
        for config_key, model in self._models.items():
            try:
                if hasattr(model, 'shutdown'):
                    model.shutdown()
                logger.info(f"模型已清理: {config_key}")
            except Exception as e:
                logger.error(f"清理模型时出错 {config_key}: {e}")
        
        self._models.clear()
        self._model_configs.clear()
        
        # 强制清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        logger.info("所有模型已清理完成")
    
    def get_model_count(self) -> int:
        """获取已加载的模型数量"""
        return len(self._models)
    
    def list_models(self):
        """列出所有已加载的模型"""
        for config_key, config in self._model_configs.items():
            logger.info(f"模型 {config_key}: {config['model_path']}")

# 全局实例
model_manager = GlobalModelManager()