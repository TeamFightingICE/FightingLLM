# run.py - 最终优化版本
from pyftg.socket.aio.gateway import Gateway
from pyftg.utils.logging import DEBUG, set_logging
import sys
import asyncio
import typer
from typing_extensions import Annotated, Optional
from agent import LLMAgent
from model_manager import model_manager  # 导入全局模型管理器
from dotenv import load_dotenv
from loguru import logger
import torch
import gc
import os
import atexit

load_dotenv()
app = typer.Typer(pretty_exceptions_enable=False)

def cleanup_on_exit():
    """程序退出时的清理函数"""
    logger.info("程序退出，正在清理所有模型资源...")
    model_manager.cleanup_all_models()

# 注册退出清理函数
atexit.register(cleanup_on_exit)

def clear_gpu_memory():
    """清理GPU内存碎片"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def check_gpu_memory():
    """检查GPU内存状态"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - cached_memory
        
        logger.info(f"GPU内存: 总计{total_memory:.2f}GB, 已分配{allocated_memory:.2f}GB, 可用{free_memory:.2f}GB")
        return free_memory
    return 0

async def start_process(
    host: str,
    port: int,
    characters: list[str] = ["ZEN", "ZEN"],
    game_num: int = 1,
    llm_model: str = "",
    shots: int = 0,
    prompt_generator: int = 1,
    persona: str = "",
):
    logger.info(f"开始运行 {game_num} 场游戏")
    logger.info(f"模型: {llm_model}")
    
    # 检查初始GPU内存
    check_gpu_memory()
    
    # 生成agent名称
    if persona == "" or persona is None:
        name = f"LLM_{llm_model}_{LLMAgent.get_agent_name(prompt_generator)}_{shots}_shots"
    else:
        name = f"LLM_{llm_model}_{LLMAgent.get_agent_name(prompt_generator)}_{persona}_{shots}_shots"
    name = name.replace("/", "_").replace(":", "-")
    logger.info(f"Agent name: {name}")
    
    # 只创建一次agent - 模型会被全局管理器复用
    logger.info("正在初始化Agent（模型只会加载一次）...")
    try:
        agent1 = LLMAgent(
            llm_model=llm_model,
            shots=shots,
            prompt_generator=prompt_generator,
            persona=persona,
        )
        logger.info("Agent初始化完成")
        logger.info(f"当前已加载模型数量: {model_manager.get_model_count()}")
        
    except torch.OutOfMemoryError as e:
        logger.error(f"GPU内存不足: {e}")
        logger.error("请尝试:")
        logger.error("1. 使用更小的模型")
        logger.error("2. 降低gpu_memory_utilization参数") 
        logger.error("3. 检查是否有其他进程占用GPU")
        raise
    except Exception as e:
        logger.error(f"Agent初始化失败: {e}")
        raise
    
    # 游戏循环
    for i in range(game_num):
        logger.info(f"=" * 50)
        logger.info(f"开始第 {i+1}/{game_num} 场游戏")
        
        if i > 0:
            logger.info("复用已加载的模型和Agent")
            # 游戏间清理内存碎片
            clear_gpu_memory()
        
        gateway = None
        try:
            # 每场游戏创建新的gateway
            gateway = Gateway(host, port)
            
            # 注册AI（复用同一个agent实例，其中的模型也是复用的）
            gateway.register_ai(name, agent1)
            
            logger.info(f"开始运行第 {i+1} 场游戏...")
            
            # 运行游戏
            await gateway.run_game(characters, [name, p2], 1)
            
            logger.info(f"第 {i+1} 场游戏完成")
            
        except Exception as e:
            logger.error(f"第 {i+1} 场游戏出错: {e}")
            
        finally:
            # 确保每场游戏后都关闭gateway
            if gateway:
                try:
                    await gateway.close()
                    logger.debug(f"第 {i+1} 场游戏的gateway已关闭")
                except Exception as e:
                    logger.warning(f"关闭gateway时出错: {e}")
        
        # 每场游戏后检查内存状态
        if i < game_num - 1:
            free_memory = check_gpu_memory()
            if free_memory < 1.0:  # 如果可用内存少于1GB
                logger.warning("GPU内存不足，正在清理...")
                clear_gpu_memory()
    
    logger.info(f"=" * 50)
    logger.info(f"所有 {game_num} 场游戏已完成！")
    logger.info(f"模型统计: 共加载 {model_manager.get_model_count()} 个模型实例")

@app.command()
def main(
    host: Annotated[
        Optional[str], typer.Option(help="Host used by DareFightingICE")
    ] = "127.0.0.1",
    port: Annotated[
        Optional[int], typer.Option(help="Port used by DareFightingICE")
    ] = 31415,
    llm_model: Annotated[
        Optional[str], typer.Option("--llm_model", help="LLM model to use")
    ] = "ollama:mistral-nemo:12b",
    shots: Annotated[
        Optional[int],
        typer.Option("--few_shots", help="The number of shots for prompting"),
    ] = 0,
    prompt_generator: Annotated[
        Optional[int],
        typer.Option("--prompt_type", help="Prompt generator type"),
    ] = 1,
    game_num: Annotated[
        Optional[int],
        typer.Option("--game_num", help="The number of games to run"),
    ] = 1,
    persona: Annotated[
        Optional[str], typer.Option("--persona", help="Persona of agent")
    ] = "",
    p2: Annotated[
        Optional[str], typer.Option("--p2", help="The opponent")
    ] = "MctsAi23i",
):
    # 设置环境变量优化内存管理
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    try:
        asyncio.run(
            start_process(
                host=host,
                port=port,
                characters=["ZEN", "ZEN"],
                llm_model=llm_model,
                shots=shots,
                prompt_generator=prompt_generator,
                game_num=game_num,
                persona=persona,
                p2=p2,
            )
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在清理资源...")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # 程序结束时清理所有资源
        cleanup_on_exit()

if __name__ == "__main__":
    set_logging(log_level=DEBUG)
    app()