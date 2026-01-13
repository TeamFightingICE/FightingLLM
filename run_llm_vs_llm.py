from pyftg.socket.aio.gateway import Gateway
from pyftg.utils.logging import DEBUG, set_logging
import sys
import asyncio
import typer
from typing_extensions import Annotated, Optional
from agent import LLMAgent
from model_manager import model_manager 
from dotenv import load_dotenv
from loguru import logger
import torch
import gc
import os
import atexit

load_dotenv()
app = typer.Typer(pretty_exceptions_enable=False)

def cleanup_on_exit():
    """Cleanup function on program exit"""
    logger.info("Program exiting, cleaning up all model resources...")
    model_manager.cleanup_all_models()

# Register exit cleanup function
atexit.register(cleanup_on_exit)

def clear_gpu_memory():
    """Clean up GPU memory fragmentation"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def check_gpu_memory():
    """Check GPU memory status"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - cached_memory

        logger.info(f"GPU memory: Total {total_memory:.2f}GB, Allocated {allocated_memory:.2f}GB, Free {free_memory:.2f}GB")
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
    logger.info(f"Starting {game_num} games")
    logger.info(f"Model: {llm_model}")
    
    # Check initial GPU memory.
    check_gpu_memory()
    
    # Generate agent name
    if persona == "" or persona is None:
        name = f"LLM_{llm_model}_{LLMAgent.get_agent_name(prompt_generator)}_{shots}_shots"
    else:
        name = f"LLM_{llm_model}_{LLMAgent.get_agent_name(prompt_generator)}_{persona}_{shots}_shots"
    name = name.replace("/", "_").replace(":", "-")
    logger.info(f"Agent name: {name}")
    
    # Create the agent only once — the model will be reused by the global manager.
    logger.info("Initializing the Agent (the model will be loaded only once).")
    try:
        agent1 = LLMAgent(
            llm_model=llm_model,
            shots=shots,
            prompt_generator=prompt_generator,
            persona=persona,
        )
        logger.info("Agent initialization complete")
        logger.info(f"Current model count: {model_manager.get_model_count()}")
        
    except torch.OutOfMemoryError as e:
        logger.error(f"GPU memory insufficient: {e}")
        logger.error("Please try:")
        logger.error("1. Use a smaller model.")
        logger.error("2. Reduce gpu_memory_utilization parameter") 
        logger.error("3. Check if other processes are using GPU")
        raise
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        raise e
    
    # Game loop
    for i in range(game_num):
        logger.info(f"=" * 50)
        logger.info(f"Starting game {i+1}/{game_num}")
        
        if i > 0:
            logger.info("Reusing already loaded model and Agent")
            # Clean up memory fragmentation between matches
            clear_gpu_memory()
        
        gateway = None
        try:
            # Create a new gateway for each game.
            gateway = Gateway(host, port)
            
            # Register AI (reuse the same agent instance, with the model also reused).
            gateway.register_ai(name, agent1)
            
            logger.info(f"Starting game {i+1}...")
            
            # Run the game
            await gateway.run_game(characters, [name, p2], 1)
            
            logger.info(f"Game {i+1} completed.")
            
        except Exception as e:
            logger.error(f"Error in game {i+1}: {e}")
            
        finally:
            # Ensure the gateway is closed after each game
            if gateway:
                try:
                    await gateway.close()
                    logger.debug(f"Gateway for game {i+1} closed.")
                except Exception as e:
                    logger.warning(f"Error closing gateway: {e}")
        
        # Check memory status after each game
        if i < game_num - 1:
            free_memory = check_gpu_memory()
            if free_memory < 1.0:  
                logger.warning("Insufficient GPU memory; cleaning up...")
                clear_gpu_memory()
    
    logger.info(f"=" * 50)
    logger.info(f"All {game_num} games completed!")
    logger.info(f"Model statistics: {model_manager.get_model_count()} model instances loaded")

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
        logger.info("Received interrupt signal, cleaning up resources...")
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        cleanup_on_exit()

if __name__ == "__main__":
    set_logging(log_level=DEBUG)
    app()