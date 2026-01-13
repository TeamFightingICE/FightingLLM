## This is the repository of game playing LLM on DareFightingICE platform.

### Installation
- Python: 3.12
- vllm
- torch 2.7.0
- torchvision 0.22.0
- torchaudio 2.7.0
- transformers 4.53.2
- perft 0.16.0
- scikit-learn
- loguru

Note: Please choose the appropriate cuda version for your system
### How to run
1. Download the finetuned model from this [link](https://huggingface.co/anonymous2120/llm-fighting).
2. Start DareFightingICE
3. Run the script 
```ssh 
python run_llm_vs_mcts.py --llm_model local:path_to_local_model
```

### Finetune: Please check [this file](finetune/finetuning.py)