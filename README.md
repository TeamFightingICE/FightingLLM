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
- datasets

Note: Please choose the appropriate cuda version for your system
### How to run
1. Download the finetuned model from this [link](https://huggingface.co/anonymous2120/llm-fighting-rushdown) for rushdown fighting style and this [link](https://huggingface.co/anonymous2120/llm-fighting-zoning) for zoning fighting style, both of them can be downloaded via huggingface-cli.
2. Start DareFightingICE
3. Change the prompt defined in [prompt.py](ai/prompt.py) file in line number 90.
4. Run the script 
```ssh 
python run_llm_vs_mcts.py --llm_model local:path_to_local_model
```

### Finetuning

Please refer to the file `finetune/finetuning.py`. You are free to modify the source code as needed to finetune the LLM.

1. Prepare a dataset with the same structure as this example:  
   [Simonkami/MctsAi_Zoning](https://huggingface.co/datasets/Simonkami/MctsAi_Zoning/viewer/default/train).  
   The dataset can either be uploaded to Hugging Face or stored locally.

2. In `finetune/finetuning.py`, update:
   - **Line 201** to point to your prepared dataset.
   - **Line 71** to specify the base model you want to finetune.

3. Start finetuning by running:
   ```bash
   python finetune/finetuning.py
