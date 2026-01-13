import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback  # Added Early Stopping Callback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

print(torch.cuda.device_count())     # Output 2 indicates two cards are detected
print(torch.cuda.get_device_name(0))

COMMANDABLE_ACTIONS = [
    "FORWARD_WALK", "DASH", "BACK_STEP", "CROUCH", "JUMP", "FOR_JUMP",
    "BACK_JUMP", "STAND_GUARD", "CROUCH_GUARD", "AIR_GUARD", "STAND_A", "STAND_B", "THROW_A",
    "THROW_B", "CROUCH_A", "CROUCH_B", "STAND_FA", "STAND_FB", "CROUCH_FA",
    "CROUCH_FB", "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_D_DB_BA", "STAND_D_DB_BB",
    "STAND_D_DF_FA", "STAND_D_DF_FB", "STAND_D_DF_FC", "AIR_A",
    "AIR_B", "AIR_DA", "AIR_DB", "AIR_FA",
    "AIR_FB", "AIR_UA", "AIR_UB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB",
    "AIR_D_DF_FA", "AIR_D_DF_FB",
    # Add attribute vocabulary
]

# 1. Configure tokenizer and add special tokens first
print("Loading and configuring tokenizer...")
# meta-llama/Llama-3.2-3B-Instruct
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Add special tokens
special_tokens_dict = {'additional_special_tokens': COMMANDABLE_ACTIONS}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Successfully added {num_added_tokens} action tokens. Total vocab size: {len(tokenizer)}")

# Verify if tokens were added correctly
print("\nVerifying action token encoding:")
all_tokens_valid = True
for act in COMMANDABLE_ACTIONS:
    token_ids = tokenizer.encode(act, add_special_tokens=False)
    if len(token_ids) == 1:
        print(f"✓ '{act}' -> token_id: {token_ids[0]}")
    else:
        print(f"✗ '{act}' -> Multiple tokens: {token_ids}")
        all_tokens_valid = False

if not all_tokens_valid:
    raise ValueError("Some action tokens were not correctly added as single tokens. Please check configuration.")

# 2. Quantization configuration (Modified: Reduce quantization to avoid gradient issues)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True
)

# 4. Prepare model for k-bit training (Important!)
print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# 5. Resize embedding layer
print("Resizing token embedding layer...")
original_embeddings = model.get_input_embeddings().weight.data
original_vocab_size = original_embeddings.shape[0]
print(f"Original vocab size: {original_vocab_size}, New vocab size: {len(tokenizer)}")

# Resize embedding size
model.resize_token_embeddings(len(tokenizer))

# Important: Ensure new embedding layer can calculate gradients
new_embeddings = model.get_input_embeddings()
if hasattr(new_embeddings, 'weight'):
    new_embeddings.weight.requires_grad_(True)
    
# Initialize new token embeddings (using the mean of original embeddings)
if len(tokenizer) > original_vocab_size:
    with torch.no_grad():
        new_token_embeddings = model.get_input_embeddings().weight[original_vocab_size:]
        new_token_embeddings.data.normal_(mean=0.0, std=0.02)
        print(f"Initialized embeddings for {len(tokenizer) - original_vocab_size} new tokens")

# 6. LoRA Configuration (Modified: Added target_modules including embedding)
lora_config = LoraConfig(
    r=16,  # Increase r value to improve expressiveness
    lora_alpha=32,  # Correspondingly increase alpha
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens"  # Important: Include embedding layer
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)

# Apply LoRA
print("Applying LoRA configuration...")
model = get_peft_model(model, lora_config)

# 7. Enable gradient checkpointing and input gradients
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Print trainable parameter information
model.print_trainable_parameters()

# 8. Verify gradient setup
def verify_gradient_setup(model):
    print("\n=== Gradient Setup Verification ===")
    trainable_params = []
    non_trainable_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            non_trainable_params.append(name)
    
    print(f"Number of trainable parameters: {len(trainable_params)}")
    print("First 5 trainable parameters:")
    for name in trainable_params[:5]:
        print(f"  ✓ {name}")
    
    # Check embedding layer
    embed_params = [name for name in trainable_params if 'embed' in name.lower()]
    if embed_params:
        print(f"Embedding-related trainable parameters: {embed_params}")
    else:
        print("⚠️ No embedding-related trainable parameters found")
    
    return len(trainable_params) > 0

if not verify_gradient_setup(model):
    raise ValueError("No trainable parameters found. Please check LoRA configuration.")

class GameAIDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=2048):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.prompt_template = (
            "Instruction:{instruction}\n"
            "Input:{input}\n"
            "Output:{output}{eos_token}"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        full_text = self.prompt_template.format(
            instruction=item['instruction'],
            input=item['input'],
            output=item['output'],
            eos_token=self.tokenizer.eos_token
        )

        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze().clone()
        }

# 9. Prepare dataset
print("Loading dataset...")

dataset = load_dataset("Simonkami/MctsAi_Zoning")
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = GameAIDataset(split_dataset["train"], tokenizer)
test_dataset = GameAIDataset(split_dataset["test"], tokenizer)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# 10. Training arguments configuration (Modified: More conservative settings + Early Stopping configuration)
training_args = TrainingArguments(
    output_dir="Models_QLoRA",
    eval_strategy="steps",
    eval_steps=25,  # Evaluate every 25 steps
    learning_rate=1e-4,  # Lower learning rate
    warmup_ratio=0.03,   
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=16,  # Increase accumulation steps
    per_device_eval_batch_size=4,
    num_train_epochs=5, 
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=3,
    save_steps=500,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    report_to=[],
    optim="paged_adamw_32bit",
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Early stopping related configuration
    save_strategy="steps",  # Ensure consistency with eval_strategy
    logging_strategy="steps",
)

# 11. Configure Early Stopping Callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,      # Stop if no improvement after 10 consecutive evaluations
    early_stopping_threshold=0.001   # Minimum threshold for improvement (optional)
)

print(f"✓ Early stopping mechanism configured:")
print(f"  - Patience steps: {early_stopping_callback.early_stopping_patience}")
print(f"  - Improvement threshold: {early_stopping_callback.early_stopping_threshold}")
print(f"  - Monitored metric: eval_loss (lower is better)")

# 12. Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# 13. Initialize trainer (Add Early Stopping Callback)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    callbacks=[early_stopping_callback]  # Add early stopping callback
)

# 14. Final Verification
def final_verification():
    print("\n=== Final Pre-training Verification ===")
    
    # Test a single batch
    try:
        sample_batch = next(iter(trainer.get_train_dataloader()))
        print("✓ Data loader normal")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in sample_batch.items()})
            print(f"✓ Forward pass normal, loss: {outputs.loss.item():.4f}")
        
        model.train()
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

if not final_verification():
    print("❌ Verification failed. Please check configuration.")
    exit(1)

# 15. Start Training
print("\nStarting training...")
print("Note: Training will stop automatically if validation loss does not improve for consecutive evaluations (patience set).")
try:
    # Record training start time
    import time
    start_time = time.time()
    
    # Start training (Early stopping mechanism will be applied automatically)
    train_result = trainer.train()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTraining completed! Total time: {training_time/3600:.2f} hours")
    
    # Check if ended due to early stopping
    if hasattr(trainer.state, 'log_history'):
        final_logs = trainer.state.log_history[-1] if trainer.state.log_history else {}
        if 'early_stopping' in str(final_logs):
            print("🛑 Training ended early due to early stopping mechanism")
        else:
            print("✅ Training completed all epochs normally")
    
    # Save options
    base_save_path = "fightingice-llm-ai/Llama3.2/3b_MctsZoning_5epoch"
    
    # Save LoRA adapter
    lora_save_path = f"{base_save_path}_lora"
    print(f"\nSaving LoRA adapter to: {lora_save_path}")
    trainer.save_model(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    
    # Save training info
    training_info = {
        "training_time_hours": training_time/3600,
        "final_eval_loss": trainer.state.log_history[-1].get('eval_loss', 'N/A') if trainer.state.log_history else 'N/A',
        "total_steps": trainer.state.global_step,
        "early_stopping_triggered": 'early_stopping' in str(trainer.state.log_history[-1]) if trainer.state.log_history else False
    }
    
    import json
    with open(f"{lora_save_path}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("✅ LoRA adapter saved successfully")
    print(f"📊 Final validation loss: {training_info['final_eval_loss']}")
    print(f"📈 Total training steps: {training_info['total_steps']}")
    
    """
    # Attempt to save merged model
    try:
        merged_save_path = f"{base_save_path}_merged"
        print(f"\nSaving merged complete model to: {merged_save_path}")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_save_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_save_path)
        print("✅ Merged model saved successfully")
        del merged_model
        torch.cuda.empty_cache()
    except Exception as merge_error:
        print(f"⚠️ Failed to save merged model: {merge_error}")
        print("LoRA adapter has been saved successfully and can be used normally")
    """
    print("Training finished!")
    
except Exception as e:
    print(f"\nError occurred during training: {e}")
    import traceback
    traceback.print_exc()
    
    # Emergency save
    emergency_path = "fightingice-llm-ai/Llama3.2/emergency_save_lora"
    print(f"Performing emergency save to: {emergency_path}")
    try:
        trainer.save_model(emergency_path)
        tokenizer.save_pretrained(emergency_path)
        print("Emergency save completed")
    except Exception as save_error:
        print(f"Emergency save also failed: {save_error}")

print("\nScript execution finished")