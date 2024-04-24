import gc
import os
import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

base_model = "NousResearch/Llama-2-7b-chat-hf"
new_model = "NousResearch/Llama-2-7b-chat-hf-handigpt"

"""## Fine-tuning Mistral-8x7b"""

# Insert your dataset here
dataset = None

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

training_arguments = TrainingArguments(
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    num_train_epochs=10,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    optim="paged_adamw_8bit",
    warmup_steps=10,
    output_dir="./results",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(100)),
    peft_config=peft_config,
    dataset_text_field="instruction",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

"""Merging the base model with the trained adapter."""
# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token

# Save model
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

"""## Going further

* **DPO fine-tuning**: see [this article](https://mlabonne.github.io/blog/posts/Fine_tune_Mistral_7b_with_DPO.html)
* **Better fine-tuning tool**: see [Axolotl](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html)
* **Evaluation**: see the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) and the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
* **Quantization**: see [naive quantization](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html), [GPTQ](https://mlabonne.github.io/blog/posts/4_bit_Quantization_with_GPTQ.html), [GGUF/llama.cpp](https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html), ExLlamav2, and AWQ.
"""