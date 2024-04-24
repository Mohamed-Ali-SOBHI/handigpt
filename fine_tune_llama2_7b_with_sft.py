import gc
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training

# Chargement du modèle de base et configuration initiale
base_model = "NousResearch/Llama-2-7b-chat-hf"
new_model = "NousResearch/Llama-2-7b-chat-hf-handigpt"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# Configuration de BitsAndBytes pour la quantification
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Configuration de LoRA pour l'ajustement des paramètres
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Chargement du modèle avec la configuration de quantification
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# Charger les données
file_path = 'generated_questions.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Flatten the list of lists structure into a single list of dictionaries
flat_data = [item for sublist in data for item in sublist]

# Verify that each dictionary contains both 'instruction' and 'output' keys
if not all('instruction' in entry and 'output' in entry for entry in flat_data):
    raise ValueError("Some entries are missing 'instruction' or 'output' keys.")

# Create pairs of instruction and output
# Assuming tokenizer and tokenizer.eos_token are defined as part of your environment
texts = [entry['instruction'] + tokenizer.eos_token + entry['output'] for entry in flat_data]

# Tokenization of the pairs for the model
# Ensure the tokenizer object is properly initialized and configured as per your model's requirements
encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512)
dataset = Dataset.from_dict(encoded_inputs)

# Configuration des arguments de training
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

# Configuration du Trainer pour le fine-tuning
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(1)),  # Sélectionner une partie pour l'évaluation
    tokenizer=tokenizer
)

# Lancement du fine-tuning
trainer.train()

# Nettoyage et sauvegarde du modèle
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

# Recharger le tokenizer pour la sauvegarde
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token

# Sauvegarde du modèle et du tokenizer
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)
