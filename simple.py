from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
dataset = dataset.select(range(1000))

# Training configuration
training_args = DPOConfig(
    output_dir="SmolLM-DPO",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,
    save_steps=1000,
    logging_steps=500,
    num_train_epochs=3
)

# Initialize trainer
trainer = DPOTrainer(model=model, args=training_args, train_dataset=dataset, processing_class=tokenizer)

# Train model
trainer.train()
