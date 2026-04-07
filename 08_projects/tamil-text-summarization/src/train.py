import pandas as pd
from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments

# Load dataset
train_df = pd.read_csv("data/tamil_ilsum_2024_train.csv")
val_df = pd.read_csv("data/tamil_ilsum_2024_val.csv")

# Convert to dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# Preprocessing
def preprocess(example):

    input_text = "summarize: " + example["article"]
    target_text = example["summary"]

    inputs = tokenizer(input_text, max_length=512, truncation=True)
    labels = tokenizer(target_text, max_length=128, truncation=True)

    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)

# Training
training_args = TrainingArguments(
    output_dir="models",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

model.save_pretrained("models/tamil_model")
tokenizer.save_pretrained("models/tamil_model")