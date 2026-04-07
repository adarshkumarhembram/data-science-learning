import pandas as pd
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# Load model
tokenizer = MT5Tokenizer.from_pretrained("models/tamil_model")
model = MT5ForConditionalGeneration.from_pretrained("models/tamil_model")

# Load test data
test_df = pd.read_csv("data/tamil_ilsum_2024_test_without_summary.csv")

summaries = []

for article in test_df["article"]:
    
    inputs = tokenizer("summarize: " + article, return_tensors="pt", truncation=True)
    
    summary_ids = model.generate(inputs["input_ids"], max_length=80)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    summaries.append(summary)

test_df["generated_summary"] = summaries

test_df.to_csv("test_predictions.csv", index=False)

print("Summaries generated!")