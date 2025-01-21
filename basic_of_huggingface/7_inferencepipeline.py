from transformers import pipeline
from datasets import load_dataset, Dataset

# import dataset
klue_ynat_eval = load_dataset('klue', 'ynat', split="validation")

# config model id from hg hub
model_id = "kimsooyoung/roberta-base-klue-ynat-classification" 

# import model fro hg
model_ppline = pipeline("text-classification", model=model_id)

# Prepare sample data
sample_data = klue_ynat_eval["title"][:5]
print(f"Sample Data: {sample_data}")

# Run inference
inf_result = model_ppline(sample_data)
print(f"Inference Result: {inf_result}")