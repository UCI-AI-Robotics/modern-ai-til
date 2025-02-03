import torch
from datasets import load_dataset
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM
)

# Inference pipeline with HG transformer modules
def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", 
        # quantization options
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16
    )
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer
    )
    return pipe

model_id = "shangrilar/yi-ko-6b-text2sql"
hf_pipe = make_inference_pipeline(model_id)

hf_pipe(
    example, do_sample=False,
    return_full_text=False, 
    max_length=1024, 
    truncation=True
)
# SELECT COUNT(*) FROM players WHERE username LIKE '%admin%';