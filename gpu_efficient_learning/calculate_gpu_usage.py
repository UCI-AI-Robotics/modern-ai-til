import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# print gpu usage utilizing torch's API
def print_gpu_usage():
    # if device is GPU
    if torch.cuda.is_available():
        # get allocated memory and print result
        used_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Used GPU Memory: {used_memory}")
    else:
        print("Current device is CPU")

print_gpu_usage()

# import model and tokenizer
def load_model_and_tokenizer(model_id, peft=None):
    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # define model\
    model = None
    if peft is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"":0})

    # print gpu usage with helper func
    print_gpu_usage()

    # return model & tokenizer
    return model, tokenizer

model_id = "EleutherAI/polyglot-ko-1.3b"
model, tokenizer = load_model_and_tokenizer(model_id) # GPU 메모리 사용량: 2.599 GB
print(f"model.dtype: {model.dtype}")