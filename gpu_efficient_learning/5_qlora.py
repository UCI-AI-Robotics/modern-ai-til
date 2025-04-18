from utils import (
    load_model_and_tokenizer,
    make_dummy_dataset,
    print_gpu_usage,
    train_model,
    cleanup,
)

import gc
import torch
from transformers import TrainingArguments

# Helper function for GPU usage experiments
def gpu_memory_experiment(
        batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        model_id="EleutherAI/polyglot-ko-1.3b",
        peft=None
    ):
    print(f"batch_size: {batch_size}")
    
    # prepare model/tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, peft=peft)
    
    # Gradient Checkpoint option setup
    if gradient_checkpointing == True or peft == 'qlora':
        model.config.use_cache = False

    # prepare dummy dataset
    dataset = make_dummy_dataset()

    # prepare training argument
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        output_dir="./result",
        num_train_epochs=1
      )

    # model train start with model/dataset/training_args 
    try:
        train_model(model, dataset, training_args)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(e)
        else:
            raise e
    finally:
        del model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_usage()

cleanup()
print_gpu_usage()

gpu_memory_experiment(
    batch_size=4,
    gradient_accumulation_steps=4,
    # gradient_checkpointing=True,
    peft='qlora'
)
# batch_size: 4
# trainable params: 1,572,864 || all params: 1,333,383,168 || trainable%: 0.11796039111242178
# train_model Used GPU Memory : 1.3078560829162598
# Optimizer Memory Usage: 0.012 GB
# Gradient Memory Usage: 0.006 GB
#  Used GPU Memory : 0.015870094299316406

# gpu_memory_experiment(
#     batch_size=16,
#     peft='qlora'
# )
# batch_size: 16
# trainable params: 1,572,864 || all params: 1,333,383,168 || trainable%: 0.11796039111242178
# load_model_and_tokenizer Used GPU Memory : 1.1434111595153809
# train_model Used GPU Memory : 1.6985268592834473
# Optimizer Memory Usage: 0.012 GB
# Gradient Memory Usage: 0.006 GB
#  Used GPU Memory : 0.015870094299316406

# => optimizer memory twice than normal LoRA
# => model/tokenizer memory about half than normal LoRA
# => total mem usage 4.3 => 1.7