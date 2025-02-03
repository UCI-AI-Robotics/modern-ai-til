import os

base_model = "beomi/Yi-Ko-6B"
finetuned_model = "yi-ko-6b-text2sql"
data-path = "data/"

command = f"""
autotrain llm \
--train \
--model {base_model} \
--project-name {finetuned_model} \
--data-path {data-path} \
--text-column text \
--lr 2e-4 \
--batch-size 8 \
--epochs 1 \
--block-size 1024 \
--warmup-ratio 0.1 \
--lora-r 16 \
--lora-alpha 32 \
--lora-dropout 0.05 \
--weight-decay 0.01 \
--gradient-accumulation 8 \
--mixed-precision fp16 \
--use-peft \
--quantization int4 \
--trainer sft
"""

os.system(command)
