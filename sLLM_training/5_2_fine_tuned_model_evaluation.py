# Inference evaluation with GPT-API

import os
from constants import OPENAI_API_KEY

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define the evaluation file path
ft_eval_filepath = "text2sql_evaluation_finetuned.jsonl"  # Replace with the actual evaluation file name

# Construct the command as a multi-line string
command = f"""
python api_request_parallel_processor.py \
--requests_filepath requests/{ft_eval_filepath}  \
--save_filepath results/{ft_eval_filepath} \
--request_url https://api.openai.com/v1/chat/completions \
--max_requests_per_minute 2500 \
--max_tokens_per_minute 100000 \
--token_encoding_name cl100k_base \
--max_attempts 5 \
--logging_level 20
"""

# Execute the command
os.system(command)
