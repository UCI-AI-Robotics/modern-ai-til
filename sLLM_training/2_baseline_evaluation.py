import os
from constants import OPENAI_API_KEY

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

eval_filepath = "text2sql_evaluation.jsonl"  # Replace with the actual evaluation file name

command = f"""
python api_request_parallel_processor.py \
--requests_filepath requests/{eval_filepath}  \
--save_filepath results/{eval_filepath} \
--request_url https://api.openai.com/v1/chat/completions \
--max_requests_per_minute 2500 \
--max_tokens_per_minute 100000 \
--token_encoding_name cl100k_base \
--max_attempts 5 \
--logging_level 20
"""

os.system(command)

# import openai

# openai.api_key = OPENAI_API_KEY

# # Fetch available models using the new API method
# models = openai.models.list()

# # Print available model IDs
# print([model.id for model in models.data])
