# Test fine-tuned sLLM with pipeline

from datasets import load_dataset
from utils import (
    make_prompt,
    make_inference_pipeline, 
    make_requests_for_gpt_evaluation
)

model_id = "shangrilar/yi-ko-6b-text2sql"
hf_pipe = make_inference_pipeline(model_id)

# Import Dataset
df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
df = df.to_pandas()
for idx, row in df.iterrows():
	prompt = make_prompt(row['context'], row['question'])
	df.loc[idx, 'prompt'] = prompt

# Inference execution and generate 
gen_sqls = hf_pipe(
	df['prompt'].tolist(), do_sample=False,
	return_full_text=False, max_length=1024, truncation=True
)
gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
df['gen_sql'] = gen_sqls

# Convert Inference results into Jsonl for evaluation
ft_eval_filepath = "text2sql_evaluation_finetuned.jsonl"
make_requests_for_gpt_evaluation(df, ft_eval_filepath)
