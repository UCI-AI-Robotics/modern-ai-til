from utils import (
    make_prompt,
    change_jsonl_to_csv
)
import json

eval_filepath = "text2sql_evaluation.jsonl"

base_eval = change_jsonl_to_csv(
    f"results/{eval_filepath}", 
    "results/yi_ko_6b_eval.csv", 
    "prompt", "resolve_yn"
)
base_eval['resolve_yn'] = base_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
num_correct_answers = base_eval.query("resolve_yn == 'yes'").shape[0]

print(f"num_correct_answers: {num_correct_answers}")