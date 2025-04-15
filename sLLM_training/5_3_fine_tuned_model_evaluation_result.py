# Checkout Answer accuracy

import json
from utils import (
    change_jsonl_to_csv
)

ft_eval_filepath = "text2sql_evaluation_finetuned.jsonl"

ft_eval = change_jsonl_to_csv(
	f"results/{ft_eval_filepath}", 
	"results/yi_ko_6b_eval.csv", 
	"prompt", "resolve_yn"
)
ft_eval['resolve_yn'] = ft_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
num_correct_answers = ft_eval.query("resolve_yn == 'yes'").shape[0]

print(f"num_correct_answers: {num_correct_answers}")