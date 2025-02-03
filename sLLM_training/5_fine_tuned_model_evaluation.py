# sql 생성 수행
gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False,
                   return_full_text=False, max_length=1024, truncation=True)
gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
df['gen_sql'] = gen_sqls

# 평가를 위한 requests.jsonl 생성
ft_eval_filepath = "text2sql_evaluation_finetuned.jsonl"
make_requests_for_gpt_evaluation(df, ft_eval_filepath)

# GPT-4 평가 수행
!python api_request_parallel_processor.py \
  --requests_filepath requests/{ft_eval_filepath} \
  --save_filepath results/{ft_eval_filepath} \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 2500 \
  --max_tokens_per_minute 100000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20

ft_eval = change_jsonl_to_csv(f"results/{ft_eval_filepath}", "results/yi_ko_6b_eval.csv", "prompt", "resolve_yn")
ft_eval['resolve_yn'] = ft_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
num_correct_answers = ft_eval.query("resolve_yn == 'yes'").shape[0]
num_correct_answers