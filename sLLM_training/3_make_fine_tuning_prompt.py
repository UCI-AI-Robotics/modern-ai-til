from datasets import load_dataset
from utils import (
    make_prompt
)

# Load private dataset and then convert into csv
df_sql = load_dataset("shangrilar/ko_text2sql", "origin")["train"]
df_sql = df_sql.to_pandas()
df_sql = df_sql.dropna().sample(frac=1, random_state=42)
df_sql = df_sql.query("db_id != 1")

for idx, row in df_sql.iterrows():
  df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'])

df_sql.to_csv('data/train.csv', index=False)
