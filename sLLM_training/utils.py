import json
import pandas as pd
from pathlib import Path
import os
from constants import OPENAI_API_KEY

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Function to generate a prompt for SQL generation
def make_prompt(ddl, question, query=''):
    """
    Creates a formatted prompt string for SQL generation.

    Parameters:
    ddl (str): The DDL (Data Definition Language) statements.
    question (str): The natural language question that needs to be converted into SQL.
    query (str, optional): An existing SQL query (default is an empty string).

    Returns:
    str: A formatted prompt string.
    """
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
{ddl}

### Question:
{question}

### SQL:
{query}"""
    return prompt


# Function to generate JSONL formatted evaluation requests for GPT
def make_requests_for_gpt_evaluation(df, filename, dir='requests'):
    """
    Generates evaluation requests in JSONL format for GPT.

    Parameters:
    df (DataFrame): A Pandas DataFrame containing SQL generation results.
    filename (str): The name of the output JSONL file.
    dir (str, optional): Directory to save the JSONL file (default is 'requests').

    Returns:
    None
    """
    # Create the directory if it does not exist
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)

    prompts = []
    # Iterate over each row in the DataFrame to create prompts
    for idx, row in df.iterrows():
        prompts.append("""Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" + f"""

DDL: {row['context']}
Question: {row['question']}
gt_sql: {row['answer']}
gen_sql: {row['gen_sql']}"""
)

    # Construct JSONL format requests for GPT model
    jobs = [{"model": "gpt-4-turbo-preview", "response_format": { "type": "json_object" }, 
             "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]

    # Save the JSONL requests to a file
    with open(Path(dir, filename), "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


# Function to convert JSONL results to CSV
def change_jsonl_to_csv(input_file, output_file, prompt_column="prompt", response_column="response"):
    """
    Converts a JSONL file containing GPT responses into a CSV file.

    Parameters:
    input_file (str): Path to the input JSONL file.
    output_file (str): Path to the output CSV file.
    prompt_column (str, optional): Name of the prompt column in the CSV (default is "prompt").
    response_column (str, optional): Name of the response column in the CSV (default is "response").

    Returns:
    DataFrame: The converted DataFrame.
    """
    prompts = []
    responses = []

    # Read the JSONL file and extract the prompts and responses
    with open(input_file, 'r') as json_file:
        for data in json_file:
            prompts.append(json.loads(data)[0]['messages'][0]['content'])  # Extract prompt
            responses.append(json.loads(data)[1]['choices'][0]['message']['content'])  # Extract response

    # Create a DataFrame and save it as CSV
    df = pd.DataFrame({prompt_column: prompts, response_column: responses})
    df.to_csv(output_file, index=False)
    return df
