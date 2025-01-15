from transformers import AutoModel, AutoTokenizer

# sample input 
text = "What is huggingface transformer?"

# Automodel
bert_model = AutoModel.from_pretrained("bert-base-uncased")
# AutoTokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# prepare tokenized input
tokenized_text = bert_tokenizer(text, return_tensors="pt")
# Inference
bert_output = bert_model(**tokenized_text)

# Exactly same thing with GPT-2
gpt2_model = AutoModel.from_pretrained("gpt2")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenized_text = gpt2_tokenizer(text, return_tensors="pt")
gpt2_output = gpt2_model(**tokenized_text)

