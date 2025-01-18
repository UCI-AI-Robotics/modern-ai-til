from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer

# import tokenizer
model_id = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# sentance to token
token = tokenizer("토크나이저는 텍스트를 토큰 단위로 나눈다")
print(f"token: {token}")

# input_ids to token
print(f"input_ids: {tokenizer.convert_ids_to_tokens(token['input_ids'])}")

# input_ids decode
print(f"Decoded result: {tokenizer.decode(token['input_ids'])}")

# input_ids decode without special chars
print(f"Decoded result w/o special chars: {tokenizer.decode(token['input_ids'], skip_special_tokens=True)}")

# tokenizer with multiple sentences
multi_token_1 = tokenizer(["첫 번째 문장", "두 번째 문장"])
multi_token_2 = tokenizer([["첫 번째 문장", "두 번째 문장"]])
print(f"multi_token_1: {multi_token_1}")
print(f"multi_token_2: {multi_token_2}")

# tokenizer batch decode 
multi_token_1_decoded = tokenizer.batch_decode(multi_token_1['input_ids'])
multi_token_2_decoded = tokenizer.batch_decode(multi_token_2['input_ids'])
print(f"multi_token_1_decoded: {multi_token_1_decoded}")
print(f"multi_token_2_decoded: {multi_token_2_decoded}")

# tokenizer from various models klue/roberta-base, klue/bert-base, roberta-base
klue_roberta_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
klue_bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# bert model seperate sentence division 
sentence = [["첫 번째 문장", "두 번째 문장"]]
sentence_en = [["first sentence", "second sentence"]]
print(f"Result from klue/roberta-base: {klue_roberta_tokenizer(sentence)}")
print(f"Result from klue/bert-base: {klue_bert_tokenizer(sentence)}")
print(f"Result from roberta-base: {roberta_tokenizer(sentence_en)}")

# attention mask
sentence = [["첫 번째 문장은 두 번째 문장보다 길다", "두 번째 문장"]]
print(f"Attention Mask from klue/roberta-base: {klue_bert_tokenizer(sentence, padding='longest')}")
