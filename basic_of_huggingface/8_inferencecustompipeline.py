import torch
from torch.nn.functional import softmax
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define custom Pipeline Class
class CustomPipeline(object):
    def __init__(self, model_id):
        # create tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        # import model
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        # model into inference mode (eval mode)
        self._model.eval()

    def __call__(self, input_data):
        # Tokenize input with padding and truncation for batch processing
        tokenized_input = self._tokenizer(
            input_data, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )

        # Model inference with no_grad
        with torch.no_grad():
            model_output = self._model(**tokenized_input)
            output_logits = model_output.logits

        # Get probabilities and predictions
        probabilities = torch.softmax(output_logits, dim=-1)
        scores, labels = torch.max(probabilities, dim=-1)
        labels_str = [self._model.config.id2label[label_idx] for label_idx in labels.tolist()]

        # Prepare output dict
        return [{"label": label, "score": score.item()} for label, score in zip(labels_str, scores)]

# Initialize pipeline
model_id = "kimsooyoung/roberta-base-klue-ynat-classification"
custom_pipeline = CustomPipeline(model_id)

# prepare dataset
klue_ynat_eval = load_dataset('klue', 'ynat', split="validation")
sample_data = klue_ynat_eval["title"][:5]

# inference and print
inf_result = custom_pipeline(sample_data)
print(f"Inference Result: {inf_result}")
