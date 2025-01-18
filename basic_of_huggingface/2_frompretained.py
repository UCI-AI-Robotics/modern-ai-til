from transformers import AutoModel, AutoModelForSequenceClassification

# import model body
model_id = "klue/roberta-base"
body_only_model = AutoModel.from_pretrained(model_id)

# import model body with head
model_id = "SamLowe/roberta-base-go_emotions"
body_with_head_model = AutoModelForSequenceClassification.from_pretrained(model_id)

# import model body with random head
model_id = "klue/roberta-base"
body_with_random_head_model = AutoModelForSequenceClassification.from_pretrained(model_id)
