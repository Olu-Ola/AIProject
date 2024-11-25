model = transformers.AutoModelForSequenceClassification.from_pretrained("./outputs/2024-11-25-09-49-46-278589/best_model/")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)