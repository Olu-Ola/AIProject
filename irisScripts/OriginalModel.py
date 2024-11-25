# Load model, tokenizer, and model_wrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
training_args = textattack.TrainingArgs(
    num_epochs=3,
    learning_rate=2e-5,
)

trainer = textattack.Trainer(model_wrapper, task_type='classification', attack=None, train_dataset=train_dataset, eval_dataset=val_dataset, training_args=training_args)

trainer.train()