
import pandas as pd
from sklearn.model_selection import train_test_split
import transformers
import textattack
import transformers
from transformers import AdamW
import numpy as np


output_csv = "../Dataset/IMDB/output_file.csv"
df = pd.read_csv(output_csv)

# Split into train (first 900 rows) and test (last 100 rows)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = textattack.datasets.Dataset(train_df.values.tolist(), ["input"])
test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"]) #, {0: 1, 1: 0} , ["Positive","Negative"])
# Load model, tokenizer, and model_wrapper
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(textattack.shared.utils.device) # to set the model on the device

optimizer = AdamW(model.parameters(), lr=2e-5)
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

training_args = textattack.TrainingArgs(
    num_epochs=2,             # Number of epochs
    per_device_train_batch_size=16, # Batch size
    learning_rate=2e-5,             # Learning rate
    early_stopping_epochs=85
)
trainer = textattack.Trainer(model_wrapper, task_type='classification', attack=None, train_dataset=train_dataset, eval_dataset=test_dataset, training_args=training_args)

trainer.train()
#trainer.evaluate()