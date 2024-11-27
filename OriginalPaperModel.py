import transformers
from transformers import AdamW, BertForSequenceClassification, Trainer, TrainingArguments
import textattack
import pandas as pd
from sklearn.model_selection import train_test_split

output_csv = "./Dataset/IMDB/output_file.csv"
df = pd.read_csv(output_csv)

# Split into train (first 900 rows) and test (last 100 rows)
train_df = df.iloc[:900]  # it should be 900
#test_df = df.iloc[-10:]  # it should be 10
#test_df = df.iloc[:10]
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_dataset = textattack.datasets.Dataset(train_df.values.tolist(), ["input"])#, {0: 1, 1: 0} , ["Positive","Negative"])
val_dataset = textattack.datasets.Dataset(val_df.values.tolist(), ["input"])#, {0: 1, 1: 0} , ["Positive","Negative"])

# Load a pretrained BERT model for classification
#model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./resultsOriginalPaperModel",         # Directory to save results
    run_name = "bert-finetune-run1",
    report_to="none",
    num_train_epochs=2,             # Number of epochs
    per_device_train_batch_size=16, # Batch size
    learning_rate=2e-5,             # Learning rate
    evaluation_strategy="epoch",    # Evaluate after each epoch
)



# Initialize Trainer
trainer = Trainer(
    model=model, 
    args=training_args.to_dict(),
    train_dataset=train_dataset,   # Your training dataset
    eval_dataset=val_dataset,     # Your validation dataset
)

# Train the model
trainer.train()
