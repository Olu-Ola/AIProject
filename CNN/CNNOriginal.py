import transformers
import textattack
from textattack import Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger_eng')
from textattack.attack_results import SuccessfulAttackResult
from textattack.loggers import CSVLogger
from sklearn.model_selection import train_test_split
import pandas as pd
from textattack.loggers import CSVLogger
from textattack.attack_results import SuccessfulAttackResult
from IPython.display import display, HTML

output_csv = "../Dataset/IMDB/output_file.csv"
df = pd.read_csv(output_csv)

# Split into train (first 900 rows) and test (last 100 rows)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = textattack.datasets.Dataset(train_df.values.tolist(), ["input"])
test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"])
print(len(test_dataset))

model = textattack.models.helpers.word_cnn_for_classification.WordCNNForClassification.from_pretrained("cnn-imdb")

emb_layer = textattack.models.helpers.glove_embedding_layer.GloveEmbeddingLayer(emb_layer_trainable=False)
word2id = emb_layer.word2id
tokenizer = textattack.models.tokenizers.glove_tokenizer.GloveTokenizer(word2id, pad_token_id = 0, unk_token_id=100)
#tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)


training_args = textattack.TrainingArgs(
    num_epochs=2,             # Number of epochs
    per_device_train_batch_size=16, # Batch size
    learning_rate=2e-5,             # Learning rate
    early_stopping_epochs=85
)
trainer = textattack.Trainer(model_wrapper, task_type='classification', attack=None, train_dataset=train_dataset, eval_dataset=test_dataset, training_args=training_args)

trainer.train()
#trainer.evaluate()