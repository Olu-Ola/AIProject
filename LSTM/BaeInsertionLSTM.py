import transformers
import textattack
from textattack import Attacker
from textattack.attack_recipes import BAEGarg2019 
from textattack.datasets import Dataset
import pandas as pd
from textattack.attack_results import SuccessfulAttackResult
from textattack.loggers import CSVLogger
from sklearn.model_selection import train_test_split
import pandas as pd
from textattack.loggers import CSVLogger
from textattack.attack_results import SuccessfulAttackResult
from IPython.display import display, HTML

from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR, GreedySearch
from textattack.transformations import WordInsertionMaskedLM, WordSwapMaskedLM

output_csv = "../Dataset/IMDB/output_file.csv"
df = pd.read_csv(output_csv)

# Split into train (first 900 rows) and test (last 100 rows)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"])
print(len(test_dataset))
model = textattack.models.helpers.lstm_for_classification.LSTMForClassification.from_pretrained("./outputs/2024-11-28-02-21-13-606552/best_model/")

emb_layer = textattack.models.helpers.glove_embedding_layer.GloveEmbeddingLayer(emb_layer_trainable=False)
word2id = emb_layer.word2id
tokenizer = textattack.models.tokenizers.glove_tokenizer.GloveTokenizer(word2id, pad_token_id = 0, unk_token_id=100)
#tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)

transformation = WordInsertionMaskedLM()
constraints = [RepeatModification(), StopwordModification()]
constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
constraints.append(use_constraint)
#
# Goal is untargeted classification.
#
goal_function = UntargetedClassification(model_wrapper)
search_method = GreedySearch()

baerecipe = BAEGarg2019(goal_function, constraints, transformation, search_method)

# Attack the dataset
attack_results = Attacker(baerecipe, test_dataset, textattack.AttackArgs(num_examples=-1, log_to_csv="BaeInsertionLSTM.csv")).attack_dataset()

# Increase column width for better readability
pd.options.display.max_colwidth = 480

# Initialize logger
logger = CSVLogger(color_method="html")

# Log attack results
for result in attack_results:
    if isinstance(result, SuccessfulAttackResult):
        logger.log_attack_result(result)

# Create DataFrame and display
results = pd.DataFrame.from_records(logger.row_list)
#display(HTML(results[["original_text", "perturbed_text"]].to_html(escape=False)))
# Save DataFrame as HTML
results[["original_text", "perturbed_text"]].to_html("BertI_results.html", escape=False)

print("Results saved to results.html. Open this file in a browser to view.")


# Ensure logger is properly closed
logger.flush()
logger.close()
