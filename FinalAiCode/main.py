import textattack
import transformers
import pandas as pd
from sklearn.model_selection import train_test_split


from textattack.attack_recipes import AttackRecipe
from textattack import Attacker

from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (InputColumnModification, RepeatModification,StopwordModification)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR, GreedySearch
from textattack.transformations import WordInsertionMaskedLM, WordSwapMaskedLM, WordSwapEmbedding


from textattack.attack_results import SuccessfulAttackResult
from IPython.display import display, HTML
from textattack.loggers import CSVLogger

print("""
  _____        _                 _   
 |  __ \      | |               | |  
 | |  | | __ _| |_ __ _ ___  ___| |_ 
 | |  | |/ _` | __/ _` / __|/ _ \ __|
 | |__| | (_| | || (_| \__ \  __/ |_ 
 |_____/ \__,_|\__\__,_|___/\___|\__|

""")
dataset_choice = input("Please select the dataset: \n 1. IMDB \n 2. YELP \n 3. Amazon \nYour Choice: ")

if dataset_choice == 1:
    file = "../Dataset/IMDB.csv"
elif dataset_choice == 2:
    file = "../Dataset/YELP.csv"
elif dataset_choice == 3:
    file = "../Dataset/Amazon.csv"
else:
    print("Please select from given options.")
    exit()

# Create Pandas Dataframe from the input file
df = pd.read_csv(file)
outputFile = "IMDB"
# Split the Pandas Dataframe into Train & Test Data Frames
train_df, test_df = train_test_split(df, test_size=0.1, random_state=62)

# Convert Pandas DataFrame into TextAttack Dataset
train_dataset = textattack.datasets.Dataset(train_df.values.tolist(), ["input"])
test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"]) 


print("""
  _                     _   __  __           _      _ 
 | |     ___   __ _  __| | | \  / | ___   __| | ___| |
 | |    / _ \ / _` |/ _` | | |\/| |/ _ \ / _` |/ _ \ |
 | |___| (_) | (_| | (_| | | |  | | (_) | (_| |  __/ |
 |______\___/ \__,_|\__,_| |_|  |_|\___/ \__,_|\___|_|                                          
""")

# Get input from user about which model to load?
print("Please select which model to load: \n 1. BertModel \n 2. LSTM \n 3. CNN")
model_choice = input("Your Choice: ")

# Whether to load the model from library or local file
FileOrLib = input("\n\nLoad Model from Library or File? \n 1. Library \n 2. File \nYour Choice: ")
if FileOrLib == "2":
    model_file = input("Please give the absolute or relative path to file: \nYour Choice: ")

# Whether to train the loaded model again or evaluate it
print("\n\nDo you wish to train the model or evaluate it? \n 1. Train \n 2. Test")
TrainOrTest = input("Your Choice: ")

# Depending on the choice of the model, load the model
if model_choice == "1":
    outputFile += "TextFooler"
    if FileOrLib != "2":
        model_file = "bert-base-uncased"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_file).to(textattack.shared.utils.device) # to set the model on the device
    
elif model_choice == "2":
    outputFile += "LSTM"
    if FileOrLib != "2":
        model_file = "lstm-imdb"
    model = textattack.models.helpers.lstm_for_classification.LSTMForClassification.from_pretrained(model_file).to(textattack.shared.utils.device)

elif model_choice == "3":
    outputFile += "CNN"
    if FileOrLib != "2":
        model_file = "cnn-imdb"
    model = textattack.models.helpers.word_cnn_for_classification.WordCNNForClassification.from_pretrained(model_file).to(textattack.shared.utils.device)

else:
    print("Please insert 1, 2 or 3.")
    exit

# Get the Tokenizer for the model
if model_choice == "1":
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
elif model_choice == "2" or model_choice == "3":
    emb_layer = textattack.models.helpers.glove_embedding_layer.GloveEmbeddingLayer(emb_layer_trainable=False)
    word2id = emb_layer.word2id
    tokenizer = textattack.models.tokenizers.glove_tokenizer.GloveTokenizer(word2id, pad_token_id = 0, unk_token_id=100)

# Wrap the model
model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)

# Set the trainer Arguments
training_args = textattack.TrainingArgs(
    num_epochs=2,             # Number of epochs
    per_device_train_batch_size=16, # Batch size
    learning_rate=2e-5,             # Learning rate
    early_stopping_epochs=85
)
trainer = textattack.Trainer(model_wrapper, task_type='classification', attack=None, train_dataset=train_dataset, eval_dataset=test_dataset, training_args=training_args)

# Depending on the choice, train or test the loaded model.
if TrainOrTest == "1":
    trainer.train()
else:
    trainer.evaluate()


print("""
     __   _   _             _   
    /  \ | |_| |_ __ _  ___| | __
   / /\ \| __| __/ _` |/ __| |/ /
  / ____ \ |_| || (_| | (__|   < 
 /_/    \_\__|\__\__,_|\___|_|\_\\

""")

# Run the attacks on the model.
print("Which attack should be applied: \n 1. TextFooler \n 2. Bae Replacement Only \n 3. Bae Insertion Only \n 4. Bae Insertion + Replacement")
attack_recipe = input("Your Choice: ")




constraints = [RepeatModification(), StopwordModification()]

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

if attack_recipe == "1":
    outputFile += "TextFooler"
    transformation = WordSwapEmbedding(max_candidates=50)
    stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
    )
    constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
    input_column_modification = InputColumnModification(
        ["premise", "hypothesis"], {"premise"}
    )
    constraints.append(input_column_modification)
    constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
    use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
    constraints.append(use_constraint)

elif attack_recipe == "2":
    outputFile += "BaeReplace"
    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
    transformation = WordSwapMaskedLM(max_candidates=50)
    search_method = GreedySearch(goal_function, constraints, transformation, search_method)
elif attack_recipe == "3":
    outputFile = "BaeInsert"
    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
    transformation = WordInsertionMaskedLM()
else:
    outputFile += "BaeInsertAndReplace"
    constraints.append(PartOfSpeech(allow_verb_noun_swap=False, compare_against_original=False))
    transformations = [WordSwapMaskedLM(max_candidates=50),WordInsertionMaskedLM()]
    transformation = textattack.transformations.composite_transformation.CompositeTransformation(transformations)
    
attack = AttackRecipe(goal_function, constraints, transformation, search_method)

attack_results = Attacker(attack, test_dataset, textattack.AttackArgs(num_examples=-1, log_to_csv="TestFoolerLSTM.csv")).attack_dataset()

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
results[["original_text", "perturbed_text"]].to_html("TextFoolerresults.html", escape=False)

print("Results saved to results.html. Open this file in a browser to view.")


# Ensure logger is properly closed
logger.flush()
logger.close()








