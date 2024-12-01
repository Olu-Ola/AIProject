import textattack
import transformers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from attack import attackrecipe

from baseModel import baseModel
from attack import attackrecipe


from textattack import Attacker

from textattack.metrics.attack_metrics import WordsPerturbed
from textattack.metrics.attack_metrics.attack_success_rate import AttackSuccessRate

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from textattack.loggers import CSVLogger
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult

from IPython.display import display, HTML

"""from textattack.attack_results import SuccessfulAttackResult
from IPython.display import display, HTML
from textattack.loggers import CSVLogger"""


print("""
  _____        _                 _   
 |  __ \      | |               | |  
 | |  | | __ _| |_ __ _ ___  ___| |_ 
 | |  | |/ _` | __/ _` / __|/ _ \ __|
 | |__| | (_| | || (_| \__ \  __/ |_ 
 |_____/ \__,_|\__\__,_|___/\___|\__|

""")
dataset_choice = input("Please select the dataset: \n 1. IMDB \n 2. YELP \n 3. Amazon \nYour Choice: ")

if dataset_choice == "1":
    file = "./Dataset/imdbDataset.csv"
    outputFile = "Imdb"
elif dataset_choice == "2":
    file = "./Dataset/yelpDataset.csv"
    outputFile = "Yelp"
elif dataset_choice == "3":
    file = "./Dataset/amazonDataset.csv"
    outputFile = "Amazon"
else:
    print("Please select from given options.")
    exit()

# Create Pandas Dataframe from the input file
df = pd.read_csv(file)

# Split the Pandas Dataframe into Train & Test Data Frames
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
#train_df = df[10:]
#test_df = df[:20]
# Convert Pandas DataFrame into TextAttack Dataset
train_dataset = textattack.datasets.Dataset(train_df.values.tolist(), ["input"])
test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"]) 

print("Dataset Loaded. Train & Test Split : 900 / 100.\n\n")


print("""
  _                     _   __  __           _      _ 
 | |     ___   __ _  __| | | \  / | ___   __| | ___| |
 | |    / _ \ / _` |/ _` | | |\/| |/ _ \ / _` |/ _ \ |
 | |___| (_) | (_| | (_| | | |  | | (_) | (_| |  __/ |
 |______\___/ \__,_|\__,_| |_|  |_|\___/ \__,_|\___|_|                                          
""")

# which model to load?
model_choice = input("Please select which model to load: \n 1. Bert \n 2. LSTM \n 3. CNN \nYour Choice: ")


# from library or from local file
FileOrLib = input("\n\nLoad Model from Library or File? \n 1. Library \n 2. File \nYour Choice: ")
if FileOrLib == "2":
    model_file = input("Please give the absolute or relative path to file: \nPath: ")

# train model or evaluate it
TrainOrTest = input("Do you wish to train the model or evaluate it? \n 1. Train \n 2. Test \nYour Choice: ")

# Depending on the choice of the model, load the model
if model_choice == "1":
    outputFile += "Bert"
    if FileOrLib != "2":
        model_file = "bert-base-uncased"
    model = baseModel.Bert(model_file)
    
elif model_choice == "2":
    outputFile += "LSTM"
    if FileOrLib != "2":
        model_file = "lstm-imdb"
    model = baseModel.LSTM(model_file)

elif model_choice == "3":
    outputFile += "CNN"
    if FileOrLib != "2":
        model_file = "cnn-imdb"
    model = baseModel.CNN(model_file)
else:
    print("Please insert 1, 2 or 3.")
    exit

# Set the trainer Arguments
training_args = textattack.TrainingArgs(
    num_epochs=2,             # Number of epochs
    per_device_train_batch_size=16, # Batch size
    learning_rate=2e-5,             # Learning rate
    early_stopping_epochs=85
)
trainer = textattack.Trainer(model, task_type='classification', attack=None, train_dataset=train_dataset, eval_dataset=test_dataset, training_args=training_args)

# Depending on the choice, train or test the loaded model.
if TrainOrTest == "1":
    trainer.train()
else:
    trainer.evaluate()

model = trainer.model_wrapper

print("""
     __   _   _             _   
    /  \ | |_| |_ __ _  ___| | __
   / /\ \| __| __/ _` |/ __| |/ /
  / ____ \ |_| || (_| | (__|   < 
 /_/    \_\__|\__\__,_|\___|_|\_\\

""")

# Run the attacks on the model.
print("Which attack should be applied: \n 1. TextFooler \n 2. Bae Replacement Only \n 3. Bae Insertion Only \n 4. Bae Insertion + Replacement")
attack_choice = input("Your Choice: ")

# in order to make an attack in Textattack library, we need 4 parameters. Goal Function, Constraints, Trasnformation and SearchMethod.

if attack_choice == "1":
    outputFile += "TextFooler"
    attack = attackrecipe.textfooler(model)


elif attack_choice == "2":
    outputFile += "BaeReplace"
    attack = attackrecipe.bertR(model)

elif attack_choice == "3":
    outputFile = "BaeInsert"
    attack = attackrecipe.bertI(model)

else:
    outputFile += "BaeInsertAndReplace"
    attack = attackrecipe.bertIR(model)

attack.__repr__()

results = []
print("Saving the results to " + outputFile)
for i, a in enumerate(attack):
    results.append(Attacker(a, test_dataset, textattack.AttackArgs(num_examples = -1, enable_advance_metrics=True, log_to_csv= "csv/"+outputFile+".csv", disable_stdout=True)).attack_dataset())

accuracy = []
maxpert = []
maxpertuntilsuccess = []
maxperturbation = [0,0.1,0.2,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1]
for i in results:
    accuracy.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])
    maxpert.append(WordsPerturbed().calculate(i)["max_words_changed"])
    maxpertuntilsuccess.append(WordsPerturbed().calculate(i)["num_words_changed_until_success"])

print(accuracy)



#attack_results = Attacker(attack, test_dataset, textattack.AttackArgs(num_examples = -1, enable_advance_metrics=True, log_to_csv= "csv/"+outputFile+".csv" )).attack_dataset()

"""
failedAttack = 0
maxPert = 0
max_words_changed = 0
num_words_changed = 0
#originalaccuracy = AttackSuccessRate.calculate(attack_results)["original_accuracy"]

words_perturbed = []
accuracies = []

for i, result in enumerate(attack_results):
    if isinstance(result, SkippedAttackResult):
        failedAttack += 1
        continue
    elif isinstance(result, FailedAttackResult):
        failedAttack += 1
        continue
    else:
        print("accuracy drop: ")
        print((failedAttack * 100.0) / (i+1)) 

        all_num_words = len(result.original_result.attacked_text.words)
        num_words_changed += len(result.original_result.attacked_text.all_words_diff(result.perturbed_result.attacked_text))
        #max_words_changed = max(max_words_changed , num_words_changed)
        #maxPert = max_words_changed/100
        
        print("MaxPerturbation: ")
        #print(maxPert)
        print(num_words_changed/100)
        #words_perturbed.append(maxPert)
        words_perturbed.append(num_words_changed/100)
        accuracies.append((failedAttack * 100.0) / (i+1))
"""

plt.figure(figsize=(8, 6))
plt.plot(maxperturbation, accuracy, 'o-', label="Attack Results")
plt.xlabel("Max Words Perturbed %")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Words Perturbed During Attack")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(outputFile + "Plot.png")
print("Plot saved as 'accuracy_vs_words_perturbed.png'")
############################################################################
plt.figure(figsize=(8, 6))
plt.plot(maxpert, accuracy, 'o-', label="Attack Results")
plt.xlabel("Max Words Perturbed")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Max Words Perturbed During Attack")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(outputFile + "MaxWordsPlot.png")
print("Plot saved as 'accuracy_vs_words_perturbed.png'")

############################################################################
plt.figure(figsize=(8, 6))
plt.plot(maxpertuntilsuccess, accuracy, 'o-', label="Attack Results")
plt.xlabel("Max Words Perturbed")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Max Words Perturbed During Attack")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(outputFile + "UntilSuccessPlot.png")
print("Plot saved as 'accuracy_vs_words_perturbed.png'")


"""
words_perturbed = []
accuracies = []

for result in attack_results:
    words_perturbed.append(result.perturbed_result.score)
    accuracies.append()

plt.figure(figsize=(8, 6))
plt.plot(words_perturbed, accuracies, 'o-', label="Attack Results")
plt.xlabel("Words Perturbed")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Words Perturbed During Attack")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(outputFile + "Plot.png")
print("Plot saved as 'accuracy_vs_words_perturbed.png'")
"""
"""
# Step 6: Analyze results and compute average USE similarity
total_use_similarity = 0
num_results_with_use = 0

for result in attack_results:
    # Access the advanced metrics from the result
    use_similarity = result.perturbed_result.advanced_metrics.get("USE", None)
    if use_similarity is not None:
        total_use_similarity += use_similarity
        num_results_with_use += 1

# Compute average USE similarity
average_use_similarity = (
    total_use_similarity / num_results_with_use if num_results_with_use > 0 else 0
)

print(f"Average USE Similarity: {average_use_similarity:.4f}")"""


"""
pd.options.display.max_colwidth = 480

# Initialize logger
logger = CSVLogger(filename="csv/"+outputFile+"Logger.csv", color_method="html")
# Log attack results
for result in attack_results:
    if isinstance(result, SuccessfulAttackResult):
        logger.log_attack_result(result)

# Create DataFrame and display
results = pd.DataFrame.from_records(logger.row_list)
#display(HTML(results[["original_text", "perturbed_text"]].to_html(escape=False)))
# Save DataFrame as HTML
results[["original_text", "perturbed_text"]].to_html("html/"+outputFile+".html", escape=False)

print("Results saved to results.html. Open this file in a browser to view.")


# Ensure logger is properly closed
logger.flush()
logger.close()
"""