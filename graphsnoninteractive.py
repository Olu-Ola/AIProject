import textattack
import transformers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from attack import attackrecipe

from baseModel import baseModel
from graphattack import attackrecipe


from textattack import Attacker

from textattack.metrics.attack_metrics import WordsPerturbed
from textattack.metrics.attack_metrics.attack_success_rate import AttackSuccessRate

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from textattack.loggers import CSVLogger
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult

from IPython.display import display, HTML

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
train_df = train_df[:10]
test_df = test_df[-10:]
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

#model = trainer.model_wrapper

print("""
     __   _   _             _   
    /  \ | |_| |_ __ _  ___| | __
   / /\ \| __| __/ _` |/ __| |/ /
  / ____ \ |_| || (_| | (__|   < 
 /_/    \_\__|\__\__,_|\___|_|\_\\

""")

# Run the attacks on the model.
allAttacks = []

# in order to make an attack in Textattack library, we need 4 parameters. Goal Function, Constraints, Trasnformation and SearchMethod.

allAttacks.append(attackrecipe.textfooler(model))
allAttacks.append(attackrecipe.bertR(model))
allAttacks.append(attackrecipe.bertI(model))
allAttacks.append(attackrecipe.bertIR(model))

# allAttacks[4].attacks[10]


results = []
for j,attack in enumerate(allAttacks):
    for i, a in enumerate(attack):
        results.append(Attacker(a, test_dataset, textattack.AttackArgs(num_examples = -1, enable_advance_metrics=True, disable_stdout=True)).attack_dataset())

# results -? [4] & [10]
# result - [4] 0> textfooler, [1].bertR, ....
# textfooler-> 10times the results

    
#accuracy = []
maxpert = []
maxpertuntilsuccess = []
maxperturbation = [0,0.1,0.2,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1]

accuracyTextFooler = []
accuracyBaeR = []
accuracyBaeI = []
accuracyBaeIR = []

for k in results:
    for i in k:
        if i == 0:
            accuracyTextFooler.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])
        elif i == 1:
            accuracyBaeR.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])
        elif i == 2:
            accuracyBaeI.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])
        elif i == 3:
            accuracyBaeIR.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])

        
        #maxpert.append(WordsPerturbed().calculate(i)["max_words_changed"])
        #maxpertuntilsuccess.append(WordsPerturbed().calculate(i)["num_words_changed_until_success"])

#print(accuracy)

plt.figure(figsize=(8, 6))
plt.plot(maxperturbation, accuracyTextFooler, label='TEXTFOOLER', color='blue', linewidth=1)
plt.plot(maxperturbation, accuracyBaeR, label='BAE-R', color='orange', linewidth=1)
plt.plot(maxperturbation, accuracyBaeI, label='BAE-I', color='green', linewidth=1)
plt.plot(maxperturbation, accuracyBaeIR, label='BAE-R+I', color='purple', linewidth=1)

plt.xlabel(" Maximum Percent Perturbation")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs. Maximum Percent Perturbation")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig("./png/" + outputFile + "Plot.png")
print("Plot saved as " + outputFile + ".png'")

