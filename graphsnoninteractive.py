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



#file = "./Dataset/imdbDataset.csv"
#outputFile = "Imdb"

#file = "./Dataset/yelpDataset.csv"
#outputFile = "Yelp"

file = "./Dataset/amazonDataset.csv"
outputFile = "Amazon"


# Create Pandas Dataframe from the input file
df = pd.read_csv(file)

# Split the Pandas Dataframe into Train & Test Data Frames
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
#train_df = train_df[:10]
#test_df = test_df[-10:]
# Convert Pandas DataFrame into TextAttack Dataset
train_dataset = textattack.datasets.Dataset(train_df.values.tolist(), ["input"])
test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"]) 

print("Dataset "+outputFile+"Loaded. Train & Test Split : 900 / 100.\n\n")


print("""
  _                     _   __  __           _      _ 
 | |     ___   __ _  __| | | \  / | ___   __| | ___| |
 | |    / _ \ / _` |/ _` | | |\/| |/ _ \ / _` |/ _ \ |
 | |___| (_) | (_| | (_| | | |  | | (_) | (_| |  __/ |
 |______\___/ \__,_|\__,_| |_|  |_|\___/ \__,_|\___|_|                                          
""")


# Depending on the choice of the model, load the model

#outputFile += "Bert"
#model_file = "../ModelsOutputs/imdbBert/best_model/"
#model = baseModel.Bert(model_file)
    

outputFile += "LSTM"
model_file = "../ModelsOutputs/imdbLstm/best_model/"
model = baseModel.LSTM(model_file)


#outputFile += "CNN"
#model_file = "../ModelsOutputs/imdbCnn/best_model/"
#model = baseModel.CNN(model_file)

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


textfoolerresults = []
baeRresults = []
baeIresults = []
baeIRresults = []
for i, attack in enumerate(allAttacks):
    for a in attack:
        if i == 0:
            textfoolerresults.append(Attacker(a, test_dataset, textattack.AttackArgs(num_examples = -1, enable_advance_metrics=True, disable_stdout=True)).attack_dataset())
        elif i == 1:
            baeRresults.append(Attacker(a, test_dataset, textattack.AttackArgs(num_examples = -1, enable_advance_metrics=True, disable_stdout=True)).attack_dataset())
        elif i == 2:
            baeIresults.append(Attacker(a, test_dataset, textattack.AttackArgs(num_examples = -1, enable_advance_metrics=True, disable_stdout=True)).attack_dataset())
        elif i == 3:
            baeIRresults.append(Attacker(a, test_dataset, textattack.AttackArgs(num_examples = -1, enable_advance_metrics=True, disable_stdout=True)).attack_dataset())

# results -? [4] & [10]
# result - [4] 0> textfooler, [1].bertR, ....
# textfooler-> 10times the results

    
#accuracy = []
#maxpert = []
#maxpertuntilsuccess = []
maxperturbation = [0,0.2,0.40,0.60,0.80,1]
#maxperturbation = [0,0.2]
accuracyTextFooler = []
accuracyBaeR = []
accuracyBaeI = []
accuracyBaeIR = []

for i in textfoolerresults:
    accuracyTextFooler.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])
for i in baeRresults:
    accuracyBaeR.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])
for i in baeIresults:
    accuracyBaeI.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])
for i in baeIRresults:
    accuracyBaeIR.append(AttackSuccessRate().calculate(i)["attack_accuracy_perc"])

"""
for k in results:
    for i in range(len(allAttacks)):
        if i == 0:
            accuracyTextFooler.append(AttackSuccessRate().calculate(k)["attack_accuracy_perc"])
        elif i == 1:
            accuracyBaeR.append(AttackSuccessRate().calculate(k)["attack_accuracy_perc"])
        elif i == 2:
            accuracyBaeI.append(AttackSuccessRate().calculate(k)["attack_accuracy_perc"])
        elif i == 3:
            accuracyBaeIR.append(AttackSuccessRate().calculate(k)["attack_accuracy_perc"])  
"""
  
        
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

