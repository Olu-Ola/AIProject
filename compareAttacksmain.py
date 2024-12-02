import pandas as pd
from sklearn.model_selection import train_test_split
import textattack
from baseModel import baseModel

from attack import attackrecipe
from textattack import Attacker
from textattack.attack_results import SuccessfulAttackResult
from textattack.shared import AttackedText
from textattack.loggers import CSVLogger



datasets = ["./Dataset/imdbDataset.csv", "./Dataset/yelpDataset.csv", "./Dataset/amazonDataset.csv"]
modelNames =[]
modelNames.append("bert-base-uncased")
modelNames.append("lstm-imdb")
modelNames.append("cnn-imdb")
#modelNames.append(["bert-base-uncased","lstm-yelp","cnn-yelp"])
#modelNames.append(["bert-base-uncased","lstm","cnn"])
datasetnames = ["imdb", "yelp", "amazon"]
for i, file in enumerate(datasets):
    print("""
     _____        _                 _   
    |  __ \      | |               | |  
    | |  | | __ _| |_ __ _ ___  ___| |_ 
    | |  | |/ _` | __/ _` / __|/ _ \ __|
    | |__| | (_| | || (_| \__ \  __/ |_ 
    |_____/ \__,_|\__\__,_|___/\___|\__|

    """)
    # Create Pandas Dataframe from the input file
    df = pd.read_csv(file)

    # Split the Pandas Dataframe into Train & Test Data Frames
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    #train_df = df[:10]
    test_df = test_df[-15:]
    # Convert Pandas DataFrame into TextAttack Dataset
    #train_dataset = textattack.datasets.Dataset(train_df.values.tolist(), ["input"])
    test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"]) 

    print("Dataset "+ datasetnames[i] +" Loaded.\n\n")


    print("""
    _                     _   __  __           _      _ 
    | |     ___   __ _  __| | | \  / | ___   __| | ___| |
    | |    / _ \ / _` |/ _` | | |\/| |/ _ \ / _` |/ _ \ |
    | |___| (_) | (_| | (_| | | |  | | (_) | (_| |  __/ |
    |______\___/ \__,_|\__,_| |_|  |_|\___/ \__,_|\___|_|                                          
    """)
    
    models = []
    outputFile = []
    modelfiles = ["Bert", "Lstm", "Cnn"]
    #models.append(baseModel.Bert("./ModelsOutputs/" + datasetnames[i] + modelfiles[0] + "/best_model/")) # modelName[i][0]
    
    #models.append(baseModel.LSTM("./ModelsOutputs/" + datasetnames[i] + modelfiles[1]+ "/best_model/")) # modelName[i][1]
    
    models.append(baseModel.CNN("./ModelsOutputs/" + datasetnames[i] + modelfiles[2]+ "/best_model/")) # modelName[i][2]

    print("""
         __   _   _             _   
        /  \ | |_| |_ __ _  ___| | __
       / /\ \| __| __/ _` |/ __| |/ /
      / ____ \ |_| || (_| | (__|   < 
     /_/    \_\__|\__\__,_|\___|_|\_\\

    """)

    attacks = []
    outputfilenames = []
    for j, model in enumerate(models):
        attacks.append(attackrecipe.textfooler(model))
        outputfilenames.append(datasetnames[i]+modelfiles[j]+"textfooler")
        attacks.append(attackrecipe.bertR(model))
        outputfilenames.append(datasetnames[i]+modelfiles[j]+"baeR")
        attacks.append(attackrecipe.bertI(model))
        outputfilenames.append(datasetnames[i]+modelfiles[j]+"baeI")
        attacks.append(attackrecipe.bertIR(model))
        outputfilenames.append(datasetnames[i]+modelfiles[j]+"baeIR")
    
    # Dataset = Dataset[i]
    # Model = model[0]
    # attacks = [4]--> textFooler, R, I , IR

    results = []
    for k, attack in enumerate(attacks):
        results.append(Attacker(attack, test_dataset, textattack.AttackArgs(num_examples = -1, enable_advance_metrics=False, disable_stdout=True)).attack_dataset())

    # I have 4 results but there are len(datasaet) instances of [originaltext],[perturbedtext]

    rows = []
    for k in range(len(test_dataset)):
        row = {"original_text": None, "textfooler_result": None, "bertr_result": None, "berti_result": None, "bertir_result": None}
        for j in range(len(attacks)):
                if isinstance(results[j][k], SuccessfulAttackResult):
                    original_text, perturbed_text = results[j][k].diff_color("html")
                    original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
                    perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
                    if row["original_text"] is None:                                           
                        row["original_text"] = original_text
                    if j == 0:                       
                        row["textfooler_result"] = perturbed_text
                    elif j == 1:
                        row["bertr_result"] = perturbed_text
                    elif j == 2:
                        row["berti_result"] = perturbed_text
                    elif j == 3:
                        row["bertir_result"] = perturbed_text
                    
                    #print(perturbed_text)

                    #print(results[j][i].perturbed_text())
                else:
                    if j == 0:
                        row["textfooler_result"] = "AttackFailed"
                    elif j == 1:
                        row["bertr_result"] = "AttackFailed"
                    elif j == 2:
                        row["berti_result"] = "AttackFailed"
                    elif j == 3:
                        row["bertir_result"] = "AttackFailed"
                    #print("AttackFailed")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Display or save the DataFrame
    #print(df)
    #df.to_csv(datasetnames[i]+"attack_results.csv", index=False)

    html_file = "./html/Cnn"+datasetnames[i]+"attack_results.html"
    df.to_html(html_file, index=False, escape=False)  # `escape=False` allows HTML tags like <span> to be included
    print(f"Results saved to {html_file}")

    #CSVLogger.flush()
    #CSVLogger.clear()
                


    


   