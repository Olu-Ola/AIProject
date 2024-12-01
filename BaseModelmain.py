import pandas as pd
from sklearn.model_selection import train_test_split
import textattack
from baseModel import baseModel



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
    #test_df = df[-10:]
    # Convert Pandas DataFrame into TextAttack Dataset
    train_dataset = textattack.datasets.Dataset(train_df.values.tolist(), ["input"])
    test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"]) 

    print("Dataset "+ datasetnames[i] +" Loaded. Train & Test Split : 900 / 100.\n\n")


    print("""
    _                     _   __  __           _      _ 
    | |     ___   __ _  __| | | \  / | ___   __| | ___| |
    | |    / _ \ / _` |/ _` | | |\/| |/ _ \ / _` |/ _ \ |
    | |___| (_) | (_| | (_| | | |  | | (_) | (_| |  __/ |
    |______\___/ \__,_|\__,_| |_|  |_|\___/ \__,_|\___|_|                                          
    """)
    
    model = []
    outputFile = []
    model.append(baseModel.Bert(modelNames[0])) # modelName[i][0]
    outputFile.append(datasetnames[i]+"BertOutputs")

    model.append(baseModel.LSTM(modelNames[1])) # modelName[i][1]
    outputFile.append(datasetnames[i]+"LstmOutputs")

    model.append(baseModel.CNN(modelNames[2])) # modelName[i][2]
    outputFile.append(datasetnames[i]+"CnnOutputs")
        
    for j in range(3):
        # Set the trainer Arguments
        training_args = textattack.TrainingArgs(
            num_epochs=2,                   # Number of epochs
            per_device_train_batch_size=16, # Batch size
            learning_rate=2e-5,             # Learning rate
            early_stopping_epochs=2,
            output_dir = "../BestModels/"+outputFile[j]
        )

        trainer = textattack.Trainer(
            model[j], 
            task_type='classification', 
            attack=None, 
            train_dataset=train_dataset, 
            eval_dataset=test_dataset, 
            training_args=training_args
        )

        # Depending on the choice, train or test the loaded model.
        trainer.train()
        #trainer.evaluate()
