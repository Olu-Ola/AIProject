# AIProject

## File Structure

    ### Dataset (Folder)
        The Dataset folder contains the csv files and a TxtOriginalDataset, a folder that contains the original text files and the python code to transform them into csv files.

    
    ### html (Folder)
        Stores all the results of compareAttackmain.py

    ### png (Folder)
        Stores all the graphs

    ### csv (Folder)
        stores all the csv files of attacks and models

    ### ScreenShots
        constains the screenshots of each model original accuracy and textattacks results.

    ### graphattack.py and graphsmain.py 
         used to generate the accuracy vs perturbed words percentages

    ### main.py
        the main file to reproduce all the results

    #### attack.py
        contains all the attack recipes with all their transformations, constrainst, goal functions and searchmethods.

    #### BaseModelmain.py
        runs all the models and train them on the given dataset with 90:10 train test ratio.

    ####  baseModel.py
        takes the input file and load the relevant model and returns the model wrapper.

    ####  compareAttacksmain.py
        compare all the attack results side by side for human evaluation



## Dataset

In this Project, we have worked with the sentiment analysis datasets namely IMDB, YELP and AMAZON, found from the link:
https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

We converted the txt files into CSVs for easy manipulation through Pandas Dataframes.

## Models

In order to replicate the experiments of the paper "BAE: BERT-based Adversarial Examples for Text Classification", we took the 3 base models namely:
wordLSTM, wordCNN and BERT. We then trained these models according to the parameters given in the paper.

## Interactive Experiments

The experiments can be performed in an interactive mode through the main.py file. The experiment has the following steps:
    1. Select the dataset to train or test the Model
    2. Select the model (wordLSTM, wordCNN, BERT)
        i.  Train the model
        ii. Test the model
    3. Give the model to attack the accuracy
    4. Select attack (TextFooler, BAE-Replace, BAE-Insert, BAE-Replace+Insert)

## Compare the results
The compareAttacksmain.py compare the results of all attacks on all datasets for a particular model. The results are generated in html file and stored in the html folder.

## Single Graphs
The graphsmain.py generates the graphs through the same interactive mode as that of main.py. A particular attack is performed with maximum perturbation constraints for multiple times and the graph is produced in Png folder.