
import pandas as pd
import os
from sklearn.model_selection import train_test_split

datasetDir = "./Dataset/IMDB/"
#os.system("rm" + datasetDir + "train.csv")
#os.system("rm" + datasetDir + "test.csv")
#os.system("rm" + datasetDir + "train_split.csv")
#os.system("rm" + datasetDir + "val_split.csv")
#os.system("rm" + datasetDir + "output_file.csv")
#os.system("mkdir ./Dataset/IMDB/")
file_path = datasetDir + "imdb_labelled.txt"
output_csv = datasetDir + "output_file.csv"
#######################################
# Step 1: Read the .txt file
df = pd.read_csv(file_path, sep=r"\s{3,}", engine="python", header=None, names=["Input", "Label"])

# Step 2: Save to a CSV file
df.to_csv(output_csv, index=False)

print(f"CSV file saved as {output_csv}")

#######################################

# Load the CSV
df = pd.read_csv(output_csv)

# Split into train (first 900 rows) and test (last 100 rows)
train_df = df.iloc[:10]  # it should be 900
#test_df = df.iloc[-10:]  # it should be 10
test_df = df.iloc[:10]
# Save the split datasets
#train_csv = "train.csv"
#test_csv = "test.csv"
#
#train_df.to_csv(train_csv, index=False)
#test_df.to_csv(test_csv, index=False)
#
#print(f"Train data saved to {train_csv}")
#print(f"Test data saved to {test_csv}")

# Load the train CSV
#train_csv = "train.csv"  # Path to your existing train.csv file
#df = pd.read_csv(train_csv)

# Split into 80% train and 20% validation
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Save the split datasets
#train_csv_split = "train_split.csv"
#val_csv_split = "val_split.csv"
#
#train_df.to_csv(train_csv_split, index=False)
#val_df.to_csv(val_csv_split, index=False)
#
#print(f"Train data saved to {train_csv_split}")
#print(f"Validation data saved to {val_csv_split}")
#######################################