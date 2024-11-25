
import pandas as pd
import os
from sklearn.model_selection import train_test_split

datasetDir = "./Dataset/IMDB/"
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
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)