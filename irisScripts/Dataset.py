
import pandas as pd
import os
from sklearn.model_selection import train_test_split


os.system("rm train.csv")
os.system("rm test.csv")
os.system("rm train_split.csv")
os.system("rm val_split.csv")
os.system("rm output_file.csv")
#######################################
# Step 1: Read the .txt file
file_path = "imdb_labelled.txt"  # Replace with your file path
df = pd.read_csv(file_path, sep=r"\s{3,}", engine="python", header=None, names=["Input", "Label"])

# Step 2: Save to a CSV file
output_csv = "output_file.csv"  # Replace with your desired output file name
df.to_csv(output_csv, index=False)

print(f"CSV file saved as {output_csv}")

#######################################

import pandas as pd

# Load the CSV
file_path = "output_file.csv"  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Split into train (first 900 rows) and test (last 100 rows)
train_df = df.iloc[:10]  # it should be 900
test_df = df.iloc[-10:]  # it should be 10

# Save the split datasets
train_csv = "train.csv"
test_csv = "test.csv"

train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Train data saved to {train_csv}")
print(f"Test data saved to {test_csv}")

#######################################

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the train CSV
train_csv = "train.csv"  # Path to your existing train.csv file
df = pd.read_csv(train_csv)

# Split into 80% train and 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split datasets
train_csv_split = "train_split.csv"
val_csv_split = "val_split.csv"

train_df.to_csv(train_csv_split, index=False)
val_df.to_csv(val_csv_split, index=False)

print(f"Train data saved to {train_csv_split}")
print(f"Validation data saved to {val_csv_split}")

##########################################