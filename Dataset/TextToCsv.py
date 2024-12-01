
import pandas as pd


file_path = input("Give the path to txt file: ")
#file_path="Dataset/TxtOriginalDataset/amazon_cells_labelled.txt"
#file_path="Dataset/TxtOriginalDataset/imdb_labelled.txt.txt
#file_path="Dataset/TxtOriginalDataset/yelp_labelled.txt.txt
#######################################
# Step 1: Read the .txt file
df = pd.read_csv(file_path, sep="\t", engine="python", header=None, names=["Input", "Label"])
print(df)

output_csv = input("Save the .csv file with name: ")
# Step 2: Save to a CSV file
df.to_csv(output_csv, index=False)

print(f"CSV file saved as {output_csv}")
