
import pandas as pd


file_path = input("Give the path to txt file: ")
#######################################
# Step 1: Read the .txt file
df = pd.read_csv(file_path, sep=r"\s{3,}", engine="python", header=None, names=["Input", "Label"])

output_csv = input("Save the .csv file with name: ")
# Step 2: Save to a CSV file
df.to_csv(output_csv, index=False)

print(f"CSV file saved as {output_csv}")
