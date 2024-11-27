import transformers
import textattack
from textattack import Attacker
from textattack.datasets import Dataset
import pandas as pd
from textattack.attack_results import SuccessfulAttackResult
from textattack.loggers import CSVLogger
from sklearn.model_selection import train_test_split
import pandas as pd
from textattack.loggers import CSVLogger
from textattack.attack_results import SuccessfulAttackResult
from IPython.display import display, HTML
from BaeI import BAEIR

output_csv = "../Dataset/IMDB/output_file.csv"
df = pd.read_csv(output_csv)

# Split into train (first 900 rows) and test (last 100 rows)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
test_dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"])
print(len(test_dataset))
model = transformers.AutoModelForSequenceClassification.from_pretrained("./outputs/2024-11-27-04-35-13-047658/best_model/")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

baerecipe = BAEIR("insert").build(model_wrapper)

# Attack the dataset
attack_results = Attacker(baerecipe, test_dataset, textattack.AttackArgs(num_examples=-1)).attack_dataset()

# Increase column width for better readability
pd.options.display.max_colwidth = 480

# Initialize logger
logger = CSVLogger(color_method="html")

# Log attack results
for result in attack_results:
    if isinstance(result, SuccessfulAttackResult):
        logger.log_attack_result(result)

# Create DataFrame and display
results = pd.DataFrame.from_records(logger.row_list)
#display(HTML(results[["original_text", "perturbed_text"]].to_html(escape=False)))
# Save DataFrame as HTML
results[["original_text", "perturbed_text"]].to_html("BertI_results.html", escape=False)

print("Results saved to results.html. Open this file in a browser to view.")


# Ensure logger is properly closed
logger.flush()
logger.close()
