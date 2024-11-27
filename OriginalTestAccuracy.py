import pandas as pd
import textattack
from textattack.commands.eval_model_command import EvalModelCommand, ModelEvalArgs
import scipy
import torch
import transformers

logger = textattack.shared.utils.logger

def _cb(s):
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")

modell = transformers.AutoModelForSequenceClassification.from_pretrained("./outputs/2024-11-26-01-41-22-862545/best_model/")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model = textattack.models.wrappers.HuggingFaceModelWrapper(modell, tokenizer)

output_csv = "./Dataset/IMDB/output_file.csv"
df = pd.read_csv(output_csv)
test_df = df.iloc[:100]
dataset = textattack.datasets.Dataset(test_df.values.tolist(), ["input"])#, {0: 1, 1: 0} , ["Positive","Negative"])

preds = []
ground_truth_outputs = []
i = 0
num_examples = len(dataset)
batch_size = 16
while i < num_examples:
    dataset_batch = dataset[i : min(num_examples, i + batch_size)] # 16 is batch size
    batch_inputs = []
    for text_input, ground_truth_output in dataset_batch:
        #print(text_input["input"])
        #print(ground_truth_output)
        attacked_text = textattack.shared.AttackedText(text_input)
        batch_inputs.append(attacked_text.tokenizer_input)
        ground_truth_outputs.append(ground_truth_output)
    batch_preds = model(batch_inputs)

    if not isinstance(batch_preds, torch.Tensor):
        batch_preds = torch.Tensor(batch_preds)

    preds.extend(batch_preds)
    i += batch_size

preds = torch.stack(preds).squeeze().cpu()
ground_truth_outputs = torch.tensor(ground_truth_outputs).cpu()

logger.info(f"Got {len(preds)} predictions.")

if preds.ndim == 1:
    # if preds is just a list of numbers, assume regression for now
    # TODO integrate with `textattack.metrics` package
    pearson_correlation, _ = scipy.stats.pearsonr(ground_truth_outputs, preds)
    spearman_correlation, _ = scipy.stats.spearmanr(ground_truth_outputs, preds)

    logger.info(f"Pearson correlation = {_cb(pearson_correlation)}")
    logger.info(f"Spearman correlation = {_cb(spearman_correlation)}")
else:
    guess_labels = preds.argmax(dim=1)
    successes = (guess_labels == ground_truth_outputs).sum().item()
    perc_accuracy = successes / len(preds) * 100.0
    perc_accuracy = "{:.2f}%".format(perc_accuracy)
    logger.info(f"Correct {successes}/{len(preds)} ({_cb(perc_accuracy)})")