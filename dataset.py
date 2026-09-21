from datasets import load_dataset, concatenate_datasets
from kaggleds import kaggle_train_set, kaggle_test_set, kaggle_val_set
from utils import *

def score(x, id):
    if id == 0:
        t = {
            "entailment": ENTAILMENT,
            "contradiction": CONTRADICTION,
            "neutral": NEUTRAL
        }
    elif id == 1:
        t = {
            0: ENTAILMENT,
            1: NEUTRAL,
            2: CONTRADICTION
        }

    x[SCORE] = t.get(x[SCORE], -1)
    return x

_filter = lambda x: x[SCORE] != -1 

_ds1 = load_dataset("sentence-transformers/all-nli", name="pair-score")
_ds2 = (load_dataset("stanfordnlp/snli", split="test")
        .rename_columns({"premise":SENTENCE1_COLUMN, "hypothesis": SENTENCE2_COLUMN, "label": SCORE})
        .map(lambda x: score(x, 1))
        .filter(_filter))

_ds3 = (load_dataset("chrishuber/kaggle_mnli", split="train")
        .select_columns([SENTENCE1_COLUMN, SENTENCE2_COLUMN, "gold_label"])
        .rename_column("gold_label", SCORE)
        .map(lambda x: score(x, 0)))

train_dataset = concatenate_datasets([_ds1['train'], _ds3, kaggle_train_set])
validation_dataset = concatenate_datasets([_ds1['dev'], _ds2, kaggle_val_set])
test_dataset = concatenate_datasets([_ds1['test'], kaggle_test_set])

if __name__ == "__main__":
    print(train_dataset[0])
    print(test_dataset[0])
    print(validation_dataset[0])
