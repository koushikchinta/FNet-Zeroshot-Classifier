from datasets import load_dataset, concatenate_datasets

SENTENCE1_COLUMN = "sentence1"
SENTENCE2_COLUMN = "sentence2"
SCORE = "score"

def score(x):
    t = {
        "entailment": 0,
        "contradiction": 2,
        "neutral": 1

    }
    x[SCORE] = t.get(x[SCORE], -1)
    return x

_ds1 = load_dataset("sentence-transformers/all-nli", name="pair-score")
_ds2 = (load_dataset("stanfordnlp/snli", split="test")
        .rename_columns({"premise":SENTENCE1_COLUMN, "hypothesis": SENTENCE2_COLUMN, "label": SCORE}))

_ds3 = (load_dataset("chrishuber/kaggle_mnli", split="train")
        .select_columns([SENTENCE1_COLUMN, SENTENCE2_COLUMN, "gold_label"])
        .rename_column("gold_label", SCORE)
        .map(score)
        .filter(lambda x: x[SCORE] != -1))
train_dataset = concatenate_datasets([_ds1['train'], _ds3])
validation_dataset = concatenate_datasets([_ds1['dev'], _ds2])
test_dataset = _ds1['test']

if __name__ == "__main__":
    print(train_dataset[0])
