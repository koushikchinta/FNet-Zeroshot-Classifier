from datasets import load_dataset, concatenate_datasets

SENTENCE1_COLUMN = "sentence1"
SENTENCE2_COLUMN = "sentence2"
SCORE = "score"

_ds1 = load_dataset("sentence-transformers/all-nli", name="pair-score")

train_dataset = _ds1['train']
validation_dataset = _ds1['dev']
test_dataset = _ds1['test']

if __name__ == "__main__":
    print(train_dataset[0])
