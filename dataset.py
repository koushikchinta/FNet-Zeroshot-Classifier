from datasets import load_dataset, concatenate_datasets

SENTENCE1_COLUMN = "sentence1"
SENTENCE2_COLUMN = "sentence2"
SCORE = "score"


def normalize_score(sample: dict, _min: float, _max: float):
    sample[SCORE] = (sample[SCORE] - _min) / (_max - _min)
    return sample


_ds1 = load_dataset("sentence-transformers/all-nli", name="pair-score", split="train")
_ds2 = load_dataset("mteb/sts12-sts", split="test").map(
    normalize_score, fn_kwargs={"_min": 0.0, "_max": 5.0}
)
_ds3 = load_dataset("SemRel/SemRel2024", name="eng", split="train").rename_column(
    "label", SCORE
)
_ds4 = load_dataset("mteb/sickr-sts", split="test").map(
    normalize_score, fn_kwargs={"_min": 1.0, "_max": 5.0}
)


train_dataset = concatenate_datasets([_ds1, _ds2, _ds3, _ds4])
validation_dataset = None
test_dataset = None

if __name__ == "__main__":
    print(train_dataset[0])
