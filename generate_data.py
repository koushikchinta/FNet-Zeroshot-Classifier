import os
import json
import time
import random
from pathlib import Path

import pandas as pd
from litellm import completion
from datasets import load_dataset


# ============================================================
# CONFIG
# ============================================================

MODEL = "ollama/qwen2.5:7b"
API_BASE = "http://localhost:11434"

TOTAL_DOCUMENTS = 10_000
DOCUMENTS_PER_BATCH = 1_000
NUM_TAGS = 40

OUTPUT_FILE = "synthetic_tag_dataset.parquet"

CHECKPOINT_DIR = Path("synthetic_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 5

NUM_BATCHES = TOTAL_DOCUMENTS // DOCUMENTS_PER_BATCH


# ============================================================
# LLM CALL
# ============================================================


def call_llm(prompt):
    """
    Call Ollama through LiteLLM and return parsed JSON.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = completion(
                model=MODEL,
                api_base=API_BASE,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            return json.loads(content)

        except Exception as e:
            print(f"LLM error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")

            if attempt < MAX_RETRIES - 1:
                time.sleep(2)

    raise RuntimeError("LLM failed after maximum retries.")


# ============================================================
# GENERATE 40 TAGS
# ============================================================


def generate_tags(batch_number):
    """
    Generate exactly 40 unique enterprise-document tags.

    These tags are generated ONCE per 1,000-document batch.
    """

    prompt = f"""
You are designing an enterprise document classification taxonomy.

Generate exactly {NUM_TAGS} unique document tags.

This taxonomy will be used to classify realistic enterprise
documents.

Requirements:

1. Generate exactly {NUM_TAGS} tags.
2. Every tag must be unique.
3. Use this format:

   Category - Specific Document Type

4. Examples:

   HR - Employee Records
   HR - Recruitment Documents
   Finance - Invoices
   Finance - Financial Reports
   Legal - Contracts
   IT - Technical Documentation
   Marketing - Campaign Assets

5. Use approximately 6-10 broad categories.

6. Include realistic enterprise document types such as:
   - policies
   - procedures
   - guides
   - reports
   - records
   - contracts
   - financial documents
   - HR documents
   - IT documents
   - engineering documents
   - marketing documents
   - legal/compliance documents
   - administrative documents
   - communications documents

7. The tags should be specific enough that a classifier
   can distinguish between semantically similar documents.

8. "Other" and "Miscellaneous" are allowed as tag types.

9. Do NOT create duplicate or nearly duplicate tags.

10. Return ONLY valid JSON.

Required format:

{{
    "tags": [
        "Category - Specific Document Type",
        "Category - Specific Document Type"
    ]
}}
"""

    result = call_llm(prompt)

    tags = result["tags"]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not isinstance(tags, list):
        raise ValueError("Tags must be a list.")

    tags = [str(tag).strip() for tag in tags]
    tags

    print(f"\nBatch {batch_number}: generated {len(tags)} tags")

    for i, tag in enumerate(tags, 1):
        print(f"  {i:02d}. {tag}")

    return tags


# ============================================================
# GENERATE ONE DOCUMENT
# ============================================================


def generate_document(tags, document_number):
    """
    Generate one realistic enterprise document and choose
    positive + hard-negative tags from the supplied taxonomy.
    """

    shuffled_tags = tags.copy()
    random.shuffle(shuffled_tags)

    tag_text = "\n".join(f"- {tag}" for tag in shuffled_tags)

    prompt = f"""
You are generating training data for an enterprise document
classification model.

Available document tags:

{tag_text}

Generate ONE realistic enterprise document.

Then classify that document using the available tags.

Requirements:

1. Generate realistic document text.
2. The text should be detailed enough to clearly represent
   an enterprise document.
3. Choose 1-3 POSITIVE tags.
4. Choose 3-6 NEGATIVE tags.
5. Negative tags must be HARD NEGATIVES whenever possible.

Hard negatives are tags that are semantically close to the
document but are still incorrect.

For example:

Document:
An employee onboarding procedure explaining how new employees
are added to internal systems.

Positive:
HR - Employee Procedures

Hard negatives:
HR - Employee Records
HR - Recruitment Documents
IT - Access Management Procedures

6. Every positive tag MUST be copied exactly from the
   available tag list.

7. Every negative tag MUST be copied exactly from the
   available tag list.

8. Do NOT invent tags.

9. Do NOT put the same tag in both positive and negative.

10. Do NOT repeat tags.

11. Positive and negative tags must contain only strings
    from the supplied taxonomy.

12. The document itself should NOT explicitly mention
    the classification tags.

13. Vary the document types and writing styles.

Examples of realistic documents:

- internal policies
- procedures
- invoices
- financial reports
- contracts
- engineering documentation
- incident reports
- HR documents
- marketing materials
- compliance reports
- technical guides
- meeting documents
- administrative documents

Return ONLY valid JSON in this format:

{{
    "text": "realistic enterprise document text here",
    "positive_tags": [
        "exact tag from the list"
    ],
    "negative_tags": [
        "exact tag from the list",
        "exact tag from the list",
        "exact tag from the list"
    ]
}}
"""

    for attempt in range(MAX_RETRIES):
        try:
            result = call_llm(prompt)

            validate_document(result, tags)

            return result

        except Exception as e:
            print(f"Document {document_number}: validation failed: {e}")

            if attempt < MAX_RETRIES - 1:
                time.sleep(1)

    raise RuntimeError(f"Could not generate document {document_number}")


# ============================================================
# VALIDATE DOCUMENT
# ============================================================


def validate_document(document, tags):

    required_keys = {"text", "positive_tags", "negative_tags"}

    if not required_keys.issubset(document.keys()):
        raise ValueError("Missing required JSON fields.")

    text = document["text"]
    positive = document["positive_tags"]
    negative = document["negative_tags"]

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    if not isinstance(text, str):
        raise ValueError("text must be a string.")

    if len(text.strip()) < 50:
        raise ValueError("Generated document is too short.")

    # --------------------------------------------------------
    # Positive tags
    # --------------------------------------------------------

    if not isinstance(positive, list):
        raise ValueError("positive_tags must be a list.")

    if not 1 <= len(positive) <= 3:
        raise ValueError("Number of positive tags must be 1-3.")

    # --------------------------------------------------------
    # Negative tags
    # --------------------------------------------------------

    if not isinstance(negative, list):
        raise ValueError("negative_tags must be a list.")

    if not 3 <= len(negative) <= 6:
        raise ValueError("Number of negative tags must be 3-6.")

    # --------------------------------------------------------
    # Exact tag membership
    # --------------------------------------------------------

    tag_set = set(tags)

    if not set(positive).issubset(tag_set):
        invalid = set(positive) - tag_set

        raise ValueError(f"Invalid positive tags: {invalid}")

    if not set(negative).issubset(tag_set):
        invalid = set(negative) - tag_set

        raise ValueError(f"Invalid negative tags: {invalid}")

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    if len(positive) != len(set(positive)):
        raise ValueError("Duplicate positive tags.")

    if len(negative) != len(set(negative)):
        raise ValueError("Duplicate negative tags.")

    # --------------------------------------------------------
    # Overlap check
    # --------------------------------------------------------

    overlap = set(positive) & set(negative)

    if overlap:
        raise ValueError(f"Positive/negative overlap: {overlap}")


# ============================================================
# EXPAND DOCUMENT INTO PAIRS
# ============================================================


def expand_document(document):

    rows = []

    text = document["text"]

    # Positive pairs
    for tag in document["positive_tags"]:
        rows.append({"text": text, "tag": tag, "score": 1.0})

    # Negative pairs
    for tag in document["negative_tags"]:
        rows.append({"text": text, "tag": tag, "score": 0.0})

    return rows


# ============================================================
# CHECKPOINT PATH
# ============================================================


def checkpoint_path(batch_number):

    return CHECKPOINT_DIR / (f"batch_{batch_number:02d}.parquet")


# ============================================================
# GENERATE ONE 1,000 DOCUMENT BATCH
# ============================================================


def generate_batch(batch_number):

    path = checkpoint_path(batch_number)

    # --------------------------------------------------------
    # Already generated?
    # --------------------------------------------------------

    if path.exists():
        print(f"\nBatch {batch_number} already exists.")

        df = pd.read_parquet(path)

        return df

    # --------------------------------------------------------
    # Generate 40 tags ONCE
    # --------------------------------------------------------

    tags = generate_tags(batch_number)

    # Save taxonomy too
    tags_path = CHECKPOINT_DIR / f"batch_{batch_number:02d}_tags.json"

    with open(tags_path, "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Generate 1,000 documents
    # --------------------------------------------------------

    all_rows = []

    start_document = (batch_number - 1) * DOCUMENTS_PER_BATCH + 1

    end_document = batch_number * DOCUMENTS_PER_BATCH

    print(f"\nGenerating documents {start_document:,} - {end_document:,}")

    for document_number in range(start_document, end_document + 1):
        document = generate_document(tags, document_number)

        rows = expand_document(document)

        all_rows.extend(rows)

        # Progress
        if (document_number - start_document + 1) % 10 == 0:
            print(
                f"Batch {batch_number}: "
                f"{document_number - start_document + 1:,}"
                f"/{DOCUMENTS_PER_BATCH:,} documents"
            )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    df = pd.DataFrame(all_rows)

    df["text"] = df["text"].astype(str)
    df["tag"] = df["tag"].astype(str)
    df["score"] = df["score"].astype("float32")

    df.to_parquet(path, index=False)

    print(f"\nSaved checkpoint:\n{path}\nRows: {len(df):,}")

    return df


# ============================================================
# GENERATE COMPLETE DATASET
# ============================================================


def generate_dataset():

    batch_dfs = []

    for batch_number in range(1, NUM_BATCHES + 1):
        print("\n" + "=" * 70)
        print(f"BATCH {batch_number}/{NUM_BATCHES}")
        print("=" * 70)

        df = generate_batch(batch_number)

        batch_dfs.append(df)

    # --------------------------------------------------------
    # Combine all batches
    # --------------------------------------------------------

    print("\nCombining batches...")

    final_df = pd.concat(batch_dfs, ignore_index=True)

    final_df["text"] = final_df["text"].astype(str)
    final_df["tag"] = final_df["tag"].astype(str)
    final_df["score"] = final_df["score"].astype("float32")

    final_df.to_parquet(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)

    print(f"Documents: {TOTAL_DOCUMENTS:,}")

    print(f"Pairwise rows: {len(final_df):,}")

    print(f"Output: {OUTPUT_FILE}")

    print("\nExample:")

    print(final_df.head(10).to_string(index=False))

    return final_df


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    dataset_df = generate_dataset()

    # --------------------------------------------------------
    # Load using Hugging Face datasets
    # --------------------------------------------------------

    dataset = load_dataset("parquet", data_files=OUTPUT_FILE)

    print("\nHugging Face dataset:")
    print(dataset)
