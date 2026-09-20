from model import Model
from embedding import get_token_embeddings, EMBEDDING_DIM
from dataset import (
    train_dataset,
    validation_dataset,
    test_dataset,
    SENTENCE1_COLUMN,
    SENTENCE2_COLUMN,
    SCORE,
)
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from config import Config
from torch.utils.data import DataLoader
import torch
import math
import os
import logging
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from logger import dict_log


MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
SCHEDULER_STATE_DICT = "scheduler_state_dict"
EPOCH = "epoch"
BEST_VAL_LOSS = "best_val_loss"


def collate_fn(batch):
    premises = [item[SENTENCE1_COLUMN] for item in batch]
    hypotheses = [item[SENTENCE2_COLUMN] for item in batch]
    labels = torch.tensor(
        [item[SCORE] for item in batch],
        dtype=torch.float,
    )

    return premises, hypotheses, labels

def test():
    logger = Config.logger
    MODEL = Model(
            EMBEDDING_DIM,
            Config.model.num_encoding_layers,
            dropout=Config.model.dropout,
    )

    checkpoint = torch.load(os.path.join(Config.train.checkpoint_dir, "best.pt"), map_location=Config.device)
    dict_log(
        {"Epoch": checkpoint[EPOCH], "Validation Loss": checkpoint[BEST_VAL_LOSS]},
        logging.INFO,
        heading="Best Checkpoint metrics",
    )

    MODEL.load_state_dict(checkpoint[MODEL_STATE_DICT])

    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=Config.test.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    MODEL.eval()
    test_loss_sum = 0.0
    test_examples = 0
    test_targets = []
    test_predictions = []
    with torch.no_grad():
        for X, Y, score in test_dataset_loader:
            x, x_mask = get_token_embeddings(X)
            y, y_mask = get_token_embeddings(Y)
            score = score.to(Config.device, non_blocking=True)

            logits = MODEL(x, x_mask, y, y_mask)
            labels = torch.where(
                logits <= 0.33,
                2,
                torch.where((logits < 0.67) & (logits > 0.33), 1, 0),
            )
            batch_size = score.size(0)
            test_loss_sum += nn.functional.mse_loss(
                logits.squeeze(-1), score.float(), reduction="sum"
            ).item()
            test_targets.append(score.cpu())
            test_predictions.append(labels.squeeze(-1).cpu())
            test_examples += batch_size

    targets = torch.cat(test_targets).numpy()
    predictions = torch.cat(test_predictions).numpy()
    dict_log(
        {
            "Test Loss": test_loss_sum / max(1, test_examples),
            "Test Accuracy": accuracy_score(targets, predictions),
            "Test Recall (Macro)": recall_score(
                targets, predictions, average="micro", zero_division=0
            ),
            "Test F1 Score (Macro)": f1_score(
                targets, predictions, average="micro", zero_division=0
            ),
        },
        logging.INFO,
        heading="Test metrics",
    )
    logger.info(
        "Test classification report:\n%s",
        classification_report(
            targets,
            predictions,
            labels=[0, 1, 2],
            target_names=["0", "1", "2"],
            zero_division=0,
        ),
    )


def train() -> None:
    logger = Config.logger

    train_dataset_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=Config.train.train_batch_size,
        collate_fn=collate_fn,
    )

    validation_dataset_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=Config.train.validation_batch_size,
        collate_fn=collate_fn,
    )

    MODEL = Model(
        EMBEDDING_DIM,
        Config.model.num_encoding_layers,
        dropout=Config.model.dropout,
    )

    OPTIMIZER = AdamW(
        MODEL.parameters(),
        lr=Config.train.base_lr,
        weight_decay=Config.train.weight_decay,
    )

    num_epochs = Config.train.epochs
    steps_per_epoch = len(train_dataset_loader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    SCHEDULER = LambdaLR(OPTIMIZER, lr_lambda)

    epoch_offset = 1
    best_val_loss = float("inf")

    if Config.train.initial_checkpoint:
        logger.info(f"loading checkpoint - {Config.train.initial_checkpoint}")
        checkpoint = torch.load(
            Config.train.initial_checkpoint, map_location=Config.device
        )
        epoch_offset = checkpoint[EPOCH] + 1
        MODEL.load_state_dict(checkpoint[MODEL_STATE_DICT])
        MODEL.to(Config.device)
        OPTIMIZER.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT])
        SCHEDULER.load_state_dict(checkpoint[SCHEDULER_STATE_DICT])
        best_val_loss = checkpoint.get(BEST_VAL_LOSS, float("inf"))
        dict_log(
            {"Epoch": checkpoint[EPOCH], "Validation Loss": best_val_loss},
            logging.INFO,
            heading="Checkpoint metrics",
        )
    else:
        MODEL.to(Config.device)

    LOSS_FN = nn.MSELoss()

    os.makedirs(Config.train.checkpoint_dir, exist_ok=True)

    logger.info("==== Training started ====")
    for epoch in range(epoch_offset, epoch_offset + num_epochs):
        MODEL.train()

        train_loss_sum = 0.0
        train_examples = 0

        for X, Y, score in train_dataset_loader:
            x, x_mask = get_token_embeddings(X)
            y, y_mask = get_token_embeddings(Y)
            score = score.to(Config.device, non_blocking=True)

            OPTIMIZER.zero_grad()
            logits = MODEL(x, x_mask, y, y_mask)
            loss = LOSS_FN(logits.squeeze(-1), score.float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), 1.0)
            OPTIMIZER.step()
            SCHEDULER.step()

            batch_size = score.size(0)
            train_loss_sum += loss.item() * batch_size
            train_examples += batch_size

        train_loss = train_loss_sum / max(1, train_examples)

        MODEL.eval()
        val_loss_sum = 0.0
        val_examples = 0

        with torch.no_grad():
            for X, Y, score in validation_dataset_loader:
                x, x_mask = get_token_embeddings(X)
                y, y_mask = get_token_embeddings(Y)
                score = score.to(Config.device, non_blocking=True)

                logits = MODEL(x, x_mask, y, y_mask)
                loss = LOSS_FN(logits.squeeze(-1), score.float())

                batch_size = score.size(0)
                val_loss_sum += loss.item() * batch_size
                val_examples += batch_size

        val_loss = val_loss_sum / max(1, val_examples)
        current_lr = SCHEDULER.get_last_lr()[0]

        dict_log({"Epoch": epoch, "Training Loss" : train_loss, "Validation Loss": val_loss, "lr": current_lr}, logging.INFO, 'Metrics', sep=' | ')

        checkpoint = {
            EPOCH: epoch,
            MODEL_STATE_DICT: MODEL.state_dict(),
            OPTIMIZER_STATE_DICT: OPTIMIZER.state_dict(),
            SCHEDULER_STATE_DICT: SCHEDULER.state_dict(),
            BEST_VAL_LOSS: val_loss,
        }

        last_path = os.path.join(Config.train.checkpoint_dir, "last.pt")
        torch.save(checkpoint, last_path)
        logger.info("Saved current checkpoint")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint[BEST_VAL_LOSS] = best_val_loss
            best_path = os.path.join(Config.train.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            logger.info("Saved current best checkpoint")

    logger.info("==== Training Completed ====")
    logger.info("==== Running on test dataset ====")
    test()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true"
    )

    args = parser.parse_args()
    if args.test:
        test()
    else:
        train()
