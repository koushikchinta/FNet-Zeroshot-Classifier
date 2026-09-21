from model import Model
from embedding import get_token_embeddings, EMBEDDING_DIM

from dataset import (
    train_dataset,
    validation_dataset,
    test_dataset,
    SENTENCE1_COLUMN,
    SENTENCE2_COLUMN,
    SCORE,
    CONTRADICTION,
    ENTAILMENT,
)

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from logger import dict_log

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)

import math
import os
import logging


MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
SCHEDULER_STATE_DICT = "scheduler_state_dict"
EPOCH = "epoch"
BEST_VAL_LOSS = "best_val_loss"


def setup_ddp():
    """
    Initialize DDP using torchrun environment variables.
    """

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def collate_fn(batch):

    premises = [
        item[SENTENCE1_COLUMN]
        for item in batch
    ]

    hypotheses = [
        item[SENTENCE2_COLUMN]
        for item in batch
    ]

    labels = torch.tensor(
        [item[SCORE] for item in batch],
        dtype=torch.float,
    )

    return premises, hypotheses, labels


def is_main_process(rank):
    return rank == 0


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
):

    model_state_dict = model.module.state_dict()

    checkpoint = {
        EPOCH: epoch,
        MODEL_STATE_DICT: model_state_dict,
        OPTIMIZER_STATE_DICT: optimizer.state_dict(),
        SCHEDULER_STATE_DICT: scheduler.state_dict(),
        BEST_VAL_LOSS: best_val_loss,
    }

    torch.save(checkpoint, path)


def train():

    rank, local_rank, world_size = setup_ddp()

    device = torch.device(
        f"cuda:{local_rank}"
    )

    logger = Config.logger

    if is_main_process(rank):
        logger.info(
            f"DDP initialized with {world_size} GPUs"
        )

    # --------------------------------------------------
    # Dataset samplers
    # --------------------------------------------------

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    train_dataset_loader = DataLoader(
        train_dataset,
        batch_size=Config.train.train_batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    validation_dataset_loader = DataLoader(
        validation_dataset,
        batch_size=Config.train.validation_batch_size,
        sampler=validation_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = Model(
        EMBEDDING_DIM,
        Config.model.num_encoding_layers,
        dropout=Config.model.dropout,
    )

    model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=Config.train.base_lr,
        weight_decay=Config.train.weight_decay,
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    num_epochs = Config.train.epochs

    steps_per_epoch = len(train_dataset_loader)

    total_steps = (
        steps_per_epoch * num_epochs
    )

    warmup_steps = int(
        0.06 * total_steps
    )

    def lr_lambda(step):

        if step < warmup_steps:
            return step / max(
                1,
                warmup_steps,
            )

        progress = (
            step - warmup_steps
        ) / max(
            1,
            total_steps - warmup_steps,
        )

        return 0.5 * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda,
    )

    # --------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------

    epoch_offset = 1
    best_val_loss = float("inf")

    if Config.train.initial_checkpoint:

        if is_main_process(rank):
            logger.info(
                f"Loading checkpoint - "
                f"{Config.train.initial_checkpoint}"
            )

        checkpoint = torch.load(
            Config.train.initial_checkpoint,
            map_location=device,
        )

        epoch_offset = (
            checkpoint[EPOCH] + 1
        )

        model.module.load_state_dict(
            checkpoint[MODEL_STATE_DICT]
        )

        optimizer.load_state_dict(
            checkpoint[OPTIMIZER_STATE_DICT]
        )

        scheduler.load_state_dict(
            checkpoint[SCHEDULER_STATE_DICT]
        )

        best_val_loss = checkpoint.get(
            BEST_VAL_LOSS,
            float("inf"),
        )

        if is_main_process(rank):

            dict_log(
                {
                    "Epoch": checkpoint[EPOCH],
                    "Validation Loss": best_val_loss,
                },
                logging.INFO,
                heading="Checkpoint metrics",
            )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    loss_fn = nn.MSELoss()

    os.makedirs(
        Config.train.checkpoint_dir,
        exist_ok=True,
    )

    if is_main_process(rank):

        logger.info(
            "==== Training started ===="
        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(
        epoch_offset,
        epoch_offset + num_epochs,
    ):

        train_sampler.set_epoch(epoch)

        model.train()

        train_loss_sum = 0.0
        train_examples = 0

        for X, Y, score in train_dataset_loader:

            x, x_mask = get_token_embeddings(X)
            y, y_mask = get_token_embeddings(Y)

            score = score.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x,
                x_mask,
                y,
                y_mask,
            )

            loss = loss_fn(
                logits.squeeze(-1),
                score.float(),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            batch_size = score.size(0)

            train_loss_sum += (
                loss.item() * batch_size
            )

            train_examples += batch_size

        # --------------------------------------------------
        # Aggregate training loss across GPUs
        # --------------------------------------------------

        train_stats = torch.tensor(
            [
                train_loss_sum,
                train_examples,
            ],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            train_stats,
            op=dist.ReduceOp.SUM,
        )

        train_loss = (
            train_stats[0]
            / train_stats[1]
        ).item()

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        val_loss_sum = 0.0
        val_examples = 0

        validation_sampler.set_epoch(epoch)

        with torch.no_grad():

            for X, Y, score in validation_dataset_loader:

                x, x_mask = get_token_embeddings(X)
                y, y_mask = get_token_embeddings(Y)

                score = score.to(
                    device,
                    non_blocking=True,
                )

                logits = model(
                    x,
                    x_mask,
                    y,
                    y_mask,
                )

                loss = loss_fn(
                    logits.squeeze(-1),
                    score.float(),
                )

                batch_size = score.size(0)

                val_loss_sum += (
                    loss.item() * batch_size
                )

                val_examples += batch_size

        # --------------------------------------------------
        # Aggregate validation loss
        # --------------------------------------------------

        val_stats = torch.tensor(
            [
                val_loss_sum,
                val_examples,
            ],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            val_stats,
            op=dist.ReduceOp.SUM,
        )

        val_loss = (
            val_stats[0]
            / val_stats[1]
        ).item()

        current_lr = (
            scheduler.get_last_lr()[0]
        )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        if is_main_process(rank):

            dict_log(
                {
                    "Epoch": epoch,
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                    "lr": current_lr,
                },
                logging.INFO,
                "Metrics",
                sep=" | ",
            )

            # --------------------------------------------------
            # Last checkpoint
            # --------------------------------------------------

            last_path = os.path.join(
                Config.train.checkpoint_dir,
                "last.pt",
            )

            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
            )

            logger.info(
                "Saved current checkpoint"
            )

            # --------------------------------------------------
            # Best checkpoint
            # --------------------------------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_path = os.path.join(
                    Config.train.checkpoint_dir,
                    "best.pt",
                )

                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                )

                logger.info(
                    "Saved current best checkpoint"
                )

    if is_main_process(rank):

        logger.info(
            "==== Training Completed ===="
        )

    cleanup_ddp()


def test():

    device = torch.device(
        Config.device
    )

    logger = Config.logger

    model = Model(
        EMBEDDING_DIM,
        Config.model.num_encoding_layers,
        dropout=Config.model.dropout,
    )

    checkpoint = torch.load(
        os.path.join(
            Config.train.checkpoint_dir,
            "best.pt",
        ),
        map_location=device,
    )

    dict_log(
        {
            "Epoch": checkpoint[EPOCH],
            "Validation Loss": checkpoint[BEST_VAL_LOSS],
        },
        logging.INFO,
        heading="Best Checkpoint metrics",
    )

    model.load_state_dict(
        checkpoint[MODEL_STATE_DICT]
    )

    model.to(device)
    model.eval()

    test_dataset_loader = DataLoader(
        test_dataset,
        batch_size=Config.test.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loss_sum = 0.0
    test_examples = 0

    test_targets = []
    test_predictions = []

    with torch.no_grad():

        for X, Y, score in test_dataset_loader:

            x, x_mask = get_token_embeddings(X)
            y, y_mask = get_token_embeddings(Y)

            score = score.to(
                device,
                non_blocking=True,
            )

            logits = model(
                x,
                x_mask,
                y,
                y_mask,
            )

            labels = torch.where(
                logits <= 0.33,
                2,
                torch.where(
                    (logits < 0.67)
                    & (logits > 0.33),
                    1,
                    0,
                ),
            )

            score = torch.where(
                score == CONTRADICTION,
                2,
                torch.where(
                    score == ENTAILMENT,
                    0,
                    1,
                ),
            )

            batch_size = score.size(0)

            test_loss_sum += nn.functional.mse_loss(
                logits.squeeze(-1),
                score.float(),
                reduction="sum",
            ).item()

            test_targets.append(
                score.cpu()
            )

            test_predictions.append(
                labels.squeeze(-1).cpu()
            )

            test_examples += batch_size

    targets = torch.cat(
        test_targets
    ).numpy()

    predictions = torch.cat(
        test_predictions
    ).numpy()

    dict_log(
        {
            "Test Loss": (
                test_loss_sum
                / max(1, test_examples)
            ),
            "Test Accuracy": accuracy_score(
                targets,
                predictions,
            ),
            "Test Recall (Macro)": recall_score(
                targets,
                predictions,
                average="micro",
                zero_division=0,
            ),
            "Test F1 Score (Macro)": f1_score(
                targets,
                predictions,
                average="micro",
                zero_division=0,
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
            target_names=[
                "0",
                "1",
                "2",
            ],
            zero_division=0,
        ),
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.test:
        test()
    else:
        train()