from model import Model
from embedding import get_token_embeddings, EMBEDDING_DIM
from dataset import train_dataset, validation_dataset
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from config import Config
from torch.utils.data import DataLoader
import torch
import math
import os


MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
EPOCH = "epoch"
BEST_VAL_LOSS = "best_val_loss"


def train() -> None:

    train_dataset_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=Config.train.train_batch_size
    )

    validation_dataset_loader = DataLoader(
        validation_dataset, shuffle=False, batch_size=Config.train.validation_batch_size
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
        checkpoint = torch.load(
            Config.train.initial_checkpoint, map_location=Config.device
        )
        epoch_offset = checkpoint[EPOCH] + 1
        MODEL.load_state_dict(checkpoint[MODEL_STATE_DICT])
        MODEL.to(Config.device)
        OPTIMIZER.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT])
        best_val_loss = checkpoint.get(BEST_VAL_LOSS, float("inf"))
    else:
        MODEL.to(Config.device)

    os.makedirs(Config.train.checkpoint_dir, exist_ok=True)

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
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), score.float())
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
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), score.float()
                )

                batch_size = score.size(0)
                val_loss_sum += loss.item() * batch_size
                val_examples += batch_size

        val_loss = val_loss_sum / max(1, val_examples)
        current_lr = SCHEDULER.get_last_lr()[0]

        print(
            f"epoch {epoch} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | lr {current_lr:.6f}"
        )

        checkpoint = {
            EPOCH: epoch,
            MODEL_STATE_DICT: MODEL.state_dict(),
            OPTIMIZER_STATE_DICT: OPTIMIZER.state_dict(),
            BEST_VAL_LOSS: best_val_loss,
        }

        last_path = os.path.join(Config.train.checkpoint_dir, "last.pt")
        torch.save(checkpoint, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint[BEST_VAL_LOSS] = best_val_loss
            best_path = os.path.join(Config.train.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
