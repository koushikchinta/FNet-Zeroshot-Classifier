from model import Model
from config import Config
from embedding import EMBEDDING_DIM, get_token_embeddings
import torch
import os
from train import MODEL_STATE_DICT

def infer(sentence1, sentence2) -> float:
    checkpoint = torch.load(os.path.join(Config.train.checkpoint_dir,"best.pt"), Config.device)
    model = Model(EMBEDDING_DIM, Config.model.num_encoding_layers, Config.model.dropout)
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])

    model.eval()

    with torch.no_grad():
        x, x_mask = get_token_embeddings([sentence1])
        y, y_mask = get_token_embeddings([sentence2])
        logits = model(x, x_mask, y, y_mask)
        return logits[0].item()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence1")
    parser.add_argument("sentence2")

    args = parser.parse_args()

    print(infer(args.sentence1, args.sentence2))