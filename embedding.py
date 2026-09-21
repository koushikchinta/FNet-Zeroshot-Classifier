import os

import torch
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# DDP device
# --------------------------------------------------

if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

_embedding_model = SentenceTransformer(
    "minishlab/potion-multilingual-128M"
)

_embedding_model.to(device)
_embedding_model.eval()
_embedding_model.requires_grad_(False)


EMBEDDING_DIM = _embedding_model.get_embedding_dimension()


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

_tokenizer = _embedding_model[0].tokenizer
_tokenizer.enable_padding()


# --------------------------------------------------
# Token embeddings
# --------------------------------------------------

def get_token_embeddings(
    texts: list[str],
) -> tuple[torch.FloatTensor, torch.BoolTensor]:

    encodings = _tokenizer.encode_batch(texts)

    token_ids = torch.tensor(
        [e.ids for e in encodings],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        [e.attention_mask for e in encodings],
        dtype=torch.bool,
        device=device,
    )

    with torch.no_grad():

        embedding_weights = (
            _embedding_model[0]
            .embedding
            .weight
        )

        token_embeddings = embedding_weights[
            token_ids
        ]

    return token_embeddings, attention_mask