import torch
from config import Config
from sentence_transformers import SentenceTransformer

device = Config.device

_embedding_model = SentenceTransformer("minishlab/potion-multilingual-128M")
EMBEDDING_DIM = _embedding_model.get_embedding_dimension()
_embedding_model.to(device)
_embedding_model.eval()
_embedding_model.requires_grad_(False)

_tokenizer = _embedding_model[0].tokenizer
_tokenizer.enable_padding()


def get_token_embeddings(
    texts: list[str],
) -> tuple[torch.FloatTensor, torch.BoolTensor]:

    encodings = _tokenizer.encode_batch(texts)

    token_ids = torch.tensor(
        [e.ids for e in encodings], dtype=torch.long, device=device
    )

    attention_mask = torch.tensor(
        [e.attention_mask for e in encodings], dtype=torch.bool, device=device
    )

    with torch.no_grad():
        token_embeddings = _embedding_model[0].embedding.weight[token_ids]

    return token_embeddings, attention_mask


if __name__ == "__main__":
    texts = ["hi this is working", "hi this is another"]
    token_embeddings, attention_mask = get_token_embeddings(texts)
    print(token_embeddings.requires_grad)
