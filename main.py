from model import Model
from embedding import get_token_embeddings, EMBEDDING_DIM

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", "cp")
