'''
Generate embeddings for dataset using sentence-transformer
'''
import json
import os
import numpy as np
import time
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch

from file_utils import load_data, save_data

def generate_embeddings(
        emb_model:str, 
        input_path:str,
        save_emb_path:str,
        batch_size:int=256,
        test_tag:str="text"
    ):
    input_data = load_data(input_path)
    texts = [test_tag for x in input_data]
    num = len(texts)

    model = SentenceTransformer(emb_model)

    embeddings = []
    bar = tqdm(range(0, num, batch_size), desc='generate embeddings', ncols=80)
    for i in range(0, num, batch_size):
        embeddings += model.encode(texts[i:i+batch_size]).tolist()
        bar.update(1)
    embeddings = np.array(embeddings)

    # Normalize
    embeddings = torch.tensor(embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=1).numpy()

    np.save(save_emb_path, embeddings)
    print(f"Embeddings saved to {save_emb_path}")
    print(f"Shape: {embeddings.shape}")



if __name__ == "__main__":
    # TODO: Please download "gte-large-zh" model first then put it in the corresponding dir
    # generate embedding for boson test set
    generate_embeddings(
        emb_model="models/thenlper/gte-large-zh",
        input_path=f"data/benchmark_data/boson/test.jsonl",
        save_emb_path=f"data/benchmark_data/boson/test_GTElargeEmb_text.npy",
        batch_size=32,
        test_tag="sentence"
    )
    generate_embeddings(
        emb_model="models/thenlper/gte-large-zh",
        input_path=f"data/benchmark_data/boson/train.jsonl",
        save_emb_path=f"data/benchmark_data/boson/test_GTElargeEmb_text.npy",
        batch_size=32,
        test_tag="sentence"
    )

    
    

