# Handles reading data files, chunking, embedding via sentence-transformers, and FAISS index creation/loading.

import os
import json
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

class Ingestor:
    def __init__(self, data_file: str = "../Data/innovate_inc_q3_2025.txt",
                 faiss_dir: str = "../Vectors",
                 embedding_model_name: str = "intfloat/e5-large-v2",
                 chunk_size: int = 512,
                 chunk_overlap: float = 0.2):
        self.data_file = data_file
        self.faiss_dir = faiss_dir
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = None
        self.index = None
        self.metadata = []  # list of dicts: {"id": int, "text": str, "source": filename}
        self.dimension = None
        os.makedirs(self.faiss_dir, exist_ok=True)
        self.index_path = os.path.join(self.faiss_dir, "vectors.index")
        self.meta_path = os.path.join(self.faiss_dir, "metadata.json")

    def load_model(self):
        if self.model is None:
            print(f"[Ingestor] Loading embedding model {self.embedding_model_name} ...")
            self.model = SentenceTransformer(self.embedding_model_name)
            # define embedding dimension
            # run dummy to get dimension
            vec = self.model.encode("test")
            self.dimension = int(vec.shape[0])

    def chunk_text(self, text: str) -> List[str]:
        tokens = text.split()
        step = int(self.chunk_size * (1 - self.chunk_overlap))
        chunks = []
        for i in range(0, max(1, len(tokens) - 1), step):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(" ".join(chunk))
            if i + self.chunk_size >= len(tokens):
                break
        return chunks

    def ensure_index(self):
        # If index exists, load; else create by ingesting default file
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.load_index()
                print("[Ingestor] Loaded existing FAISS index.")
                return
            except Exception as e:
                print("[Ingestor] Failed to load index:", e)
        # Else create
        print("[Ingestor] Creating FAISS index from data file:", self.data_file)
        self.load_model()
        self.create_index_from_file(self.data_file)

    def create_index_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = self.chunk_text(text)
        print(f"[Ingestor] Chunked into {len(chunks)} chunks.")
        embeddings = []
        metas = []
        for i, chunk in enumerate(tqdm(chunks, desc="Embedding")):
            vec = self.model.encode(chunk)
            embeddings.append(vec)
            metas.append({"id": i, "text": chunk, "source": os.path.basename(filepath)})
        embeddings = np.vstack(embeddings).astype("float32")
        self.dimension = embeddings.shape[1]
        # Create FAISS index
        index = faiss.IndexFlatIP(self.dimension)  # inner product similarity; we will normalize
        # normalize embeddings
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metas, f, ensure_ascii=False, indent=2)
        self.index = index
        self.metadata = metas
        print("[Ingestor] FAISS index created and saved.")

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        # attempt to set dimension
        if self.metadata:
            # use first embedding via model to get dimension if model available
            if self.model is None:
                self.load_model()
            dim = self.dimension or self.model.encode("test").shape[0]
            self.dimension = dim

    def ingest_and_append_file(self, filepath: str):
        # load existing index; append new embeddings and update metadata
        if self.index is None:
            self.load_index()
        if self.model is None:
            self.load_model()
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = self.chunk_text(text)
        embeddings = []
        new_metas = []
        start_id = len(self.metadata)
        for i, chunk in enumerate(tqdm(chunks, desc="Embedding new file")):
            vec = self.model.encode(chunk)
            embeddings.append(vec)
            new_metas.append({"id": start_id + i, "text": chunk, "source": os.path.basename(filepath)})
        embeddings = np.vstack(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        # Append to existing index
        self.index.add(embeddings)
        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        self.metadata.extend(new_metas)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"[Ingestor] Appended {len(new_metas)} chunks from {filepath} to index.")

    def retrieve(self, query: str, top_k: int = 4) -> List[dict]:
        """
        Return list of metadata dicts with keys: id, text, score
        """
        if self.model is None:
            self.load_model()
        vec = self.model.encode(query).astype("float32")
        faiss.normalize_L2(vec.reshape(1, -1))
        D, I = self.index.search(vec.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results
