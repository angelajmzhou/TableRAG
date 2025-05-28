import os
import time
import urllib3
import sys
import faiss
import transformers
import warnings
import threading
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tqdm as tqdm
from collections import defaultdict
from transformers import AutoModel
import transformers
from utils.tool_utils import *
import ntlk
from more_itertools import chunked
import numpy as np
import pickle
from typing import Dict, List, Union, Tuple, Any
from utils.utils import read_plain_csv

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
device = "cuda:0"

class SemanticRetriever() :
    """
    Retrieving process, containing recall and rerank.
    """
    def __init__(self,
        chunks: List[str],
        chunk_index： Dict,
        llm_path: str = None,
        reranker_path: str = None,
        save_path: str = "./retrieval_result/embedding.pkl"
    ) -> None:
    self.embedding_model = Embedder(llm_path)
    self.reranker = Reranker(reranker_path)

    if os.path.exists(save_path) :
        doc_embeddings, self.chunks, self.chunk_file_index = self.load_embeddings(save_path)
        self.chunk_index = {idx: ch for idx, ch in enumerate(self.chunks)}
    else :
        self.chunks = chunks
        self.chunk_index = chunk_index
        self.chunk_file_index = chunk_file_index
        doc_embeddings = self.embed_doc(chunks, save_path=save_path)
    
    self.thread_local = threading.local()
    self.index_lock = threading.RLock()

    print("embedding size", doc_embeddings.shape)
    self.res = faiss.StandardGpuResources()
    self.index_IP = self.build_index(doc_embeddings)

    def embed_doc(self, chunks: List[str], batch_size: int = 512, save_path: str = None) -> Any :
        """
        Embed documents in batches for improved performance

        Args:
            chunks: List of text chunks to embed

        Returns:
            np.ndarray: Array of document embeddings
        """
        encode_vecs = []
        iterator = tqdm(range(0, len(chunks), batch_size)) if len(chunks) >= 100 \
            else range(0, len(chunks), batch_size)

        for i in iterator :
            batch = chunks[i: i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)

            if len(batch) == 1 :
                encode_vecs.append(batch_embeddings)
            else :
                encode_vecs.extend(batch_embeddings) 
        