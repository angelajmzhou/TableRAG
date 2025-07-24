import os
import time
import urllib3
import sys
import faiss
import json
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
import nltk
from more_itertools import chunked
import numpy as np
import pickle
from typing import Dict, List, Union, Tuple, Any
from utils.utils import read_plain_csv

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
device = "cuda:0"

class SemanticRetriever :
    """
    Retrieving process, containing recall and rerank.
    """
    def __init__(
        self,
        chunks: List[str],
        chunk_index: Dict,
        chunk_file_index: Dict = None,
        llm_path: str = None,
        reranker_path: str = None,
        save_path: str = "./retrieval_result/embedding.pkl"
    ) -> None:
        #load models
        self.embedding_model = Embedder(llm_path)
        self.reranker = Reranker(reranker_path)
        #if doc embedding exists, restore it
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
        #add all doc embeddings to faiss index
        print("adding doc embeddings to faiss index")
        self.index_IP = self.build_faiss_index(doc_embeddings)
        print("embedding completed")

    def embed_doc(self, chunks: List[str], batch_size: int = 64, save_path: str = None) -> Any:
        """
        Embed documents in batches with memory exception handling

        Args:
            chunks: List of text chunks to embed
            batch_size: Initial batch size for encoding
            save_path: Optional path to save embeddings

        Returns:
            np.ndarray: Document embeddings
        """
        encode_vecs = []
        print("embed doc begin")
        iterator = tqdm(range(0, len(chunks), batch_size)) if len(chunks) >= 100 else range(0, len(chunks), batch_size)

        for i in iterator:
            retries = 0
            current_batch_size = batch_size
            while retries < 3:
                try:
                    batch = chunks[i : i + current_batch_size]
                    batch_embeddings = self.embedding_model.encode(batch)
                    encode_vecs.extend(batch_embeddings)
                    break  # success
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"[OOM] Retrying with smaller batch size: {current_batch_size}")
                    retries += 1
            else:
                raise RuntimeError(f"Failed to process batch {i}-{i+batch_size} after 3 retries")

        encode_vecs = np.array(encode_vecs)
        if len(encode_vecs.shape) == 3:
            encode_vecs = encode_vecs.reshape(-1, encode_vecs.shape[-1])

        if save_path:
            self.save_embeddings(encode_vecs, chunks, save_path)
            print("Embedding Vectors Saved.")

        return encode_vecs

    
    def save_embeddings(self, embeddings: Any, chunks: List[str], save_path: str) -> None :
        """
        Save embeddings and optionally the original chunks.

        Args:
            embeddings: numpy array of embeddings
            chunks: orignal text chunks
            save_path: path to save the embeddings
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.pkl') :
            data = {
                "embeddings": embeddings,
                "chunks": chunks,
                "chunk_file_index": self.chunk_file_index
            }
            with open(save_path, "wb") as f :
                pickle.dump(data, f)
            print(f"Embeddings and chunks saved to {save_path}")

    @staticmethod
    def load_embeddings(load_path: str = None) -> Tuple[Any, Any] :
        """
        Load saved embeddings.

        Args:
            load_path: Path to the saved embeddings.
        
        Returns:
            embeddings if .npy file, (embeddings, chunks) if .pkl file
        """
        if load_path.endswith('.npy') :
            return np.load(load_path)
        elif load_path.endswith('.pkl') :
            with open(load_path, 'rb') as f :
                data = pickle.load(f)
            return data['embeddings'], data['chunks'], data['chunk_file_index']
    
    #(num_vectors, dim)
    def build_faiss_index(self, dense_vector: Any) -> Any :
        print("Building Index.")
        with self.index_lock :
            _, dim = dense_vector.shape
            #builds dot product index for cosine similarity calculations
            index_IP = faiss.IndexFlatIP(dim)
            co = faiss.GpuClonerOptions()
            index_gpu = index_IP #cpu
            index_gpu = faiss.index_cpu_to_gpu(provider=self.res, device=0, index=index_IP, options=co) #gpu
            #add the vector to faiss index (vector data structure)
            index_gpu.add(dense_vector)
    
            return index_gpu

    def retrieve(self, query, recall_num, rerank_num):
        # Step 1: Perform dense retrieval to get candidate documents and their file names
        docs, ori_file_name = self.recall(query, recall_num)
        # Step 2: Rerank the retrieved documents using a reranker model
        reranked_docs, rerank_scores, filenames = self.rerank(query, docs, rerank_num, ori_file_name)
        # Step 3: Return top reranked documents, their scores, and corresponding filenames
        return reranked_docs, rerank_scores, filenames

    def recall(self, query: str, topn: int) -> List[str]:
        # Encode the query into a dense vector representation
        query_emb = self.embed_doc(query)
        # Search the FAISS index for topn most similar documents using inner product similarity
        with self.index_lock:
            D, I = self.index_IP.search(query_emb, topn)
        # Map the returned indices to actual document chunks and their source file names
        ori_docs = [self.chunk_index[i] for i in I[0]]
        ori_file_name = [self.chunk_file_index[i] for i in I[0]]
        return ori_docs, ori_file_name

    def rerank(self, query: str, docs: List[str], topn: int, ori_file_name: List[str]) -> Tuple[List[str], List[int]]:
        # Create query-document pairs for reranking
        pairs = [[query, d] for d in docs]
        # Compute similarity scores using a separate reranker (e.g., a cross-encoder)
        scores = self.reranker.compute_score(pairs)
        # Sort the documents based on the reranker scores in descending order
        sroted_pairs = sorted(zip(scores, docs, ori_file_name), reverse=True)
        # Unzip the sorted tuples into separate lists
        score_sorted, doc_sorted, filename_sorted = zip(*sroted_pairs)
        # Return the top-n reranked documents, scores, and filenames
        return doc_sorted[:topn], score_sorted[:topn], filename_sorted[:topn]


class MixedDocRetriever :
    def __init__(
        self,
        doc_dir_path: str,
        excel_dir_path: str,
        llm_path: str = None,
        reranker_path: str = None,
        save_path: str = "./retrieve_result/embedding.pkl"
    ) -> None:
        self.ori_documents = self.load_hybrid_dataset(doc_dir_path, excel_dir_path)
        #is a dict containing both .xlsx and .json files mapped to their markdown/plaintext contents

        print("Loading done.")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #create dict, mapping filenames to ~1000 char chunks of markdown/plaintext data by filename
        #"processing cases"
        doc_chunking_dict = self.doc_chunking()
        #returns mapping of chunk to index/filename by count (number wrt to all chunks)
        self.chunks, self.chunk_to_index, self.chunk_to_filename = self.build_index(doc_chunking_dict)

        #create semantic retriever (faiss)
        self.semantic_retriever = SemanticRetriever(
            chunks=self.chunks,
            chunk_index=self.chunk_to_index,
            chunk_file_index=self.chunk_to_filename,
            llm_path=llm_path,
            reranker_path=reranker_path,
            save_path=save_path
        )

    #store the .xlsx->markdown and .json versions in dict under respective file names
    def load_hybrid_dataset(self, doc_dir_path: str, excel_dir_path: str) -> Dict[str, List[str]] :
        all_docs = defaultdict(list)
        for file in tqdm(os.listdir(excel_dir_path)) :
            content = excel_to_markdown(os.path.join(excel_dir_path, file))
            excel_content = content
            #convert excel to markdown table, which has Table name: "{key}" at beginning
            all_docs[file] = excel_content
        
        for file in tqdm(os.listdir(doc_dir_path)) :
            with open(os.path.join(doc_dir_path, file), 'r', encoding="utf-8") as fin :
                data_split = json.load(fin)
            key_value_doc = ''
            for key, item in data_split.items() :
                #.items() gives a list of tuples, which is then split into key/value pairs and appended to doc.
                key_value_doc += f"{key} {item}\n"
            #so this un-JSONS the doc information
            all_docs[file] = key_value_doc
        return all_docs


    def build_index(self, chunking_dict: Dict) -> Tuple :
        flatten_chunks = []
        chunk_to_index = defaultdict(list)
        chunk_to_filename = defaultdict(str)
        cnt = 0
        for idx, (key, item) in enumerate(chunking_dict.items()) :
            #for each item in the dict, append its contents to flatten_chunks,
            #and map the chunk number to the index and filename.
            flatten_chunks += item
            for i in item :
                chunk_to_index[cnt] = i
                chunk_to_filename[cnt] = key
                cnt += 1
        return flatten_chunks, chunk_to_index, chunk_to_filename

    def nltk_single_doc_chunking(self, doc: str, key: str) -> Tuple[List[str], str] :
        #splits the text of the document into 1000 char chunks, ideally preserving sentences + paragraphs
        all_splits = self.text_splitter.split_text(doc)
        add_file_name_splits = []
        for split_chunk in all_splits :
            add_file_name_chunk = f"File name: {key}\n" + split_chunk
            add_file_name_splits.append(add_file_name_chunk)
        #return all chunks annotated by their origin, and the corresponding key
        return add_file_name_splits, key

    #"processing cases" progress bar
    def doc_chunking(self, max_workers=10) -> Dict :
        doc_chunkings = defaultdict(list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor :
            future_to_case = {executor.submit(self.nltk_single_doc_chunking, doc, key): doc for key, doc in self.ori_documents.items()}
            #for each item in hybrid dataset, both markdown tables (.xlsx) and plain text (.json), ignore keys in dict
            #as file info is already present in body of item
            for future in tqdm(as_completed(future_to_case), total=len(self.ori_documents), desc="Processing cases") :
                try :
                    single_doc_chunking, key = future.result()
                    #put it all in a large dict, with key being the filename
                    doc_chunkings[key] += single_doc_chunking
                except Exception as e :
                    print(f"Case processing generated exception: {e}")
        return doc_chunkings

    def retrieve(self, query: str, recall_nun: int = 50, rerank_num: int = 5) :
        return self.semantic_retriever.retrieve(query, recall_nun, rerank_num)





