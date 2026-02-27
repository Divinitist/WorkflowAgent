import numpy as np
import json
import faiss
import os
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(self, chunks_file, model_name='BAAI/bge-small-zh-v1.5'):
        self.model = SentenceTransformer(model_name)
        index_path = f'{chunks_file}.faiss'
        with open(chunks_file, 'r', encoding='utf-8') as file:
            self.chunks = json.load(file)
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"已加载 FAISS 索引，包含 {self.index.ntotal} 个向量。")
            return
        # 建立 FAISS 索引，FlatL2 是暴力搜索
        else:
            self.index = None
            # 用模型嵌入之后转成 np.array
            embeddings = self.model.encode([chunk['chunk_text'] for chunk in self.chunks], show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            faiss.write_index(self.index, index_path)
            print(f"FAISS 添加了 {self.index.ntotal} 个 {dimension} 维向量。")
    def retrieve(self, query, top_k=5):
        query_embedding = np.array(self.model.encode([query])).astype('float32')
        # distances 是基于 FlatL2 的距离，indices 是对应的向量索引
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            results.append({
                "chpt_id": self.chunks[idx]["chpt_id"],
                "chunk_id": self.chunks[idx]["chunk_id"],
                "score": float(distances[0][list(indices[0]).index(idx)]),
                "chunk_text": self.chunks[idx]["chunk_text"]
            })
        return results