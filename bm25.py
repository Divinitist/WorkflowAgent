import jieba
from rank_bm25 import BM25Okapi
import numpy as np
import json

class BM25Retriever:
    def __init__(self, chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as file:
            self.chunks = json.load(file)
        self.tokenized_corpus = [self.tokenize(chunk['chunk_text']) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize(self, text):
        return [t for t in jieba.cut(text.replace('\n', ' ')) if len(t.strip()) > 1]

    def retrieve(self, query, top_k=5):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 获取得分最高的top_k个索引
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_n_indices:
             results.append({
                 "chpt_id": self.chunks[idx]["chpt_id"],
                 "chunk_id": self.chunks[idx]["chunk_id"],
                 "score": scores[idx],
                 "chunk_text": self.chunks[idx]["chunk_text"]
             })
        return results