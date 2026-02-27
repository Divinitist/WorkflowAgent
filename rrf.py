def rrf_fusion(results_list, k=60, top_k=5):
    scores = {}
    visit = set()
    fusion_results = []
    def update_scores(results):
        # rank从0开始，会不会有问题？
        for rank, res in enumerate(results):
            key = (res['chpt_id'], res['chunk_id'])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            if key not in visit:
                visit.add(key)
                fusion_results.append({
                    "chpt_id": res['chpt_id'],
                    "chunk_id": res['chunk_id'],
                    "score": -1,  # 占位，稍后更新
                    "chunk_text": res['chunk_text']
                })
    for results in results_list:
        update_scores(results)
    for res in fusion_results:
        key = (res['chpt_id'], res['chunk_id'])
        res['score'] = scores[key]
    return sorted(fusion_results, key=lambda x: x['score'], reverse=True)[:top_k]