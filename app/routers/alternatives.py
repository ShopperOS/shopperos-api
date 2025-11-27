from fastapi import APIRouter, Depends, Query
import numpy as np
from app.services.embeddings import EmbeddingService, get_embedding_service

router = APIRouter()

@router.post("/get_alternatives")
def get_alternatives(
    product_id: str,
    k: int = Query(default=10, le=50),
    svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Get similar alternative products with reasons.
    """
    # Get source product
    source = svc.get_product(product_id)
    if source is None:
        return {"error": "Product not found"}
    
    # Get embedding
    emb = svc.get_embedding(product_id)
    if emb is None:
        return {"error": "No embedding for product"}
    
    # Search similar
    indices, scores = svc.search_similar(emb, k=k * 2)
    
    alternatives = []
    source_cat = source["category"]
    source_color = source["color"]
    
    for i, idx in enumerate(indices):
        pid = svc.idx_to_id[idx]
        
        # Skip self
        if pid == str(product_id).lstrip("0"):
            continue
        
        prod = svc.get_product(pid)
        if prod is None:
            continue
        
        # Generate reason
        reasons = []
        if prod["category"] == source_cat:
            reasons.append(f"Same style: {source_cat}")
        if prod["color"] == source_color:
            reasons.append(f"Same color: {source_color}")
        if not reasons:
            reasons.append("Similar style")
        
        prod["similarity"] = float(scores[i])
        prod["reasons"] = reasons
        alternatives.append(prod)
        
        if len(alternatives) >= k:
            break
    
    return {
        "source_product": source,
        "alternatives": alternatives
    }
