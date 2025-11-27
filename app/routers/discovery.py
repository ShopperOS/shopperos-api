from fastapi import APIRouter, Depends
import numpy as np
from app.services.embeddings import EmbeddingService, get_embedding_service

router = APIRouter()

@router.get("/get_discovery_feed")
def get_discovery_feed(
    user_id: str,
    svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Get discovery feed with personalized sections.
    """
    sections = []
    
    # Section 1: Just For Today (personalized with variety)
    just_for_today = []
    if user_id in svc.user_taste_vectors:
        taste = svc.user_taste_vectors[user_id]
        # Add small noise for daily variety
        noise = np.random.randn(len(taste)) * 0.1
        varied_taste = taste + noise
        varied_taste = varied_taste / (np.linalg.norm(varied_taste) + 1e-8)
        
        indices, scores = svc.search_similar(varied_taste, k=10)
        for i, idx in enumerate(indices[:5]):
            pid = svc.idx_to_id[idx]
            prod = svc.get_product(pid)
            if prod:
                prod["affinity_score"] = float(scores[i])
                just_for_today.append(prod)
    else:
        # Cold start
        for pid in svc.popular_products[:5]:
            prod = svc.get_product(pid)
            if prod:
                just_for_today.append(prod)
    
    sections.append({
        "title": "Just For Today",
        "type": "personalized",
        "products": just_for_today
    })
    
    # Section 2: New Arrivals in Your Style
    new_arrivals = []
    if user_id in svc.user_taste_vectors:
        taste = svc.user_taste_vectors[user_id]
        indices, scores = svc.search_similar(taste, k=20)
        
        seen = set(p["id"] for p in just_for_today)
        for i, idx in enumerate(indices):
            pid = svc.idx_to_id[idx]
            if pid in seen:
                continue
            prod = svc.get_product(pid)
            if prod:
                prod["affinity_score"] = float(scores[i])
                new_arrivals.append(prod)
                seen.add(pid)
            if len(new_arrivals) >= 5:
                break
    
    sections.append({
        "title": "New Arrivals in Your Style",
        "type": "new_arrivals",
        "products": new_arrivals
    })
    
    # Section 3: Trending
    trending = []
    trending_ids = svc.trending.get("7d", [])[:10]
    for item in trending_ids:
        pid = item if isinstance(item, str) else item.get("product_id", item)
        prod = svc.get_product(str(pid))
        if prod:
            trending.append(prod)
        if len(trending) >= 5:
            break
    
    sections.append({
        "title": "Trending",
        "type": "trending",
        "products": trending
    })
    
    return {"sections": sections}
