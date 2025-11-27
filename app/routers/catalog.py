from fastapi import APIRouter, Depends, Query
from typing import Optional, List
import numpy as np
from app.services.embeddings import EmbeddingService, get_embedding_service

router = APIRouter()

def add_image_url(product: dict) -> dict:
    """Generate R2 image URL from product ID"""
    if product and product.get("id"):
        pid = str(product["id"]).zfill(10)
        folder = pid[:3]
        product["image_url"] = f"https://pub-7907e757920c43e5a62f414cfbe74387.r2.dev/{folder}/{pid}.jpg"
    return product

@router.get("/debug")
def debug(svc: EmbeddingService = Depends(get_embedding_service)):
    """Debug endpoint to check data loading"""
    pop_sample = svc.popular_products[:3] if svc.popular_products else []
    return {
        "num_products": len(svc.products),
        "num_popular": len(svc.popular_products),
        "popular_sample": pop_sample,
        "sample_product_ids": svc.products["id"].head(5).tolist()
    }

@router.post("/get_personalized_catalog")
def get_personalized_catalog(
    user_id: str,
    k: int = Query(default=20, le=100),
    category_filter: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    exclude_purchased: bool = True,
    svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Get personalized product catalog for a user.
    """
    if user_id not in svc.user_taste_vectors:
        products = []
        for item in svc.popular_products[:k]:
            pid = item['product_id'] if isinstance(item, dict) else item
            pid = str(pid).lstrip('0')
            prod = svc.get_product(pid)
            if prod:
                prod["affinity_score"] = item.get('affinity_score', 0.5) if isinstance(item, dict) else 0.5
                add_image_url(prod)
                products.append(prod)
        return {"products": products, "is_cold_start": True}
    
    taste_vector = svc.user_taste_vectors[user_id]
    indices, scores = svc.search_similar(taste_vector, k=k * 3)
    
    products = []
    for i, idx in enumerate(indices):
        pid = svc.idx_to_id[idx]
        prod = svc.get_product(pid)
        
        if prod is None:
            continue
        
        if category_filter and prod["category"] != category_filter:
            continue
        if price_min and prod["price"] < price_min:
            continue
        if price_max and prod["price"] > price_max:
            continue
        
        prod["affinity_score"] = float(scores[i])
        add_image_url(prod)
        products.append(prod)
        
        if len(products) >= k:
            break
    
    return {"products": products, "is_cold_start": False}


@router.post("/compute_taste_from_calibration")
def compute_taste_from_calibration(
    liked_ids: List[str],
    disliked_ids: List[str] = [],
    svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Compute taste vector from onboarding calibration (20 swipes).
    """
    if not liked_ids:
        return {"error": "Need at least one liked product"}
    
    liked_vectors = []
    for pid in liked_ids:
        emb = svc.get_embedding(pid)
        if emb is not None:
            liked_vectors.append(emb)
    
    if not liked_vectors:
        return {"error": "No valid products found"}
    
    taste_vector = np.mean(liked_vectors, axis=0)
    
    if disliked_ids:
        disliked_vectors = []
        for pid in disliked_ids:
            emb = svc.get_embedding(pid)
            if emb is not None:
                disliked_vectors.append(emb)
        if disliked_vectors:
            dislike_avg = np.mean(disliked_vectors, axis=0)
            taste_vector = taste_vector - 0.3 * dislike_avg
    
    taste_vector = taste_vector / (np.linalg.norm(taste_vector) + 1e-8)
    
    indices, scores = svc.search_similar(taste_vector, k=20)
    
    products = []
    seen = set(liked_ids + disliked_ids)
    for i, idx in enumerate(indices):
        pid = svc.idx_to_id[idx]
        if pid in seen:
            continue
        prod = svc.get_product(pid)
        if prod:
            prod["affinity_score"] = float(scores[i])
            add_image_url(prod)
            products.append(prod)
        if len(products) >= 10:
            break
    
    return {
        "taste_vector": taste_vector.tolist(),
        "recommendations": products
    }


@router.get("/get_calibration_products")
def get_calibration_products(
    n: int = Query(default=20, le=50),
    svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Get diverse products for onboarding calibration.
    """
    categories = svc.products["product_type_name"].value_counts().head(10).index.tolist()
    
    products = []
    per_category = max(2, n // len(categories))
    
    for cat in categories:
        cat_prods = svc.products[svc.products["product_type_name"] == cat].sample(
            min(per_category, len(svc.products[svc.products["product_type_name"] == cat]))
        )
        for _, row in cat_prods.iterrows():
            prod = {
                "id": row["id"],
                "name": row["name"],
                "category": row["product_type_name"],
                "color": row["colour_group_name"],
                "price": row.get("price", 49.99),
                "image_url": None
            }
            add_image_url(prod)
            products.append(prod)
        if len(products) >= n:
            break
    
    return {"products": products[:n]}