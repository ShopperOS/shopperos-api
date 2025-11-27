from fastapi import APIRouter, Depends, Query
from typing import Optional, List
import numpy as np
from app.services.embeddings import EmbeddingService, get_embedding_service

router = APIRouter()

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
    # Check if user has taste vector
    if user_id not in svc.user_taste_vectors:
        # Cold start - return popular products
        products = []
        for pid in svc.popular_products[:k]:
            prod = svc.get_product(pid)
            if prod:
                prod["affinity_score"] = 0.5
                products.append(prod)
        return {"products": products, "is_cold_start": True}
    
    # Get user taste vector
    taste_vector = svc.user_taste_vectors[user_id]
    
    # Search similar products
    indices, scores = svc.search_similar(taste_vector, k=k * 3)
    
    products = []
    for i, idx in enumerate(indices):
        pid = svc.idx_to_id[idx]
        prod = svc.get_product(pid)
        
        if prod is None:
            continue
        
        # Apply filters
        if category_filter and prod["category"] != category_filter:
            continue
        if price_min and prod["price"] < price_min:
            continue
        if price_max and prod["price"] > price_max:
            continue
        
        prod["affinity_score"] = float(scores[i])
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
    Returns personalized recommendations for the new user.
    """
    if not liked_ids:
        return {"error": "Need at least one liked product"}
    
    # Get embeddings for liked items
    liked_vectors = []
    for pid in liked_ids:
        emb = svc.get_embedding(pid)
        if emb is not None:
            liked_vectors.append(emb)
    
    if not liked_vectors:
        return {"error": "No valid products found"}
    
    # Compute taste as average of likes
    taste_vector = np.mean(liked_vectors, axis=0)
    
    # Subtract dislikes if any
    if disliked_ids:
        disliked_vectors = []
        for pid in disliked_ids:
            emb = svc.get_embedding(pid)
            if emb is not None:
                disliked_vectors.append(emb)
        if disliked_vectors:
            dislike_avg = np.mean(disliked_vectors, axis=0)
            taste_vector = taste_vector - 0.3 * dislike_avg
    
    # Normalize
    taste_vector = taste_vector / (np.linalg.norm(taste_vector) + 1e-8)
    
    # Get recommendations
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
    # Sample across categories
    categories = svc.products["product_type_name"].value_counts().head(10).index.tolist()
    
    products = []
    per_category = max(2, n // len(categories))
    
    for cat in categories:
        cat_prods = svc.products[svc.products["product_type_name"] == cat].sample(
            min(per_category, len(svc.products[svc.products["product_type_name"] == cat]))
        )
        for _, row in cat_prods.iterrows():
            products.append({
                "id": row["id"],
                "name": row["name"],
                "category": row["product_type_name"],
                "color": row["colour_group_name"],
                "price": row.get("price", 49.99),
                "image_url": row.get("image_url", "")
            })
        if len(products) >= n:
            break
    
    return {"products": products[:n]}
