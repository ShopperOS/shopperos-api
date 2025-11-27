from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
import numpy as np
from collections import Counter
from app.services.embeddings import EmbeddingService, get_embedding_service

router = APIRouter()


class GiftSuggestionsRequest(BaseModel):
    gift_list_items: List[str] = []
    k: int = 10
    diversify: bool = True


class GiftForUserRequest(BaseModel):
    recipient_user_id: str
    k: int = 10
    price_min: Optional[float] = None
    price_max: Optional[float] = None


@router.post("/get_gift_suggestions")
def get_gift_suggestions(
    request: GiftSuggestionsRequest,
    svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Get gift suggestions based on items already in a gift list.
    """
    gift_list_items = request.gift_list_items
    k = request.k
    diversify = request.diversify
    
    # Empty list - return popular gift-worthy items
    if not gift_list_items:
        gift_categories = ["Dress", "Sweater", "Bag"]
        suggestions = []
        
        for cat in gift_categories:
            cat_prods = svc.products[svc.products["product_type_name"] == cat]
            if len(cat_prods) > 0:
                samples = cat_prods.sample(min(3, len(cat_prods)))
                for _, row in samples.iterrows():
                    suggestions.append({
                        "product_id": row["id"],
                        "name": row["name"],
                        "category": row["product_type_name"],
                        "color": row["colour_group_name"],
                        "price": row.get("price", 49.99),
                        "reason": f"Popular {cat.lower()} gift"
                    })
        
        return {"suggestions": suggestions[:k], "is_empty_list": True}
    
    # Get embeddings for items in list
    list_vectors = []
    list_categories = []
    list_colors = []
    
    for pid in gift_list_items:
        pid_str = str(pid).lstrip("0")
        emb = svc.get_embedding(pid_str)
        if emb is not None:
            list_vectors.append(emb)
            prod = svc.get_product(pid_str)
            if prod:
                list_categories.append(prod["category"])
                list_colors.append(prod["color"])
    
    if not list_vectors:
        return get_gift_suggestions(
            GiftSuggestionsRequest(gift_list_items=[], k=k, diversify=diversify), 
            svc=svc
        )
    
    # Compute gift list taste
    list_taste = np.mean(list_vectors, axis=0)
    list_taste = list_taste / (np.linalg.norm(list_taste) + 1e-8)
    
    # Search
    indices, scores = svc.search_similar(list_taste, k=k * 5)
    
    suggestions = []
    seen_ids = set(str(p).lstrip("0") for p in gift_list_items)
    seen_categories = set() if diversify else None
    
    top_category = Counter(list_categories).most_common(1)[0][0] if list_categories else None
    top_color = Counter(list_colors).most_common(1)[0][0] if list_colors else None
    
    for i, idx in enumerate(indices):
        pid = svc.idx_to_id[idx]
        
        if pid in seen_ids:
            continue
        
        prod = svc.get_product(pid)
        if prod is None:
            continue
        
        cat = prod["category"]
        color = prod["color"]
        
        if diversify and cat in seen_categories:
            continue
        
        # Generate reason
        if cat == top_category:
            reason = f"Matches their love of {cat.lower()}s"
        elif color == top_color:
            reason = f"In their favorite color ({color})"
        else:
            reason = "Complements items on their list"
        
        suggestions.append({
            "product_id": pid,
            "name": prod["name"],
            "category": cat,
            "color": color,
            "price": prod["price"],
            "similarity": float(scores[i]),
            "reason": reason
        })
        
        if diversify:
            seen_categories.add(cat)
        
        if len(suggestions) >= k:
            break
    
    return {"suggestions": suggestions, "is_empty_list": False}


@router.post("/get_gift_suggestions_for_user")
def get_gift_suggestions_for_user(
    request: GiftForUserRequest,
    svc: EmbeddingService = Depends(get_embedding_service)
):
    """
    Get gift suggestions based on recipient's taste profile.
    """
    recipient_user_id = request.recipient_user_id
    k = request.k
    price_min = request.price_min
    price_max = request.price_max
    
    if recipient_user_id not in svc.user_taste_vectors:
        # Fallback to popular gifts
        return get_gift_suggestions(
            GiftSuggestionsRequest(gift_list_items=[], k=k),
            svc=svc
        )
    
    taste = svc.user_taste_vectors[recipient_user_id]
    indices, scores = svc.search_similar(taste, k=k * 5)
    
    exclude_categories = ["Underwear bottom", "Underwear Tights", "Socks"]
    
    suggestions = []
    seen_categories = set()
    
    for i, idx in enumerate(indices):
        pid = svc.idx_to_id[idx]
        prod = svc.get_product(pid)
        
        if prod is None:
            continue
        
        cat = prod["category"]
        price = prod["price"]
        
        # Filters
        if cat in exclude_categories:
            continue
        if price_min and price < price_min:
            continue
        if price_max and price > price_max:
            continue
        if cat in seen_categories:
            continue
        
        suggestions.append({
            "product_id": pid,
            "name": prod["name"],
            "category": cat,
            "color": prod["color"],
            "price": price,
            "affinity": float(scores[i]),
            "reason": f"Matches their style ({scores[i]:.0%} affinity)"
        })
        
        seen_categories.add(cat)
        
        if len(suggestions) >= k:
            break
    
    return {"suggestions": suggestions}