import numpy as np
import pickle
import os
import httpx
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent.parent.parent / "data"

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        print("Loading embeddings...")
        
        # Load embeddings
        self.embeddings = np.load(DATA_DIR / "embeddings" / "hybrid_embeddings.npy", allow_pickle=True)
        
        # Load mappings
        with open(DATA_DIR / "embeddings" / "id_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        self.id_to_idx = mappings["id_to_idx"]
        self.idx_to_id = mappings["idx_to_id"]
        
        # Load products
        import pandas as pd
        self.products = pd.read_parquet(DATA_DIR / "processed" / "products_extended.parquet")
        self.products["id"] = self.products["id"].astype(str).str.lstrip("0")
        
        # Load demo taste vectors (fallback)
        with open(DATA_DIR / "taste" / "user_taste_vectors.pkl", "rb") as f:
            self.demo_taste_vectors = pickle.load(f)
        
        # Cache for fetched taste vectors
        self.user_taste_vectors_cache = {}
        
        # Load popular/trending
        with open(DATA_DIR / "taste" / "popular_products.pkl", "rb") as f:
            self.popular_products = pickle.load(f)
        
        with open(DATA_DIR / "taste" / "trending.pkl", "rb") as f:
            self.trending = pickle.load(f)
        
        # Edge function config
        self.functions_url = os.environ.get("LOVABLE_FUNCTIONS_URL", "")
        self.api_key = os.environ.get("RAILWAY_API_KEY", "")
        
        print(f"Loaded: {len(self.products)} products, {len(self.demo_taste_vectors)} demo users")
        print(f"Edge functions URL configured: {bool(self.functions_url)}")
    
    def get_user_taste_vector(self, user_id: str) -> np.ndarray:
        """Get taste vector for user - checks edge function first, then demo data"""
        # Check cache first
        if user_id in self.user_taste_vectors_cache:
            return self.user_taste_vectors_cache[user_id]
        
        # Try fetching from Lovable edge function
        if self.functions_url and self.api_key:
            try:
                response = httpx.get(
                    f"{self.functions_url}/get_user_taste_vector",
                    params={"user_id": user_id},
                    headers={"X-API-Key": self.api_key},
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("found") and data.get("taste_vector"):
                        taste_vector = np.array(data["taste_vector"], dtype=np.float32)
                        self.user_taste_vectors_cache[user_id] = taste_vector
                        print(f"Fetched taste vector for user {user_id} from edge function")
                        return taste_vector
            except Exception as e:
                print(f"Error fetching taste vector from edge function: {e}")
        
        # Check demo data
        if user_id in self.demo_taste_vectors:
            return self.demo_taste_vectors[user_id]
        
        return None
    
    def has_taste_vector(self, user_id: str) -> bool:
        """Check if user has a taste vector"""
        return self.get_user_taste_vector(user_id) is not None
    
    def search_similar(self, query_vec: np.ndarray, k: int = 10):
        """Numpy-based similarity search"""
        scores = self.embeddings @ query_vec
        top_k = np.argsort(scores)[-k:][::-1]
        return top_k, scores[top_k]
    
    def get_product(self, product_id: str) -> dict:
        """Get product by ID"""
        pid = str(product_id).lstrip("0")
        prod = self.products[self.products["id"] == pid]
        if len(prod) == 0:
            return None
        row = prod.iloc[0]
        return {
            "id": row["id"],
            "name": row["name"],
            "category": row["product_type_name"],
            "color": row["colour_group_name"],
            "price": row.get("price", 49.99),
            "brand": row.get("index_group_name", ""),
            "image_url": row.get("image_url", ""),
            "product_url": row.get("product_url", "")
        }
    
    def get_embedding(self, product_id: str) -> np.ndarray:
        """Get embedding for a product"""
        pid = str(product_id).lstrip("0")
        if pid not in self.id_to_idx:
            return None
        idx = self.id_to_idx[pid]
        return self.embeddings[idx]

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
```

Also add `httpx` to your `requirements.txt`:
```
httpx