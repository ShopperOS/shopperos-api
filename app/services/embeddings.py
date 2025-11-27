import numpy as np
import pickle
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
        self.embeddings = np.load(DATA_DIR / "embeddings" / "hybrid_embeddings.npy")
        
        # Load mappings
        with open(DATA_DIR / "embeddings" / "id_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        self.id_to_idx = mappings["id_to_idx"]
        self.idx_to_id = mappings["idx_to_id"]
        
        # Load products
        import pandas as pd
        self.products = pd.read_parquet(DATA_DIR / "processed" / "products_extended.parquet")
        self.products["id"] = self.products["id"].astype(str).str.lstrip("0")
        
        # Load taste vectors
        with open(DATA_DIR / "taste" / "user_taste_vectors.pkl", "rb") as f:
            self.user_taste_vectors = pickle.load(f)
        
        # Load popular/trending
        with open(DATA_DIR / "taste" / "popular_products.pkl", "rb") as f:
            self.popular_products = pickle.load(f)
        
        with open(DATA_DIR / "taste" / "trending.pkl", "rb") as f:
            self.trending = pickle.load(f)
        
        print(f"Loaded: {len(self.products)} products, {len(self.user_taste_vectors)} users")
    
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
            "brand": row.get("index_name", ""),
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
