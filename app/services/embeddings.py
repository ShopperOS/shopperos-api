import numpy as np
import pickle
import os
from pathlib import Path
from functools import lru_cache
from supabase import create_client, Client

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Supabase client
def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("Warning: Supabase credentials not set")
        return None
    return create_client(url, key)

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
        
        # Cache for Supabase taste vectors
        self.user_taste_vectors_cache = {}
        
        # Load popular/trending
        with open(DATA_DIR / "taste" / "popular_products.pkl", "rb") as f:
            self.popular_products = pickle.load(f)
        
        with open(DATA_DIR / "taste" / "trending.pkl", "rb") as f:
            self.trending = pickle.load(f)
        
        # Supabase client
        self.supabase = get_supabase_client()
        
        print(f"Loaded: {len(self.products)} products, {len(self.demo_taste_vectors)} demo users")
    
    def get_user_taste_vector(self, user_id: str) -> np.ndarray:
        """Get taste vector for user - checks Supabase first, then demo data"""
        # Check cache first
        if user_id in self.user_taste_vectors_cache:
            return self.user_taste_vectors_cache[user_id]
        
        # Check Supabase
        if self.supabase:
            try:
                result = self.supabase.table("user_taste_vectors").select("taste_vector").eq("user_id", user_id).execute()
                if result.data and len(result.data) > 0:
                    taste_vector = np.array(result.data[0]["taste_vector"], dtype=np.float32)
                    self.user_taste_vectors_cache[user_id] = taste_vector
                    return taste_vector
            except Exception as e:
                print(f"Error fetching taste vector from Supabase: {e}")
        
        # Check demo data
        if user_id in self.demo_taste_vectors:
            return self.demo_taste_vectors[user_id]
        
        return None
    
    def save_user_taste_vector(self, user_id: str, taste_vector: np.ndarray) -> bool:
        """Save taste vector to Supabase"""
        if not self.supabase:
            print("Supabase not configured, cannot save taste vector")
            return False
        
        try:
            # Convert numpy array to list for JSON storage
            vector_list = taste_vector.tolist()
            
            # Upsert to Supabase
            self.supabase.table("user_taste_vectors").upsert({
                "user_id": user_id,
                "taste_vector": vector_list,
                "updated_at": "now()"
            }, on_conflict="user_id").execute()
            
            # Update cache
            self.user_taste_vectors_cache[user_id] = taste_vector
            
            print(f"Saved taste vector for user {user_id}")
            return True
        except Exception as e:
            print(f"Error saving taste vector to Supabase: {e}")
            return False
    
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