# ShopperOS API

Personalized shopping intelligence API.

## Local Setup

1. Copy your data files:
```bash
mkdir -p data/embeddings data/processed data/taste

# From /Users/berlin/data/
cp embeddings/hybrid_embeddings.npy data/embeddings/
cp embeddings/id_mappings.pkl data/embeddings/
cp processed/products_extended.parquet data/processed/
cp taste/user_taste_vectors.pkl data/taste/
cp taste/popular_products.pkl data/taste/
cp taste/trending.pkl data/taste/
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run locally:
```bash
uvicorn app.main:app --reload
```

4. Test at http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/get_personalized_catalog` | POST | Taste-ranked products |
| `/api/get_alternatives` | POST | Similar products |
| `/api/get_discovery_feed` | GET | Discovery sections |
| `/api/get_gift_suggestions` | POST | Gift recommendations |
| `/api/get_calibration_products` | GET | Onboarding products |
| `/api/compute_taste_from_calibration` | POST | Compute taste from swipes |

## Deploy to Railway

1. Push to GitHub
2. Connect Railway to your repo
3. Add environment variables (if any)
4. Deploy

Data files (~600MB) will need to be:
- Bundled in Docker image, OR
- Stored in S3/GCS and loaded at startup

## Connect to Lovable

In Lovable, call these endpoints:
```javascript
const API_URL = "https://your-railway-app.up.railway.app";

// Get personalized catalog
const response = await fetch(`${API_URL}/api/get_personalized_catalog`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ user_id: "abc123", k: 20 })
});
```
