from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import catalog, alternatives, discovery, gifts

app = FastAPI(
    title="ShopperOS API",
    description="Personalized shopping intelligence layer",
    version="1.0.0"
)

# CORS for Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(catalog.router, prefix="/api", tags=["catalog"])
app.include_router(alternatives.router, prefix="/api", tags=["alternatives"])
app.include_router(discovery.router, prefix="/api", tags=["discovery"])
app.include_router(gifts.router, prefix="/api", tags=["gifts"])

@app.get("/")
def root():
    return {"status": "ok", "service": "ShopperOS API"}

@app.get("/health")
def health():
    return {"status": "healthy"}
