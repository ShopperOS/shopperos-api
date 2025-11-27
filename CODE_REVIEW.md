# Code Review: ShopperOS API

## Overview
FastAPI-based personalized shopping intelligence API with embedding-based recommendations. Overall structure is clean, but several improvements needed for production readiness.

---

## 🔴 Critical Issues

### 1. **Security: CORS Configuration**
**Location:** `app/main.py:14`
- **Issue:** `allow_origins=["*"]` allows any origin to access the API
- **Risk:** CSRF attacks, unauthorized access
- **Fix:** Use environment variable for allowed origins:
  ```python
  import os
  allow_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
  ```

### 2. **Error Handling: Missing Exception Handling**
**Location:** Multiple files
- **Issue:** No try-catch blocks around file I/O, numpy operations, or database queries
- **Risk:** Unhandled exceptions crash the API
- **Examples:**
  - `embeddings.py:_load()` - FileNotFoundError if data files missing
  - `catalog.py:get_personalized_catalog()` - KeyError if user_id format unexpected
  - `alternatives.py:get_alternatives()` - No validation of product_id format

### 3. **Data Validation: Missing Pydantic Models**
**Location:** All router files
- **Issue:** Using raw function parameters instead of Pydantic request/response models
- **Risk:** No automatic validation, unclear API contracts, harder to document
- **Fix:** Create request/response models:
  ```python
  from pydantic import BaseModel, Field
  
  class PersonalizedCatalogRequest(BaseModel):
      user_id: str = Field(..., min_length=1)
      k: int = Field(default=20, ge=1, le=100)
      category_filter: Optional[str] = None
      price_min: Optional[float] = Field(None, ge=0)
      price_max: Optional[float] = Field(None, ge=0)
  ```

### 4. **Singleton Pattern: Thread Safety**
**Location:** `app/services/embeddings.py:9-15`
- **Issue:** Singleton implementation not thread-safe
- **Risk:** Race condition during initialization in multi-threaded environments
- **Fix:** Use `threading.Lock()` or rely on FastAPI's dependency injection

---

## 🟡 Important Issues

### 5. **Performance: Inefficient Similarity Search**
**Location:** `app/services/embeddings.py:47-51`
- **Issue:** Full matrix multiplication for every query: `self.embeddings @ query_vec`
- **Impact:** O(n*d) complexity where n=products, d=embedding_dim
- **Recommendation:** 
  - For large catalogs (>10k products), use approximate nearest neighbor (ANN) libraries:
    - `faiss` (Facebook AI Similarity Search)
    - `annoy` (Spotify)
    - `hnswlib`
  - Current approach fine for <10k products

### 6. **Memory: Large Data Structures in Memory**
**Location:** `app/services/embeddings.py:21-43`
- **Issue:** All embeddings, products, and user vectors loaded into memory at startup
- **Impact:** High memory usage (~600MB+ as noted in README)
- **Recommendation:**
  - Consider lazy loading for less frequently accessed data
  - Use memory-mapped files for embeddings
  - Implement pagination for product data

### 7. **Code Quality: Inconsistent Error Responses**
**Location:** Multiple router files
- **Issue:** Some endpoints return `{"error": "..."}`, others may raise exceptions
- **Examples:**
  - `catalog.py:compute_taste_from_calibration()` returns error dict
  - `alternatives.py:get_alternatives()` returns error dict
  - But other endpoints may raise unhandled exceptions
- **Fix:** Use FastAPI's `HTTPException` consistently:
  ```python
  from fastapi import HTTPException, status
  
  if not liked_ids:
      raise HTTPException(
          status_code=status.HTTP_400_BAD_REQUEST,
          detail="Need at least one liked product"
      )
  ```

### 8. **Data Consistency: ID Normalization**
**Location:** Multiple files
- **Issue:** Inconsistent ID normalization with `str(product_id).lstrip("0")`
- **Examples:**
  - `alternatives.py:37` - `str(product_id).lstrip("0")`
  - `gifts.py:46` - `str(pid).lstrip("0")`
  - `embeddings.py:55` - `str(product_id).lstrip("0")`
- **Risk:** Potential mismatches if IDs have different formats
- **Fix:** Centralize ID normalization in `EmbeddingService`:
  ```python
  def normalize_id(self, product_id: str) -> str:
      """Normalize product ID format"""
      return str(product_id).lstrip("0")
  ```

### 9. **Missing Input Validation**
**Location:** `app/routers/catalog.py:125`
- **Issue:** `get_calibration_products()` doesn't validate `n` parameter bounds
- **Current:** `n: int = Query(default=20, le=50)` - only upper bound
- **Fix:** Add lower bound: `n: int = Query(default=20, ge=1, le=50)`

### 10. **Type Hints: Incomplete**
**Location:** Multiple files
- **Issue:** Missing return type hints on several functions
- **Examples:**
  - `catalog.py:get_personalized_catalog()` - no return type
  - `discovery.py:get_discovery_feed()` - no return type
- **Fix:** Add return types for better IDE support and type checking

---

## 🟢 Minor Issues & Improvements

### 11. **Code Duplication: Product Dictionary Creation**
**Location:** Multiple files
- **Issue:** Similar product dict creation logic repeated:
  - `catalog.py:26-29`, `catalog.py:144-151`
  - `gifts.py:29-36`
- **Fix:** Centralize in `EmbeddingService.get_product()` (already exists, but ensure all use it)

### 12. **Magic Numbers**
**Location:** Multiple files
- **Issue:** Hardcoded values without explanation:
  - `catalog.py:36` - `k * 3` (why 3x?)
  - `catalog.py:98` - `0.3 * dislike_avg` (why 0.3?)
  - `discovery.py:22` - `0.1` noise factor
- **Fix:** Extract to named constants or configuration

### 13. **Logging: Missing Structured Logging**
**Location:** All files
- **Issue:** Only `print()` statements, no proper logging
- **Fix:** Use Python's `logging` module:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info(f"Loading embeddings...")
  ```

### 14. **Configuration: Hardcoded Paths**
**Location:** `app/services/embeddings.py:6`
- **Issue:** Data directory path hardcoded
- **Fix:** Use environment variable:
  ```python
  DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent.parent / "data"))
  ```

### 15. **API Documentation: Missing Examples**
**Location:** All router files
- **Issue:** Docstrings present but no OpenAPI examples
- **Fix:** Add example values to Pydantic models or use `example` parameter in route decorators

### 16. **Testing: No Test Files**
**Location:** Entire codebase
- **Issue:** No unit tests or integration tests
- **Recommendation:** Add tests using `pytest` and `httpx`:
  - Unit tests for `EmbeddingService`
  - Integration tests for API endpoints
  - Mock data for testing

### 17. **Dependencies: Missing Version Pins**
**Location:** `requirements.txt`
- **Issue:** Some dependencies may have breaking changes in minor versions
- **Current:** All versions pinned (good!)
- **Note:** Consider adding `pip-tools` or `poetry` for better dependency management

### 18. **Docker: Missing .dockerignore**
**Location:** Root directory
- **Issue:** No `.dockerignore` file
- **Risk:** Unnecessary files copied to Docker image
- **Fix:** Create `.dockerignore`:
  ```
  __pycache__
  *.pyc
  .git
  .gitignore
  README.md
  CODE_REVIEW.md
  *.md
  .env
  ```

### 19. **Missing .gitignore**
**Location:** Root directory
- **Issue:** No `.gitignore` file found
- **Risk:** Committing sensitive data, cache files, etc.
- **Fix:** Create `.gitignore` for Python projects

### 20. **Health Check: Basic Implementation**
**Location:** `app/main.py:30-32`
- **Issue:** Health check doesn't verify data loaded successfully
- **Fix:** Check if EmbeddingService is initialized:
  ```python
  @app.get("/health")
  def health():
      svc = get_embedding_service()
      return {
          "status": "healthy",
          "products_loaded": len(svc.products),
          "users_loaded": len(svc.user_taste_vectors)
      }
  ```

---

## 📊 Code Quality Metrics

### Positive Aspects
✅ Clean separation of concerns (routers/services)  
✅ FastAPI best practices (dependency injection)  
✅ Singleton pattern for expensive initialization  
✅ Good endpoint organization  
✅ Clear function names and structure  

### Areas for Improvement
⚠️ Error handling (missing try-catch blocks)  
⚠️ Input validation (no Pydantic models)  
⚠️ Type hints (incomplete)  
⚠️ Logging (using print instead of logging)  
⚠️ Testing (no test coverage)  

---

## 🚀 Recommended Priority Actions

### High Priority (Before Production)
1. ✅ Add proper error handling with try-catch blocks
2. ✅ Implement Pydantic request/response models
3. ✅ Fix CORS configuration
4. ✅ Add consistent error responses using HTTPException
5. ✅ Add health check that verifies data loading

### Medium Priority (Soon)
6. ✅ Add structured logging
7. ✅ Centralize ID normalization
8. ✅ Add input validation bounds
9. ✅ Extract magic numbers to constants
10. ✅ Add .gitignore and .dockerignore

### Low Priority (Nice to Have)
11. ✅ Add unit and integration tests
12. ✅ Consider ANN library for large-scale similarity search
13. ✅ Add API documentation examples
14. ✅ Implement lazy loading for large datasets
15. ✅ Add configuration via environment variables

---

## 📝 Additional Notes

### Architecture Suggestions
- Consider adding a caching layer (Redis) for frequently accessed products
- For horizontal scaling, consider externalizing embeddings to a vector database (Pinecone, Weaviate, Qdrant)
- Add rate limiting middleware for API protection
- Consider adding request/response middleware for logging and monitoring

### Performance Considerations
- Current similarity search is O(n*d) - acceptable for <10k products
- For larger catalogs, implement approximate nearest neighbor search
- Consider batch processing for multiple user requests
- Add response caching for popular queries

---

## Summary

The codebase is well-structured and follows FastAPI conventions, but needs improvements in error handling, validation, and security before production deployment. The core logic is sound, but production-readiness requires addressing the critical and important issues listed above.

**Estimated effort to address high-priority issues: 4-6 hours**  
**Estimated effort for full production readiness: 1-2 days**

