FROM python:3.11-slim

WORKDIR /app

# Install git-lfs
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

# Clone repo with LFS files
ARG REPO_URL
RUN git clone ${REPO_URL} . && git lfs pull

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]