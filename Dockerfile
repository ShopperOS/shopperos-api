FROM python:3.11-slim

WORKDIR /app

# Install git-lfs
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Clone repo and pull LFS files
ARG GITHUB_TOKEN
RUN git clone https://${GITHUB_TOKEN}@github.com/ShopperOS/shopperos-api.git . && git lfs pull

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]