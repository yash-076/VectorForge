# Embedding Microservice

A lightweight FastAPI service that generates semantic vector embeddings using **sentence-transformers/all-MiniLM-L6-v2**.

## Project Structure

```
VectorForge/
├── app/
│   ├── main.py                    # FastAPI application & routes
│   ├── schemas.py                 # Pydantic request/response models
│   └── services/
│       └── embedding_service.py   # Core embedding logic (SentenceTransformer)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick Start (Local)

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

The first startup will download the model (~80 MB) if it is not already cached.

## Quick Start (Docker)

```bash
docker build -t embedding-service .
docker run -p 8100:8100 embedding-service
```

## API Reference

### `POST /embed` — single text

```bash
curl -X POST http://localhost:8100/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

Response:

```json
{
  "embedding": [0.0123, -0.0456, ...]
}
```

### `POST /embed-batch` — multiple texts

```bash
curl -X POST http://localhost:8100/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"]}'
```

Response:

```json
{
  "embeddings": [[0.012, ...], [-0.034, ...]]
}
```

### `GET /health` — health check

```bash
curl http://localhost:8100/health
```

Response:

```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

## API Key Authentication (Optional)

Set the `API_KEY` environment variable to enable simple header-based auth:

```bash
# Local
API_KEY=my-secret-key uvicorn app.main:app --host 0.0.0.0 --port 8100

# Docker
docker run -e API_KEY=my-secret-key -p 8100:8100 embedding-service
```

Then include the header in every request:

```bash
curl -H "X-API-Key: my-secret-key" http://localhost:8100/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "secure request"}'
```

If `API_KEY` is **not** set, authentication is disabled and all requests are allowed.

## Environment Variables

| Variable     | Default                                      | Description                       |
| ------------ | -------------------------------------------- | --------------------------------- |
| `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2`     | HuggingFace model identifier      |
| `DEVICE`     | `cpu`                                        | Torch device (`cpu` or `cuda`)    |
| `API_KEY`    | *(unset — auth disabled)*                    | API key for `X-API-Key` header    |

## Interactive Docs

Once running, visit **http://localhost:8100/docs** for the auto-generated Swagger UI.

