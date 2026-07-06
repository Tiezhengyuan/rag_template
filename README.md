# RAG

RAG(retrieval, augmentation, and generation) application.



## Development

### local 

set environmental variables.
```
OPEN_ROUTER_API_KEY="<replace with your API KEY>"
```


```
streamlit run src/app.py
```

### production


check dependencies and build requirements.txt
```
pipreqs src
```

build image known as rag-image
```
docker build -t rag-image .
```

run container known as rag
```
docker run -d \
    --env-file .env \
    -p 8501:8501 \
    -v "$(pwd)/faiss_index:/app/faiss_index" \
    --name rag \
    rag-image
```

## Deployment

### local mode

```
docker compose up -d
```

check status if the container is running well
```
docker logs rag
```

stop containers
```
docker compose down
```

### debug

Test if API is working within container
```
docker exec -it rag sh
apt-get update && apt-get install -y curl
# test API KEY
curl -v -X POST https://openrouter.ai/api/v1/embeddings \
  -H "Authorization: Bearer $OPEN_ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/text-embedding-3-small",
    "input": "hello world"
  }'
```

check if a port is available
```
sudo lsof -i :8501 
```

Delete all containers
```
docker rm -f $(docker ps -aq)
```

