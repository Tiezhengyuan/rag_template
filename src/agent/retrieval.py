'''
I retrieval in RAG
'''
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

print()

def build_retrieval():
    # embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key = os.getenv("OPEN_ROUTER_API_KEY"),
        base_url = "https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"}

    )
    # load vectore store
    db = FAISS.load_local(
        Path.cwd() / "src" /"faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    # build retriever
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":5,
            "fetch_k":20,
        },
    )
    return retriever