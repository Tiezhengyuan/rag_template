'''
build/update vector store
'''
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/sample.pdf")
documents = loader.load()

# build vectore database
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key = os.getenv("OPEN_ROUTER_API_KEY"),
    base_url = "https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"}

)

db = FAISS.from_documents(chunks, embeddings)

# save
index_path = Path.cwd() / "faiss_index"
db.save_local(index_path)
print(f"Vector store created at {index_path}.")