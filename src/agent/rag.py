import os
from langchain_openrouter import ChatOpenRouter
from langchain_classic.chains import create_retrieval_chain

from agent.retrieval import build_retrieval
from agent.generation import build_generation
from agent.augmentation import build_augmentation


# assemble RAG
def build_chain():
    # load LLM
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        api_key = os.getenv("OPEN_ROUTER_API_KEY"),
        temperature=0,
    )

    # Retrieval: 
    retriever = build_retrieval()
    
    # augmentation
    history_aware_retriever_chain = build_augmentation(llm, retriever)

    # Generation: 
    document_chain = build_generation(llm)

    # ensemble RAG
    rag_chain = create_retrieval_chain(
        history_aware_retriever_chain,
        document_chain,
    )
    return rag_chain