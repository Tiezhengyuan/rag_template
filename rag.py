import os
from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

def build_chain():

    # embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key = os.getenv("OPEN_ROUTER_API_KEY"),
        base_url = "https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"}

    )
    # load vector store
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":5,
            "fetch_k":20,
        },
    )

    # load LLM
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        api_key = os.getenv("OPEN_ROUTER_API_KEY"),
        temperature=0,
    )

    #Agumentation: prompt template with history
    template = [
        (
            "system",
            """Given the conversation history and the latest user question,
                rewrite the question so it is self-contained.
                Do not answer the question.""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    contextualize_prompt_template = ChatPromptTemplate.from_messages(template)
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt_template,
    )

    # Generation: 
    qa_template = [
        (
            "system",
            """You are a helpful assistant.
                Answer ONLY using the retrieved context.
                If the answer cannot be found,
                say "I don't know."
            
                Context:
                {context}""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    qa_prompt_template = ChatPromptTemplate.from_messages(qa_template)
    document_chain = create_stuff_documents_chain(
        llm,
        qa_prompt_template,
    )
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    return rag_chain