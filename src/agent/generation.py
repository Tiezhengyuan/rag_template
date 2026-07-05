'''
III Generation in RAG
'''
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.base import format_document
from langchain_core.output_parsers import StrOutputParser

DEFAULT_DOCUMENT_PROMPT = ChatPromptTemplate.from_template("{page_content}")

def build_generation(llm):

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant.

                    Answer ONLY using the retrieved context.

                    If the answer cannot be found, say "I don't know."

                    Context:
                    {context}
                    """,
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(
            format_document(doc, DEFAULT_DOCUMENT_PROMPT)
            for doc in docs
        )

    generation_chain = (
        {
            "context": lambda x: format_docs(x["context"]),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return generation_chain