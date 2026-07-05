'''
II in RAG
'''
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from operator import itemgetter

def build_augmentation(llm, retriever):
    #prompt template with history
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
    contextualize_prompt = ChatPromptTemplate.from_messages(template)

    # Rewrite question
    rewrite_question_chain = (
        contextualize_prompt
        | llm
        | StrOutputParser()
    )

    # History-aware retriever
    return RunnableBranch(
        (
            lambda x: bool(x.get("chat_history")),
            rewrite_question_chain | retriever,
        ),
        itemgetter("input") | retriever,
    )