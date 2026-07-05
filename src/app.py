'''
streamlit app for chat with RAG
'''
import streamlit as st
from agent.rag import build_chain

# title UI
st.set_page_config(page_title="LangChain RAG")
st.title("📚 Chat with your documents")

# initialize history
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# RAG
chain = build_chain()

# chat UI
question = st.chat_input("Ask a question")
if question:
    # UI: input
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # UI: RAG
    with st.spinner("Thinking..."):
        response = chain.invoke({
            "input": question,
            "chat_history": st.session_state.history,
        })
        answer = response["answer"]

    st.session_state.history.extend([
        ("human", question),
        ("ai", answer),
    ])
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })

    with st.chat_message("assistant"):
        st.markdown(answer)
        if "context" in response:
            with st.expander("Retrieved Documents"):
                for i, doc in enumerate(response["context"], 1):
                    st.write(f"### Document {i}")
                    st.write(doc.page_content[:500])
                    st.write(doc.metadata)