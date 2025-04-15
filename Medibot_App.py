#%%
import os
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            task="text-generation",
            temperature=0.4,  # Low for precision
            model_kwargs={"token":HF_TOKEN,
                          "max_length":"512"}
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None

def is_greeting(prompt):
    greetings = ["hi", "hello", "hey", "greetings"]
    return prompt.strip().lower() in greetings

def is_no(prompt):
    return prompt.strip().lower() == "no"

def is_ok(prompt):
    return prompt.strip().lower() == "ok"


def main():
    st.title("👨‍⚕️MediBot-A Medical Assistant🩺")
    st.markdown("I'm here to answer your questions regarding any Medical Issues🥼💉")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_response_was_query' not in st.session_state:
        st.session_state.last_response_was_query = False

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Ask a medical question or say hi!")

    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Handle greetings
        if is_greeting(prompt):
            response = "Hello! How can I assist you today?"
            with st.chat_message('assistant'):
                st.markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            st.session_state.last_response_was_query = False
            return

        # Handle "NO" after query
        if is_no(prompt) and st.session_state.last_response_was_query:
            response = "Thank you for using MediBot. Feel free to reach out anytime!"
            with st.chat_message('assistant'):
                st.markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            st.session_state.last_response_was_query = False
            return

        # Handle "Ok" after query
        if is_ok(prompt) and st.session_state.last_response_was_query:
            response = "Glad I could help!"
            with st.chat_message('assistant'):
                st.markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            st.session_state.last_response_was_query = False
            return

        # Medical query processing
        CUSTOM_PROMPT_TEMPLATE = """
        You are a medical assistant. Using the provided context, deliver a precise, clear, and detailed answer to the question, explaining the topic in a way that is easy to understand for someone without medical expertise. Include all relevant information to make the response comprehensive but concise, using simple language. If the context does not provide enough information, state: 'I don't have enough information to answer this fully.' Do not include any source references or metadata in the response.
        Context: {context}
        Question: {question}
        Answer: """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")


        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                return

            llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
            if llm is None:
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                return_source_documents=False,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"].strip()
            result_to_show = f"{result}\n\nCan I help with any other query?"

            with st.chat_message('assistant'):
                st.markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            st.session_state.last_response_was_query = True

        except Exception as e:
            st.error(f"Error processing query: {e}")
            st.session_state.last_response_was_query = False

if __name__ == "__main__":
    main()