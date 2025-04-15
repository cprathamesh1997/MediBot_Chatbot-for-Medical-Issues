#%%
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore")

# Step 1: Load environment variables (optional, if using Hugging Face API token)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your token"  

# Step 2: Load FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded")
except Exception as e:
    print(f"Error loading FAISS vector store: {e}")
    raise

#%%
# Step 3: Set up LLM (Mistral-7B-Instruct-v0.3)
try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",  
        max_new_tokens=512,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    print("LLM initialized")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    raise

#%%
# Step 4: Define prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer: """
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Step 5: Create RetrievalQA chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("RetrievalQA chain created")
except Exception as e:
    print(f"Error creating QA chain: {e}")
    raise

#%%
# Step 6: Query the model
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
