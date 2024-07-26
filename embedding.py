from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
loader=PyPDFDirectoryLoader("./data") ## Data Ingestion
docs=loader.load() ## Document Loading
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
final_documents=text_splitter.split_documents(docs[:20]) #splitting
vectors=FAISS.from_documents(final_documents,embeddings) #vector embeddings
vectors.save_local("faiss_index")
