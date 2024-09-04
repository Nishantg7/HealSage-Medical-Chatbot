from src.helper import load_pdf, text_split,loading_llama
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import ollama

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
# print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")

text_chunks = text_split(extracted_data)

embeddings = loading_llama()

index_name="medical-chat-bot"

docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)