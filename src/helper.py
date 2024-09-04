from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import ollama

ollama.pull('nomic-embed-text')

#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents



#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def loading_llama():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings