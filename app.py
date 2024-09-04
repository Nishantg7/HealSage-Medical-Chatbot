from flask import Flask, render_template, jsonify,request
from src.helper import loading_llama
from langchain_community.vectorstores import Pinecone
import pinecone 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv 
from src.prompt import *
import os

app=Flask(__name__)


load_dotenv()
GROQ_KEY=os.getenv("GROQ_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")

embeddings = loading_llama()

index_name="medical-chat-bot"

docsearch=Pinecone.from_existing_index(index_name, embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain_type_kwargs={"prompt": PROMPT}

llm=ChatGroq(temperature=0.5, groq_api_key=GROQ_KEY, model_name="llama-3.1-70b-versatile")


retriever = docsearch.as_retriever(search_kwargs={'k':2})

qa = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa.invoke({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)