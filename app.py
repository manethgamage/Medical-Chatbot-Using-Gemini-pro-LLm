from flask import Flask, render_template, jsonify, request
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Pinecone
from langchain import PromptTemplate
from src.prompt import *
import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

os.environ['PINECONE_API_KEY'] = '900597d2-8e82-4bb0-9f45-c58ff809dade'

index_name = "medical-bot"

embeddings = GooglePalmEmbeddings(google_api_key='AIzaSyC14roYzDCvA_MNB0m1Mfc6vr74UO6yr74')

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

docsearch=Pinecone.from_existing_index(index_name, embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = ChatGoogleGenerativeAI(google_api_key="AIzaSyCMOSfEQv4qpq21bIe9RGOYctSYYQfMp8Q",model="gemini-pro", temperature=0.8)


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)



