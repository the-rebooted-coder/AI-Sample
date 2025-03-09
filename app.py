from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

app = Flask(__name__)
CORS(app)

vector_store = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ingest', methods=['POST'])
def ingest_urls():
    global vector_store
    urls = request.json.get('urls', [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    try:
        # Load webpage data
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create embeddings with HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(splits, embeddings)

        return jsonify({"status": "success", "message": f"Ingested {len(urls)} URLs"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global vector_store
    question = request.json.get('question', '')

    if not vector_store:
        return jsonify({"error": "No content ingested yet"}), 400

    try:
        # Use HuggingFace model for answering questions
        llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5, "max_new_tokens": 200})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            chain_type="stuff"
        )

        result = qa_chain({"query": question})

        return jsonify({"answer": result["result"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
