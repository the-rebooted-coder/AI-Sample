from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS
from langchain.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
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

        # Clean HTML content from each document using BeautifulSoup
        clean_docs = []
        for doc in docs:
            clean_text = BeautifulSoup(doc.page_content, "html.parser").get_text()
            doc.page_content = clean_text
            clean_docs.append(doc)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(clean_docs)

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
        # Configure the model for brief output
        llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            model_kwargs={"temperature": 0.3, "max_new_tokens": 50}
        )

        # Custom prompt instructing brevity without echoing prompt text
        custom_prompt = PromptTemplate(
            template=(
                "Answer the following question briefly and concisely. "
                "Do not repeat any prompt instructions or context.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": custom_prompt}
        )

        result = qa_chain({"query": question})
        full_output = result["result"]

        # Post-process to extract only the final answer
        if "Answer:" in full_output:
            final_answer = full_output.split("Answer:")[-1].strip()
        else:
            final_answer = full_output.strip()

        return jsonify({"answer": final_answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
