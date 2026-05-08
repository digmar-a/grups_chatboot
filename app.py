from flask import Flask, render_template, request

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Flask app
app = Flask(__name__, template_folder="templates")


# Load embeddings
embeddings = download_hugging_face_embeddings()


# Load FAISS
docsearch = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# Ollama LLM
llm = OllamaLLM(
    model="llama3"
)


# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# QA Chain
question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)


# RAG Chain
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


@app.route("/")
def index():
    return render_template("chatboot.html")


@app.route("/get", methods=["POST"])
def chat():

    try:
        msg = request.form["msg"]

        print("User:", msg)

        response = rag_chain.invoke({
            "input": msg
        })

        answer = response["answer"]

        print("Bot:", answer)

        return str(answer)

    except Exception as e:
        print("ERROR:", str(e))
        return "Error : " + str(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)