from flask import Flask, render_template, request

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# Flask App
app = Flask(__name__, template_folder="templates")


# Load Embeddings
embeddings = download_hugging_face_embeddings()


# Load FAISS Vector Store
docsearch = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)


# Ollama Local LLM
llm = OllamaLLM(
    model="phi3",
    temperature=0.3,
    num_predict=120
)


# Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# Conversational RAG Chain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)


# Home Route
@app.route("/")
def index():
    return render_template("chatboot.html")


# Chat Route
@app.route("/get", methods=["POST"])
def chat():

    try:

        # User Message
        msg = request.form["msg"]

        print("USER:", msg)

        # Get Response
        response = conversation_chain.invoke({
            "question": msg
        })

        print("FULL RESPONSE:", response)

        # Final Answer
        answer = response["answer"]

        print("BOT:", answer)

        return str(answer)

    except Exception as e:

        print("ERROR:", str(e))

        return "Error : " + str(e)


# Run Flask
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )