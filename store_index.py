from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

from langchain_community.vectorstores import FAISS


extracted_data = load_pdf_files("data/")
filter_data = filter_to_minimal_docs(extracted_data)

text_chunks = text_split(filter_data)

embeddings = download_hugging_face_embeddings()

docsearch = FAISS.from_documents(
    text_chunks,
    embeddings
)

docsearch.save_local("faiss_index")

print("FAISS vector store created successfully")