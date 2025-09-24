# chatbot.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import os

# ---------------------------
# Sources (same as ingestion)
# ---------------------------
text_sources = {
    'I.P.C': 'ipc',
    'Constitution': 'constitution',
    'Garuda': 'garuda',
    'Bhagavad Gita': 'bhagavad gita',
    'Quran': 'quran',
}

# ---------------------------
# Load Vector DBs
# ---------------------------
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

vector_dbs = {}
for name in text_sources.keys():
    path = f"vectorstores/{name}"
    if os.path.exists(path):
        vector_dbs[name] = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)

# ---------------------------
# Lightweight LLM
# ---------------------------
print("🤖 Loading model...")
generator = pipeline("text2text-generation", model="google/flan-t5-small")  # smaller model for Render

def generate_answer(question, context, source_name):
    prompt = f"""
    You are a helpful assistant.
    Answer the question only if it is clearly supported by the provided context.
    If the context is unrelated or unclear, respond with exactly: "Not mentioned in this source."

    Question: {question}
    Context from {source_name}:
    {context}

    Answer:
    """
    return generator(prompt, max_new_tokens=200, clean_up_tokenization_spaces=True)[0]['generated_text']

# ---------------------------
# QA Function
# ---------------------------
def answer_question(question, k=2):
    final_output = []
    for source_name, db in vector_dbs.items():
        docs = db.similarity_search(question, k=k)
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            answer = generate_answer(question, context, source_name).strip()
            if answer and answer != "Not mentioned in this source.":
                final_output.append(f"According to {source_name.capitalize()}: {answer}")
    return "\n\n".join(final_output) if final_output else "No relevant answer found in any source."

def chat(question: str):
    return answer_question(question)
