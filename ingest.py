import os
import requests
import tempfile
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Sources
text_sources = {
    'I.P.C': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/ipc.pdf',
    'Constitution': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/indian%20constitution.pdf',
    'Garuda': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/GarudaPurana.pdf',
    'Bhagavad Gita': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/Bhagavad-gita_As_It_Is.pdf',
    'Quran': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/quran-allah.pdf',
}

# Create vectorstores folder
os.makedirs("vectorstores", exist_ok=True)

# Embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# Download, split, embed, and save
for name, url in tqdm(text_sources.items()):
    # Download PDF
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Split
    chunks = text_splitter.split_documents(docs)

    # Create FAISS
    if chunks:
        db = FAISS.from_documents(chunks, embedder)
        db.save_local(f"vectorstores/{name.lower().replace(' ', '_')}")
        print(f"✅ Created vectorstore for {name}")
