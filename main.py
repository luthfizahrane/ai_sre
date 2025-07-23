import os
from typing import Optional
import uuid
import shutil
import asyncio
import httpx # <-- Import httpx untuk mengunduh file
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request # <-- Import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel # <-- Import BaseModel untuk request body JSON

# Memuat variabel lingkungan
load_dotenv(find_dotenv())

# Mengimpor modul-modul yang diperlukan
from embedding_process import DocumentPreprocessor
from kg_processor import KnowledgeGraphProcessor

# LangChain components untuk GPT biasa dan Heading
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Izinkan semua asal untuk pengembangan. Sesuaikan di produksi.
    allow_credentials=True,
    allow_methods=["*"], # Izinkan semua metode (GET, POST, dll.)
    allow_headers=["*"], # Izinkan semua header
)

# --- INISIALISASI PROSESOR ---
doc_preprocessor = DocumentPreprocessor()
kg_processor = KnowledgeGraphProcessor()

# Direktori sementara untuk menyimpan file PDF yang diunggah
UPLOAD_TEMP_DIR = os.getenv('UPLOAD_TEMP_DIR', 'temp_uploads')
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True) # Pastikan direktori ada

# --- Pydantic Model untuk Request Body Baru ---
class IngestPDFPayload(BaseModel):
    pdf_url: str
    article_title: str
    session_id: Optional[str] = None # Untuk implementasi multi-sesi di masa depan
    user_id: Optional[str] = None    # Untuk implementasi multi-user di masa depan

@app.get("/")
async def root():
    return {"message": "Selamat datang di Knowledge Graph API!"}

# --- ENDPOINT /ingest_pdf DIREVISI ---
@app.post("/ingest_pdf")
async def ingest_pdf_endpoint(payload: IngestPDFPayload): # <-- Menerima Payload JSON
    """
    Endpoint untuk mengunggah file PDF, memprosesnya (embeddings & KG),
    dan menyimpan hasilnya ke Supabase (document_chunks & KG nodes/edges).
    Menerima URL PDF dari Supabase Storage yang dikirim oleh backend JS.
    """
    pdf_url = payload.pdf_url
    article_title = payload.article_title
    session_id = payload.session_id # Simpan untuk penggunaan di masa depan
    user_id = payload.user_id     # Simpan untuk penggunaan di masa depan

    # Buat article_id yang akan menjadi ID utama untuk Article, Node, Edge, document_chunks
    article_id = str(uuid.uuid4()) 
    
    # Dapatkan nama file dari URL untuk penyimpanan sementara
    file_name = os.path.basename(pdf_url).split('?')[0] # Hapus query params dari URL
    if not file_name.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="URL tidak mengarah ke file PDF.")

    temp_file_path = os.path.join(UPLOAD_TEMP_DIR, f"{article_id}_{file_name}")

    try:
        # 1. Unduh file PDF dari Supabase Storage ke lokal
        print(f"Mengunduh PDF dari {pdf_url} ke {temp_file_path}...")
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url, follow_redirects=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(temp_file_path, "wb") as buffer:
                buffer.write(response.content)
        print("Pengunduhan PDF berhasil.")

        # 2. Simpan metadata artikel ke tabel public."Article"
        # Membuka koneksi database secara langsung untuk insert ini
        conn_article = None
        try:
            conn_article = await kg_processor._get_pg_connection() # Re-use koneksi dari kg_processor
            await conn_article.execute("""
                INSERT INTO public."Article" (id, title, "filePath")
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO NOTHING;
            """, article_id, article_title, pdf_url) # filePath adalah URL publik
            print(f"Metadata Artikel '{article_title}' (ID: {article_id}) disimpan ke public.Article.")
        except Exception as e:
            print(f"Error menyimpan metadata artikel ke public.Article: {e}")
            raise HTTPException(status_code=500, detail=f"Gagal menyimpan metadata artikel: {e}")
        finally:
            if conn_article:
                await conn_article.close()

        # 3. Proses PDF untuk chunks & embeddings (disimpan ke 'public.document_chunks'), dan dapatkan teks lengkap
        full_document_content = await doc_preprocessor.process_and_embed_pdf(temp_file_path, article_id, article_title)

        if not full_document_content:
            raise HTTPException(status_code=500, detail=f"Tidak ada konten yang dimuat dari PDF: {article_title}")

        # 4. Proses dokumen untuk Knowledge Graph (disimpan ke 'public.Node' & 'public.Edge')
        kg_success = await kg_processor.process_document(full_document_content, article_id, article_title)

        if kg_success:
            return JSONResponse(content={"message": f"Dokumen '{article_title}' berhasil diunduh dan diproses untuk embeddings dan KG.", "article_id": article_id})
        else:
            raise HTTPException(status_code=500, detail="Gagal memproses dokumen untuk Knowledge Graph.")

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error saat mengunduh PDF: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Gagal mengunduh PDF dari URL: {e.response.status_code}")
    except httpx.RequestError as e:
        print(f"Network Error saat mengunduh PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal mengunduh PDF (masalah jaringan): {e}")
    except Exception as e:
        print(f"Error umum saat memproses PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses file: {e}")
    finally:
        # Hapus file sementara setelah diproses
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- Endpoint Chat Biasa (menggunakan Gemini Pro) ---
prompt_gpt = PromptTemplate(
    input_variables="question",
    template="""
 Anda adalah asisten AI yang membantu. Anda akan menjawab pertanyaan sesuai apa yang ditanyakan. Berikut pertanyaannya: {question}
""")
model_gpt = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8) 
ai_chain = prompt_gpt | model_gpt

@app.post("/ask_gpt")
async def ask_gpt_endpoint(question: str = Form(...)):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    response = await ai_chain.ainvoke({"question": question})
    structured_output = {
        "answer": response.content if hasattr(response, 'content') else str(response),
        "sources": [] 
    }
    return JSONResponse(content=structured_output)

# --- Endpoint Rekomendasi Heading ---
prompt_heading = PromptTemplate(
    input_variables=["question"],
    template="""
Anda adalah asisten AI yang dapat merekomendasikan struktur dokumen. 
Anda akan memberikan rekomendasi heading maupun sub-heading berdasarkan topik yang diinginkan.

berikut topik yang ingin dibahas :{question}

output yang diharapkan sebagai berikut :

rekomendasi sub-judul dari topik tersebut adalah :
1. Cara Meningkatkan Minat Belajar
2. Manfaat dari Penggunaan AI untuk Belajar
3. Metode Flipped Classroom untuk Minat Belajar

pastikan output yang dihasilkan minimal 3 poin dan maksimal 10 poin
"""
)
model_heading = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)
heading_chain = prompt_heading | model_heading

@app.post("/ask_heading")
async def ask_heading_endpoint(question: str = Form(...)):
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    response = await heading_chain.ainvoke({"question": question})
    structured_output = {
        "answer": response.content if hasattr(response, 'content') else str(response)
    }
    return JSONResponse(content=structured_output)

# --- Endpoint Pengambilan Data Knowledge Graph ---
@app.get("/get_graph_data")
async def get_graph_data_endpoint(session_id: Optional[str] = None): # <-- Menerima session_id
    """
    Endpoint untuk mengambil semua data Knowledge Graph (nodes dan edges)
    untuk visualisasi. Data diambil dari Supabase (via kg_processor).
    """
    graph_data = await kg_processor.get_graph_data(session_id=session_id) # <-- Meneruskan session_id
    return JSONResponse(content=graph_data)

# Untuk menjalankan aplikasi ini, Anda perlu menginstal uvicorn:
# pip install uvicorn
# Lalu jalankan dari terminal:
# uvicorn main_gemini:app --reload --port 8090
