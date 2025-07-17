import os
import uuid
import shutil
import asyncio # Perlu import asyncio untuk menjalankan fungsi async
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv

# Memuat variabel lingkungan
load_dotenv(find_dotenv())

# Mengimpor modul-modul yang diperlukan
# Tidak ada lagi preprocess_module untuk embeddings/FAISS karena tidak ada RAG
# from preprocess_module import ingest_pdf, UPLOAD_TEMP_DIR 
from embedding_process import DocumentPreprocessor
from kg_processor import KnowledgeGraphProcessor # Mengimpor prosesor KG yang baru

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
doc_preprocessor = DocumentPreprocessor()
# Inisialisasi prosesor Knowledge Graph
kg_processor = KnowledgeGraphProcessor()

# Direktori sementara untuk menyimpan file PDF yang diunggah
UPLOAD_TEMP_DIR = os.getenv('UPLOAD_TEMP_DIR', 'temp_uploads')
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True) # Pastikan direktori ada

@app.get("/")
async def root():
    return {"message": "Selamat datang di Knowledge Graph API!"}

@app.post("/ingest_pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    """
    Endpoint untuk mengunggah file PDF, memprosesnya dengan LlamaIndex untuk ekstraksi KG,
    dan menyimpan hasilnya ke Supabase Pgvector.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diizinkan.")

    file_id = str(uuid.uuid4())
    temp_file_path = os.path.join(UPLOAD_TEMP_DIR, f"{file_id}_{file.filename}")
    
    

    try:
        print(f"[INFO] Mulai upload file PDF: {file.filename} (ID: {file_id})")
        # Simpan file yang diunggah sementara
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[INFO] File disimpan sementara di {temp_file_path}")

        # Baca konten PDF dari file yang disimpan
        # Menggunakan PyPDFLoader yang sama seperti sebelumnya, karena RAG dihapus
        # tapi pembacaan PDF tetap di Python
        print("[INFO] Mulai proses dan embedding PDF...")
        full_document_content = await doc_preprocessor.process_and_embed_pdf(temp_file_path, file_id, file.filename)
        print("[DEBUG] Hasil proses dan embedding, panjang konten:", len(full_document_content))

        if not full_document_content:
            print("[ERROR] Tidak ada konten yang dimuat dari PDF:", file.filename)
            raise HTTPException(status_code=500, detail=f"Tidak ada konten yang dimuat dari PDF: {file.filename}")

        # Gabungkan konten semua halaman menjadi satu string untuk ekstraksi KG
        # document_content_full = "\n\n".join([doc.page_content for doc in loaded_docs])
        
        # Proses dokumen untuk Knowledge Graph menggunakan kg_processor
        # Memanggil fungsi async dari kg_processor
        print("[INFO] Mulai proses Knowledge Graph...")
        kg_success = await kg_processor.process_document(full_document_content, file_id, file.filename)
        print(f"[DEBUG] Status proses KG: {kg_success}")

        if kg_success:
            print(f"[SUCCESS] Dokumen '{file.filename}' berhasil diproses untuk KG.")
            return JSONResponse(content={"message": f"Dokumen '{file.filename}' berhasil diunggah dan diproses untuk KG.", "document_id": file_id})
        else:
            print("[ERROR] Gagal memproses dokumen untuk Knowledge Graph.")
            raise HTTPException(status_code=500, detail="Gagal memproses dokumen untuk Knowledge Graph.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses file: {e}")
    finally:
        # Hapus file sementara setelah diproses
        if os.path.exists(temp_file_path):
            print(f"[INFO] Hapus file sementara: {temp_file_path}")
            os.remove(temp_file_path)

from langchain_google_genai import ChatGoogleGenerativeAI # <-- Perubahan Impor
from langchain.prompts import PromptTemplate
# Hapus endpoint RAG karena tidak digunakan
# @app.post("/rag")
# async def rag_endpoint(question: str = Form(...)):
#     """Endpoint RAG (dihapus)."""
#     raise HTTPException(status_code=405, detail="RAG endpoint has been removed.")

# Endpoint untuk GPT biasa (jika masih diperlukan)
# Anda bisa memindahkan logika ini ke kg_processor jika ingin sentralisasi LLM di satu tempat
# Atau biarkan di sini jika sederhana. Untuk demo, saya akan hapus impor dari rag_module.
# Anda bisa mengimplementasikannya kembali secara langsung jika mau.

# from langchain.chains import LLMChain

prompt_gpt = PromptTemplate(
    input_variables="question",
    template="""
Anda adalah asisten AI yang membantu. Anda akan menjawab pertanyaan sesuai apa yang ditanyakan. Berikut pertanyaannya: {question}
""")
model_gpt = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8) # tambahkan max_tokens=1000
ai_chain = prompt_gpt | model_gpt

@app.post("/ask_gpt")
async def ask_gpt_endpoint(question: str = Form(...)):
    """
    Endpoint untuk berinteraksi dengan GPT biasa.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    response = await ai_chain.ainvoke({"question": question}) # Menggunakan arun untuk async
    structured_output = {
        "answer": response.replace('\n', '<br />').replace('\"', '')
    }
    return JSONResponse(content=structured_output)


# Endpoint untuk rekomendasi heading (jika masih diperlukan)
# Logika ini juga bisa disederhanakan tanpa retriever jika RAG tidak digunakan
# atau Anda bisa membuat prompt yang hanya berdasarkan 'question' tanpa 'context'
# jika konteks dari dokumen tidak diperlukan untuk rekomendasi heading
@app.post("/ask_heading")
async def ask_heading_endpoint(question: str = Form(...)):
    """
    Endpoint untuk mendapatkan rekomendasi heading/sub-heading.
    """
    # Karena tidak ada RAG, kita tidak bisa mengambil 'relevant documents' lagi
    # Anda perlu memutuskan apakah rekomendasi heading akan berdasarkan:
    # 1. Seluruh KG yang ada (jika ini relevan)
    # 2. Hanya pertanyaan itu sendiri (seperti GPT biasa)
    # 3. Konten dari dokumen tertentu (jika ID dokumen dikirim)

    # Untuk demo, kita akan membuat rekomendasi heading hanya berdasarkan pertanyaan
    # tanpa konteks spesifik dokumen dari FAISS/RAG.
    # Jika Anda ingin berdasarkan KG, Anda perlu memodifikasi ini untuk query KG.
    
    prompt_heading = PromptTemplate(
        input_variables=["question"],
        template="""
    anda adalah asisten AI yang dapat merekomendasikan struktur dokumen. 
    anda akan memberikan rekomendasi heading maupun sub-heading berdasarkan topik yang diinginkan.

    berikut topik yang ingin dibahas :{question}
    
    output yang diharapkan sebagai berikut :
    
    rekomendasi sub-judul dari topik tersebut adalah :
    1. Cara Meningkatkan Minat Belajar
    2. Manfaat dari Penggunaan AI untuk Belajar
    3. Metode Flipped Classroom untuk Minat Belajar

    pastikan output yang dihasilkan minimal 3 poin dan maksimal 10 poin
    """
    )
    model_heading = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.6) # tambahkan max_tokens=1000 apabila perlu
    heading_chain = prompt_heading | model_heading

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    response = await heading_chain.invoke({"question": question})
    structured_output = {
        "answer": response.replace('\n', '<br />').replace('\"', '')
    }
    return JSONResponse(content=structured_output)


@app.get("/get_graph_data")
async def get_graph_data_endpoint():
    """
    Endpoint untuk mengambil semua data Knowledge Graph (nodes dan edges)
    untuk visualisasi. Data diambil dari Supabase (via kg_processor).
    """
    graph_data = await kg_processor.get_graph_data() # Memanggil fungsi async
    return JSONResponse(content=graph_data)

# Untuk menjalankan aplikasi ini, Anda perlu menginstal uvicorn:
# pip install uvicorn
# Lalu jalankan dari terminal:
# uvicorn main:app --reload --port 8090