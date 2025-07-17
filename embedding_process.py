import os
import uuid
import asyncio
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings # <-- Perubahan Impor
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Untuk Pgvector Supabase
import asyncpg
from pgvector.asyncpg import register_vector
from supabase import create_client, Client

# Memuat variabel lingkungan
load_dotenv(find_dotenv())

# --- Konfigurasi Google Gemini ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY') # <-- Perubahan Nama Variabel
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY tidak diatur dalam variabel lingkungan.")

# --- Konfigurasi Supabase DB (untuk Pgvector) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

SUPABASE_DB_HOST = os.environ.get("SUPABASE_DB_HOST")
SUPABASE_DB_PORT = os.environ.get("SUPABASE_DB_PORT")
SUPABASE_DB_USER = os.environ.get("SUPABASE_DB_USER")
SUPABASE_DB_PASSWORD = os.environ.get("SUPABASE_DB_PASSWORD")
SUPABASE_DB_NAME = os.environ.get("SUPABASE_DB_NAME")

if not all([SUPABASE_DB_HOST, SUPABASE_DB_PORT, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD, SUPABASE_DB_NAME]):
    raise ValueError("Pastikan semua variabel lingkungan Supabase DB (host, port, user, pass, name) diatur.")

# Inisialisasi Embeddings Google Generative AI
embeddings_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001") # <-- Perubahan Inisialisasi Model

class DocumentPreprocessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self._ensure_db_table_exists()

    async def _get_pg_connection(self):
        print("Connecting to DB with:")
        print("HOST:", SUPABASE_DB_HOST)
        print("PORT:", SUPABASE_DB_PORT)
        print("USER:", SUPABASE_DB_USER)
        print("PASSWORD:", SUPABASE_DB_PASSWORD)
        print("DBNAME:", SUPABASE_DB_NAME)
        conn = await asyncpg.connect(
            host=SUPABASE_DB_HOST,
            port=int(SUPABASE_DB_PORT),
            user=SUPABASE_DB_USER,
            password=SUPABASE_DB_PASSWORD,
            database=SUPABASE_DB_NAME
        )
        await register_vector(conn)
        return conn

    def _ensure_db_table_exists(self):
        print("Memastikan tabel 'document_chunks' ada di Supabase...")
        # CATATAN PENTING: Ubah dimensi VECTOR di SQL menjadi 768 untuk model Gemini
        # SQL untuk membuat tabel:
        # CREATE EXTENSION IF NOT EXISTS vector;
        # CREATE TABLE document_chunks (
        #     id TEXT PRIMARY KEY,
        #     doc_id TEXT NOT NULL,
        #     chunk_text TEXT,
        #     page_number INTEGER,
        #     embedding VECTOR(768), -- <--- UBAH DIMENSI MENJADI 768
        #     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        # );
        print("Pastikan tabel 'document_chunks' telah dibuat secara manual di dashboard Supabase Anda dengan dimensi embedding 768.")

    async def process_and_embed_pdf(self, file_path: str, document_id: str, document_title: str) -> str:
        print(f"Memproses dan membuat embedding untuk PDF: {document_title} (ID: {document_id})")
        
        loader = PyPDFLoader(file_path)
        loaded_docs = loader.load_and_split()

        if not loaded_docs:
            print(f"Tidak ada konten yang dimuat dari PDF: {document_title}")
            return ""

        full_document_content = ""
        chunks_to_insert = []
        
        for i, doc in enumerate(loaded_docs):
            full_document_content += doc.page_content + "\n\n"
            
            page_chunks = self.text_splitter.split_text(doc.page_content)
            
            for j, chunk_text in enumerate(page_chunks):
                chunk_unique_id = f"{document_id}-{doc.metadata.get('page', i)}-{j}"
                
                embedding_vector = await embeddings_model.aembed_query(chunk_text)
                # Tambahkan print untuk memverifikasi dimensi embedding
                print(f"DEBUG: Embedding dimension for chunk '{chunk_unique_id}': {len(embedding_vector)}")

                chunks_to_insert.append({
                    "id": chunk_unique_id,
                    "doc_id": document_id,
                    "chunk_text": chunk_text,
                    "page_number": doc.metadata.get('page', i),
                    "embedding": embedding_vector
                })

        conn = None
        try:
            conn = await self._get_pg_connection()
            for chunk in chunks_to_insert:
                await conn.execute("""
                    INSERT INTO document_chunks (id, doc_id, chunk_text, page_number, embedding)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO NOTHING;
                """, chunk["id"], chunk["doc_id"], chunk["chunk_text"], chunk["page_number"], chunk["embedding"])
            print(f"Berhasil menyimpan {len(chunks_to_insert)} chunks ke Supabase 'document_chunks'.")
        except Exception as e:
            print(f"Error menyimpan chunks ke Supabase: {e}")
            raise e
        finally:
            if conn:
                await conn.close()

        return full_document_content
