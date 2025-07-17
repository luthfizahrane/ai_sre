import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

# LangChain Imports
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Pydantic untuk mendefinisikan skema output ekstraksi
from pydantic import BaseModel, Field, ValidationError

# Supabase Client
from supabase import create_client, Client

# PostgreSQL database client untuk Pgvector setup
import asyncpg
from pgvector.asyncpg import register_vector

# Memuat variabel lingkungan
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# --- Konfigurasi Umum ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY tidak diatur dalam variabel lingkungan.")

# Konfigurasi Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_DB_HOST = os.environ.get("SUPABASE_DB_HOST")
SUPABASE_DB_PORT = os.environ.get("SUPABASE_DB_PORT")
SUPABASE_DB_USER = os.environ.get("SUPABASE_DB_USER")
SUPABASE_DB_PASSWORD = os.environ.get("SUPABASE_DB_PASSWORD")
SUPABASE_DB_NAME = os.environ.get("SUPABASE_DB_NAME")

if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_DB_HOST, SUPABASE_DB_PORT, SUPABASE_DB_USER, SUPABASE_DB_PASSWORD, SUPABASE_DB_NAME]):
    raise ValueError("Pastikan semua variabel lingkungan Supabase diatur.")

# Inisialisasi LLM dan Embeddings
llm_kg = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.5-flash", temperature=0)
embeddings_for_retrieval = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
embeddings_kg_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

# --- Skema Pydantic untuk Ekstraksi Knowledge Graph (1 Node per Artikel, Relasi Antar Artikel) ---

class ArticleContentSections(BaseModel):
    """Informasi terstruktur dari bagian-bagian artikel, disimpan dalam JSONB."""
    background: Optional[str] = Field(None, description="Isi latar belakang penelitian.")
    methodology: Optional[str] = Field(None, description="Metodologi penelitian yang digunakan.")
    purpose: Optional[str] = Field(None, description="Tujuan penelitian ini.")
    future_research: Optional[str] = Field(None, description="Rekomendasi untuk penelitian lebih lanjut.")
    research_gap: Optional[str] = Field(None, description="Kesenjangan penelitian atau batasan yang diidentifikasi.")

class ArticleMainNode(BaseModel):
    """Representasi node utama untuk sebuah artikel penelitian."""
    id: str = Field(description="ID unik untuk node artikel utama, format: doc_<doc_id>")
    title: str = Field(description="Judul dokumen penelitian ini.")
    summary: str = Field(description="Ringkasan singkat keseluruhan dokumen.")
    # extracted_details akan disimpan sebagai JSONB di DB
    extracted_details: ArticleContentSections = Field(description="Informasi terstruktur dari bagian-bagian artikel.")

# Definisikan Edges (Relasi) yang hanya antar Nodes Artikel
class Relation(BaseModel):
    source_id: str = Field(description="ID node artikel sumber relasi (format: doc_<doc_id>)")
    target_id: str = Field(description="ID node artikel target relasi (format: doc_<doc_id>)")
    type: str = Field(description="Tipe relasi (e.g., 'SERUPA_LATAR_BELAKANG', 'SERUPA_TUJUAN', 'SERUPA_METODOLOGI', 'SERUPA_PENELITIAN_LANJUT', 'SERUPA_GAP_PENELITIAN'). Gunakan format UPPER_SNAKE_CASE.")
    context: str = Field(description="Penjelasan singkat mengapa relasi ini ada.")

# Definisikan keseluruhan struktur output KG
class KnowledgeGraphOutput(BaseModel):
    """Representasi Knowledge Graph dari dokumen penelitian."""
    article_node: ArticleMainNode = Field(description="Node utama untuk artikel yang sedang diproses.")
    relations: List[Relation] = Field(default_factory=list, description="Relasi hanya antar node artikel.")

# --- Kelas Prosesor Knowledge Graph ---
class KnowledgeGraphProcessor:
    def __init__(self):
        self.supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Prompt dimodifikasi untuk meminta 1 node utama per artikel dan relasi antar artikel
        self.kg_extraction_prompt = PromptTemplate(
            input_variables=["retrieved_context", "document_id", "document_title", "existing_nodes_json", "format_instruction"],
            template="""
            Anda adalah agen ekstraksi informasi ahli yang membangun knowledge graph.
            Tugas Anda adalah mengekstrak satu Node utama yang merepresentasikan artikel penelitian INI berdasarkan KONTEKS yang DIBERIKAN.
            Kemudian, identifikasi relasi hanya antara node artikel INI dengan node artikel lain yang sudah ada dalam knowledge graph.

            Anda TIDAK AKAN DIBERIKAN dokumen penuh. Anda harus bekerja HANYA dari KONTEKS yang telah diambil.
            Jika informasi yang diminta tidak ada dalam konteks yang diberikan, biarkan kosong. Jangan membuat relasi ke node selain node artikel.

            Untuk node artikel utama, ekstrak:
            -   ID unik (gunakan format: 'doc_<document_id>')
            -   Judul dokumen
            -   Ringkasan singkat keseluruhan dokumen (maksimal 500 kata)
            -   Informasi terstruktur dari bagian Latar Belakang, Metodologi, Tujuan, Penelitian Lanjut, dan Gap Penelitian.

            Untuk relasi (hanya antar node artikel):
            -   Identifikasi relasi yang relevan antara node artikel utama INI ('doc_{document_id}') dengan node artikel lain yang sudah ada dalam 'existing_nodes_json'.
            -   Fokuskan relasi pada kemiripan dalam 5 poin utama: Latar Belakang Penelitian, Tujuan Penelitian, Metodologi Penelitian, Penelitian Lanjut, dan Gap Penelitian.
            -   Untuk setiap kemiripan yang ditemukan, buat satu relasi dengan Tipe spesifik:
                -   'SERUPA_LATAR_BELAKANG' jika latar belakangnya mirip.
                -   'SERUPA_TUJUAN' jika tujuannya mirip.
                -   'SERUPA_METODOLOGI' jika metodologinya mirip.
                -   'SERUPA_PENELITIAN_LANJUT' jika rekomendasi penelitian lanjutnya mirip.
                -   'SERUPA_GAP_PENELITIAN' jika gap penelitiannya mirip.
            -   Gunakan 'context' untuk menjelaskan mengapa relasi tersebut ada.
            -   Hanya buat relasi jika ada kemiripan yang jelas di salah satu dari 5 poin tersebut.
            -   Pastikan source_id dan target_id selalu dalam format 'doc_<doc_id>'. Jangan membuat relasi ke penulis, konsep, atau entitas lain.

            Berikut adalah KONTEKS yang telah diambil dari dokumen yang akan dianalisis:
            ### Dokumen Baru (Konteks Terambil)
            Judul Dokumen: {document_title}
            ID Dokumen: {document_id}
            Konteks Terambil: {retrieved_context}

            ### Nodes Artikel yang Sudah Ada (untuk referensi relasi)
            Ini adalah daftar nodes artikel lain yang sudah ada dalam database Anda. Gunakan ini untuk menemukan relasi antarartikel.
            {existing_nodes_json}

            Output Anda harus dalam format JSON yang sesuai dengan skema Pydantic berikut.
            Pastikan output adalah JSON murni, tanpa markdown code block (```json).
            {format_instruction}
            """,
        )
        self.kg_extraction_chain = self.kg_extraction_prompt | llm_kg 
        self._ensure_db_tables_exist()

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

    def _ensure_db_tables_exist(self):
        print("Memastikan tabel 'nodes_kg' dan 'edges_kg' ada di Supabase...")
        # CATATAN PENTING: Skema SQL untuk nodes_kg perlu diubah untuk kolom JSONB dan tanpa nodes non-artikel
        # Ini adalah skema final yang harus Anda buat/sesuaikan di Supabase
        # CREATE EXTENSION IF NOT EXISTS vector; (Jika belum aktif)
        # CREATE TABLE nodes_kg (
        #     id TEXT PRIMARY KEY,               -- ID unik untuk node artikel utama (doc_<doc_id>)
        #     type TEXT DEFAULT 'Artikel',       -- Akan selalu 'Artikel' atau 'Dokumen'
        #     title TEXT,                        -- Judul dokumen
        #     content_summary TEXT,              -- Ringkasan dokumen
        #     extracted_details JSONB,           -- Detail terstruktur dari bagian artikel (background, method, dll.)
        #     source_doc_id TEXT,                -- ID unik dokumen (sama dengan id node ini)
        #     source_doc_title TEXT,             -- Judul dokumen (sama dengan title node ini)
        #     embedding VECTOR(768),             -- Embedding dari ringkasan dokumen
        #     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        # );
        #
        # SQL untuk tabel edges_kg (tidak berubah, id otomatis):
        # CREATE TABLE edges_kg (
        #     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        #     source_id TEXT REFERENCES nodes_kg(id) ON DELETE CASCADE, -- Merujuk ke id node artikel
        #     target_id TEXT REFERENCES nodes_kg(id) ON DELETE CASCADE, -- Merujuk ke id node artikel
        #     type TEXT,                                               -- Tipe relasi (e.g., SERUPA_LATAR_BELAKANG)
        #     context TEXT,
        #     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        # );
        print("Pastikan tabel 'nodes_kg' telah diubah untuk memiliki kolom 'extracted_details' dengan tipe JSONB dan dimensi embedding 768. Dan pastikan edges_kg merujuk ke nodes_kg.")

    async def _get_existing_nodes(self) -> List[Dict[str, Any]]:
        """
        Mengambil semua nodes ARTIKEL yang sudah ada dari tabel 'nodes_kg' di Supabase.
        Ambil properti yang dibutuhkan LLM untuk identifikasi relasi.
        """
        try:
            # Ambil id, title, dan extracted_details dari nodes bertipe 'Dokumen'/'Artikel'
            response = self.supabase_client.from_('nodes_kg').select('id, title, extracted_details').eq('type', 'Dokumen').execute() 
            if response.data:
                # LLM membutuhkan extracted_details sebagai objek JSON, bukan string
                return [{**item, 'extracted_details': json.dumps(item['extracted_details']) if isinstance(item['extracted_details'], dict) else item['extracted_details']} for item in response.data]
            return []
        except Exception as e:
            print(f"Error fetching existing nodes from Supabase: {e}")
            return []
            
    async def process_document(self, full_document_content: str, document_id: str, document_title: str) -> bool:
        print(f"Memulai ekstraksi Knowledge Graph untuk dokumen '{document_title}' (ID: {document_id}) dengan RAG-driven approach...")

        conn = None # Inisialisasi koneksi PG untuk retrieval
        retrieved_context = ""
        try:
            conn = await self._get_pg_connection()

            # 1. Buat kueri internal dari dokumen baru
            query_text = f"Judul: {document_title}. Ringkasan awal: {full_document_content[:500]}..."
            
            # 2. Buat embedding dari kueri internal
            query_embedding = await embeddings_for_retrieval.aembed_query(query_text)
            
            # 3. Lakukan vector search di tabel document_chunks untuk dokumen ini
            # Penting: Pastikan chunks dari dokumen ini sudah masuk ke DB sebelum ini dipanggil
            retrieval_results = await conn.fetch("""
                SELECT chunk_text, embedding <-> $1 AS distance
                FROM document_chunks
                WHERE doc_id = $2
                ORDER BY distance
                LIMIT 10;
            """, query_embedding, document_id)

            if retrieval_results:
                retrieved_context = "\n\n".join([r['chunk_text'] for r in retrieval_results])
                print(f"[kg_processor] Berhasil mengambil {len(retrieval_results)} chunks untuk konteks.")
            else:
                # Fallback: Jika tidak ada chunks yang diambil (misal dokumen baru belum masuk chunks),
                retrieved_context = full_document_content[:2000]
                print("[kg_processor] Tidak ada chunks yang diambil, menggunakan bagian awal dokumen penuh sebagai fallback.")

        except Exception as e:
            print(f"[kg_processor] Error saat melakukan retrieval dari document_chunks: {e}")
            retrieved_context = full_document_content[:2000] 
            print("[kg_processor] Retrieval gagal, menggunakan bagian awal dokumen penuh sebagai fallback.")
        finally:
            if conn: # Pastikan koneksi ditutup jika berhasil dibuka
                await conn.close()

        # Ambil nodes artikel yang sudah ada untuk konteks LLM
        # Setelah retrieval, buka koneksi baru jika yang sebelumnya ditutup
        existing_nodes = await self._get_existing_nodes()
        existing_nodes_json = json.dumps(existing_nodes, indent=2)
        
        format_instruction = json.dumps(KnowledgeGraphOutput.model_json_schema(), indent=2)

        try:
            print("[kg_processor] Mulai ekstraksi KG dengan LLM...")
            input_data = {
                "retrieved_context": retrieved_context,
                "document_id": document_id,
                "document_title": document_title,
                "existing_nodes_json": existing_nodes_json,
                "format_instruction": format_instruction
            }
            raw_kg_output_message = await self.kg_extraction_chain.ainvoke(input_data)
            
            raw_kg_output_str = raw_kg_output_message.content
            print("[kg_processor] Output LLM:", raw_kg_output_str)
            
            json_string_to_parse = raw_kg_output_str.strip().lstrip('```json').rstrip('```').strip()
            
            kg_output_dict = json.loads(json_string_to_parse)
            
            kg_output = KnowledgeGraphOutput.model_validate(kg_output_dict)
            print("[kg_processor] Output KG tervalidasi.")
            
            print("Ekstraksi Knowledge Graph oleh LangChain berhasil.")
        except ValidationError as e:
            print(f"[kg_processor] Pydantic Validation Error: {e.errors()}")
            print(f"Raw LLM output (for validation error): {raw_kg_output_str}")
            return False
        except json.JSONDecodeError as e:
            print(f"[kg_processor] Error decoding JSON from LLM output: {e}")
            print(f"Raw LLM output: {raw_kg_output_str}")
            return False
        except Exception as e:
            print(f"Error saat ekstraksi Knowledge Graph dengan LangChain: {e}")
            return False

        conn = None # Re-inisialisasi conn untuk bagian penyimpanan
        try:
            conn = await self._get_pg_connection()

            # --- LOGIKA PENYIMPANAN NODE UTAMA ARTIKEL ---
            main_article_node = kg_output.article_node

            article_embedding_content = main_article_node.summary if main_article_node.summary else main_article_node.title
            article_node_embedding = await embeddings_kg_model.aembed_query(article_embedding_content)
            
            await conn.execute("""
                INSERT INTO nodes_kg (id, type, title, content_summary, extracted_details, source_doc_id, source_doc_title, embedding)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
                ON CONFLICT (id) DO NOTHING;
            """, 
                main_article_node.id, 
                "Artikel", # Tipe node utama (gunakan 'Artikel' atau 'Dokumen')
                main_article_node.title, 
                main_article_node.summary, 
                json.dumps(main_article_node.extracted_details.dict()), # Simpan sebagai JSONB string
                document_id, # source_doc_id
                document_title, # source_doc_title
                article_node_embedding
            )
            print(f"Node Artikel Utama '{main_article_node.id}' diproses.")

            # --- LOGIKA PENYIMPANAN EDGES ---
            # Edges sekarang hanya akan dibuat antar node artikel
            for relation_data in kg_output.relations:
                # Pastikan source_id dan target_id adalah ID node artikel
                # LLM seharusnya sudah menghasilkan format 'doc_<doc_id>'
                source_exists_q = await conn.fetchval("SELECT id FROM nodes_kg WHERE id = $1", relation_data.source_id)
                target_exists_q = await conn.fetchval("SELECT id FROM nodes_kg WHERE id = $1", relation_data.target_id)

                # Hanya insert jika kedua node (artikel) ada
                if source_exists_q and target_exists_q:
                    await conn.execute("""
                        INSERT INTO edges_kg (source_id, target_id, type, context)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (id) DO NOTHING; -- Asumsi ID UUID otomatis di edges_kg
                    """, relation_data.source_id, relation_data.target_id, relation_data.type, relation_data.context)
                    print(f"Edge '{relation_data.source_id}' -[{relation_data.type}]-> '{relation_data.target_id}' diproses.")
                else:
                    print(f"Melewatkan edge antara '{relation_data.source_id}' dan '{relation_data.target_id}' karena satu atau kedua node artikel tidak ditemukan di DB.")
            
            return True

        except Exception as e:
            print(f"Error menyimpan KG ke Supabase: {e}")
            return False
        finally:
            if conn: # Pastikan koneksi ditutup jika berhasil dibuka di try blok ini
                await conn.close()


    async def get_graph_data(self) -> Dict[str, Any]:
        nodes_data = []
        edges_data = []
        try:
            # Ambil kolom yang relevan dari nodes_kg, termasuk extracted_details
            nodes_response = self.supabase_client.from_('nodes_kg').select('id, type, title, content_summary, extracted_details, source_doc_id, source_doc_title').execute()
            if nodes_response.data:
                nodes_data = nodes_response.data
            
            # Mengambil edges (tetap sama)
            edges_response = self.supabase_client.from_('edges_kg').select('source_id, target_id, type, context').execute()
            if edges_response.data:
                edges_data = edges_response.data

            print(f"Berhasil mengambil {len(nodes_data)} nodes dan {len(edges_data)} edges dari Supabase.")
            return {"nodes": nodes_data, "edges": edges_data}
        except Exception as e:
            print(f"Error fetching graph data from Supabase: {e}")
            return {"nodes": [], "edges": []}

