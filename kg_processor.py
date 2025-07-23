import os
import json
import asyncio
import re
import uuid # Diperlukan untuk ID Edge
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
llm_kg = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-2.5-pro", temperature=0.2) # Gunakan gemini-pro untuk penalaran yang kuat
embeddings_for_retrieval = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
embeddings_kg_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")


# --- Skema Pydantic untuk Ekstraksi KG (Mencerminkan Struktur Kolom DB JS) ---
# ArticleContentSections ini adalah detail yang diekstrak dan akan di-flatten
# menjadi kolom att_goal, att_method, dll. di Node table.
# LLM akan mengekstraknya sebagai satu objek, lalu Python akan memecahnya.
class ArticleContentSections(BaseModel):
    background: Optional[str] = Field(None, description="Isi latar belakang penelitian.")
    methodology: Optional[str] = Field(None, description="Metodologi penelitian yang digunakan.")
    purpose: Optional[str] = Field(None, description="Tujuan penelitian ini.")
    future_research: Optional[str] = Field(None, description="Rekomendasi untuk penelitian lebih lanjut.")
    research_gap: Optional[str] = Field(None, description="Kesenjangan penelitian atau batasan yang diidentifikasi.")

# ArticleMainNode: Mencerminkan kolom tabel public."Node"
class ArticleMainNode(BaseModel):
    id: str = Field(description="ID node (e.g., doc_<UUID>)")
    label: str = Field(description="Label visual node (e.g., 'Artikel')")
    title: Optional[str] = Field(None, description="Judul artikel")
    att_goal: Optional[str] = Field(None, description="Tujuan penelitian (dari purpose)")
    att_method: Optional[str] = Field(None, description="Metodologi penelitian (dari methodology)")
    att_background: Optional[str] = Field(None, description="Latar belakang penelitian (dari background)")
    att_future: Optional[str] = Field(None, description="Saran penelitian lanjutan (dari future_research)")
    att_gaps: Optional[str] = Field(None, description="Gap penelitian (dari research_gap)")
    att_url: Optional[str] = Field(None, description="URL file artikel") # filePath dari Article
    type: str = Field(description="Tipe node (e.g., 'article' dari skema JS)") # Untuk skema JS
    content: str = Field(description="Ringkasan konten artikel (dari summary)") # Menggunakan 'content' sesuai JS
    articleId: str = Field(description="ID artikel utama (FK ke Article.id)") # Dari tabel Article JS

# Relation: Mencerminkan kolom tabel public."Edge"
class Relation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID unik edge (akan di-generate UUID)") # ID unik untuk Edge
    fromId: str = Field(description="ID node sumber")
    toId: str = Field(description="ID node target")
    relation: str = Field(description="Tipe relasi (e.g., SERUPA_LATAR_BELAKANG)")
    label: Optional[str] = Field(None, description="Label visual edge") # Untuk visualisasi
    color: Optional[str] = Field(None, description="Warna visual edge") # Untuk visualisasi
    articleId: str = Field(description="ID artikel utama yang terkait dengan edge") # Dari tabel Article JS

# KnowledgeGraphOutput tetap hanya berisi ArticleMainNode
class KnowledgeGraphOutput(BaseModel):
    article_node: ArticleMainNode = Field(description="Node utama untuk artikel yang sedang diproses.")


# --- Fungsi Pembantu untuk Nama Kolom CamelCase (Sesuai Skema JS) ---
def to_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def to_snake_case(camel_str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# --- Utility untuk mapping warna relasi (bisa ditempatkan di tempat lain) ---
RELATION_COLORS = {
    'SERUPA_LATAR_BELAKANG': '#1f77b4', # Biru
    'SERUPA_TUJUAN': '#ff7f0e',        # Oranye
    'SERUPA_METODOLOGI': '#2ca02c',     # Hijau
    'SERUPA_PENELITIAN_LANJUT': '#d62728', # Merah
    'SERUPA_GAP_PENELITIAN': '#9467bd', # Ungu
    'MENGUTIP': '#8c564b',              # Coklat
    'TERKAIT_DENGAN': '#e377c2',        # Pink
    'DEFAULT': '#7f7f7f'                # Abu-abu
}
def get_relation_color(relation_type: str) -> str:
    return RELATION_COLORS.get(relation_type, RELATION_COLORS['DEFAULT'])

# --- Kelas Prosesor Knowledge Graph ---
class KnowledgeGraphProcessor:
    def __init__(self):
        self.supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # PROMPT UTAMA HANYA UNTUK EKSTRAKSI NODE ARTIKEL UTAMA
        self.kg_extraction_prompt = PromptTemplate(
            input_variables=["retrieved_context", "document_id", "document_title", "existing_nodes_json", "format_instruction"],
            template="""
            Anda adalah agen ekstraksi informasi ahli yang membangun knowledge graph untuk penelitian akademik.
            
            TUGAS UTAMA:
            Ekstrak informasi terstruktur untuk node artikel BARU berdasarkan KONTEKS yang DIBERIKAN.
            Output Anda HARUS berupa data JSON yang valid sesuai skema yang diberikan, BUKAN definisi skemanya.
            Pastikan untuk mengekstrak secara eksplisit:
            - Latar Belakang Penelitian
            - Metodologi Penelitian
            - Tujuan Penelitian
            - Future Research
            - Gap Penelitian
            Jika informasi tidak ada, isi dengan 'null' atau kosong (sesuai definisi skema) daripada 'Tidak ditemukan'.

            KONTEKS ARTIKEL BARU:
            Judul: {document_title}
            ID: {document_id}
            Konten: {retrieved_context}
            
            ARTIKEL YANG SUDAH ADA (untuk referensi, bukan untuk membuat relasi otomatis di sini):
            {existing_nodes_json}
            
            INSTRUKSI EKSTRAKSI NODE ARTIKEL BARU:
            Untuk 'article_node' yang sedang diproses, ekstrak:
            - ID: 'doc_{document_id}'
            - Judul: {document_title}
            - Summary: Ringkasan komprehensif maksimal 500 kata dari seluruh dokumen.
            - Extracted Details:
              * background: Latar belakang penelitian dan motivasi
              * methodology: Metode penelitian yang digunakan
              * purpose: Tujuan dan objektif penelitian
              * future_research: Saran untuk penelitian masa depan
              * research_gap: Gap atau keterbatasan yang diidentifikasi
            Jika suatu detail tidak ditemukan, biarkan sebagai null.
            
            CONTOH OUTPUT DATA (Ikuti format ini persis):
            ```json
            {{
              "article_node": {{
                "id": "doc_example_id_123",
                "label": "Artikel",
                "title": "Contoh Judul Artikel",
                "att_goal": "Tujuan contoh",
                "att_method": "Metodologi contoh",
                "att_background": "Latar belakang contoh.",
                "att_future": "Saran penelitian contoh.",
                "att_gaps": "Gap penelitian contoh.",
                "att_url": "[https://example.com/contoh.pdf](https://example.com/contoh.pdf)",
                "type": "article",
                "content": "Ini adalah ringkasan contoh dari artikel tersebut.",
                "articleId": "example_article_id_456"
              }}
            }}
            ```

            Output harus JSON murni sesuai skema:
            {format_instruction}
            """,
        )
        self.kg_extraction_chain = self.kg_extraction_prompt | llm_kg 

        # --- PROMPT BARU UNTUK PEMBUATAN RELASI PER ASPEK SECARA EKSPLISIT OLEH LLM ---
        self.relation_prompts_map = {
            "SERUPA_LATAR_BELAKANG": PromptTemplate(
                input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
                template="""
                Anda adalah agen pembuat relasi knowledge graph.
                Tentukan apakah LATAR BELAKANG kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_LATAR_BELAKANG'.
                Output Anda HARUS berupa JSON yang valid, BUKAN teks percakapan, penjelasan, atau Markdown code block.

                CONTOH OUTPUT DATA MIRIP:
                ```json
                {{
                  "type": "SERUPA_LATAR_BELAKANG",
                  "context": "Latar belakang kedua artikel memiliki kemiripan kuat dalam membahas perlunya strategi pembelajaran inovatif di era digital."
                }}
                ```
                CONTOH OUTPUT DATA TIDAK MIRIP:
                ```json
                {{
                  "type": null,
                  "context": "Latar belakang kedua artikel tidak memiliki kemiripan signifikan; satu membahas ... dan yang lain membahas ..."
                }}
                ```
                
                Artikel 1 (ID: {article1_id}) Latar Belakang: {{article1_details.att_background}}
                Artikel 2 (ID: {article2_id}) Latar Belakang: {{article2_details.att_background}}

                Output JSON harus sesuai skema Pydantic berikut:
                {format_instruction}
                """,
            ),
            "SERUPA_TUJUAN": PromptTemplate(
                input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
                template="""
                Anda adalah agen pembuat relasi knowledge graph.
                Tentukan apakah TUJUAN kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_TUJUAN'.
                Output Anda HARUS berupa JSON yang valid, BUKAN teks percakapan, penjelasan, atau Markdown code block.

                CONTOH OUTPUT DATA MIRIP:
                ```json
                {{
                  "type": "SERUPA_TUJUAN",
                  "context": "Kedua artikel memiliki tujuan yang sama yaitu mengukur efektivitas model pembelajaran X terhadap hasil belajar Y."
                }}
                ```
                CONTOH OUTPUT DATA TIDAK MIRIP:
                ```json
                {{
                  "type": null,
                  "context": "Tujuan kedua artikel tidak memiliki kemiripan signifikan; satu bertujuan ... dan yang lain bertujuan ..."
                }}
                ```

                Artikel 1 (ID: {article1_id}) Tujuan: {{article1_details.att_purpose}}
                Artikel 2 (ID: {article2_id}) Tujuan: {{article2_details.att_purpose}}

                Output JSON harus sesuai skema Pydantic berikut:
                {format_instruction}
                """,
            ),
            "SERUPA_METODOLOGI": PromptTemplate(
                input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
                template="""
                Anda adalah agen pembuat relasi knowledge graph.
                Tentukan apakah METODOLOGI kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_METODOLOGI'.
                Output Anda HARUS berupa JSON yang valid, BUKAN teks percakapan, penjelasan, atau Markdown code block.

                CONTOH OUTPUT DATA MIRIP:
                ```json
                {{
                  "type": "SERUPA_METODOLOGI",
                  "context": "Kedua artikel sama-sama menggunakan metode penelitian kuasi-eksperimen dengan desain pretest-posttest control group."
                }}
                ```
                CONTOH OUTPUT DATA TIDAK MIRIP:
                ```json
                {{
                  "type": null,
                  "context": "Metodologi kedua artikel tidak mirip; satu menggunakan ... dan yang lain menggunakan ..."
                }}
                ```

                Artikel 1 (ID: {article1_id}) Metodologi: {{article1_details.att_method}}
                Artikel 2 (ID: {article2_id}) Metodologi: {{article2_details.att_method}}

                Output JSON harus sesuai skema Pydantic berikut:
                {format_instruction}
                """,
            ),
            "SERUPA_PENELITIAN_LANJUT": PromptTemplate(
                input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
                template="""
                Anda adalah agen pembuat relasi knowledge graph.
                Tentukan apakah REKOMENDASI PENELITIAN LANJUT kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_PENELITIAN_LANJUT'.
                Output Anda HARUS berupa JSON yang valid, BUKAN teks percakapan, penjelasan, atau Markdown code block.

                CONTOH OUTPUT DATA MIRIP:
                ```json
                {{
                  "type": "SERUPA_PENELITIAN_LANJUT",
                  "context": "Kedua artikel menyarankan penelitian lanjutan tentang perluasan sampel ke berbagai setting demografi."
                }}
                ```
                CONTOH OUTPUT DATA TIDAK MIRIP:
                ```json
                {{
                  "type": null,
                  "context": "Rekomendasi penelitian lanjut kedua artikel ini tidak mirip; satu menyarankan ... dan yang lain menyarankan ..."
                }}
                ```

                Artikel 1 (ID: {article1_id}) Penelitian Lanjut: {{article1_details.att_future}}
                Artikel 2 (ID: {article2_id}) Penelitian Lanjut: {{article2_details.att_future}}

                Output JSON harus sesuai skema Pydantic berikut:
                {format_instruction}
                """,
            ),
            "SERUPA_GAP_PENELITIAN": PromptTemplate(
                input_variables=["article1_id", "article1_details", "article2_id", "article2_details", "format_instruction"],
                template="""
                Anda adalah agen pembuat relasi knowledge graph.
                Tentukan apakah GAP PENELITIAN kedua artikel ini cukup mirip dan signifikan untuk membuat relasi 'SERUPA_GAP_PENELITIAN'.
                Output Anda HARUS berupa JSON yang valid, BUKAN teks percakapan, penjelasan, atau Markdown code block.

                CONTOH OUTPUT DATA MIRIP:
                ```json
                {{
                  "type": "SERUPA_GAP_PENELITIAN",
                  "context": "Kedua artikel mengidentifikasi gap penelitian yang serupa terkait keterbatasan data set dalam deteksi hoaks berbahasa Indonesia."
                }}
                ```
                CONTOH OUTPUT DATA TIDAK MIRIP:
                ```json
                {{
                  "type": null,
                  "context": "Gap penelitian kedua artikel tidak mirip; satu mengidentifikasi ... dan yang lain mengidentifikasi ..."
                }}
                ```

                Artikel 1 (ID: {article1_id}) Gap Penelitian: {{article1_details.att_gaps}}
                Artikel 2 (ID: {article2_id}) Gap Penelitian: {{article2_details.att_gaps}}

                Output JSON harus sesuai skema Pydantic berikut:
                {format_instruction}
                """,
            ),
        }
        # Buat LLMChain untuk setiap prompt relasi
        self.relation_chains = {
            rel_type: prompt | llm_kg 
            for rel_type, prompt in self.relation_prompts_map.items()
        }
        
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
        print("Memastikan tabel 'public.Node' dan 'public.Edge' ada di Supabase...")
        # CATATAN PENTING: Anda harus membuat/menyesuaikan tabel ini di Supabase secara manual
        # CREATE EXTENSION IF NOT EXISTS vector;
        # CREATE TABLE public."User" (...); -- Jika belum ada
        # CREATE TABLE public."BrainstormingSession" (...); -- Jika belum ada
        # CREATE TABLE public."Article" (...); -- Jika belum ada

        # Tabel public."Node" (sesuai skema JS dengan penambahan embedding)
        # CREATE TABLE public."Node" (
        #     id TEXT NOT NULL,
        #     label TEXT NOT NULL,
        #     title TEXT NULL,
        #     att_goal TEXT NULL,
        #     att_method TEXT NULL,
        #     att_background TEXT NULL,
        #     att_future TEXT NULL,
        #     att_gaps TEXT NULL,
        #     att_url TEXT NULL,
        #     type TEXT NOT NULL,
        #     content TEXT NOT NULL, -- ini adalah summary
        #     "articleId" TEXT NOT NULL,
        #     embedding VECTOR(768), -- <--- KOLOM EMBEDDING UNTUK PYTHON
        #     CONSTRAINT "Node_pkey" PRIMARY KEY (id),
        #     CONSTRAINT "Node_articleId_fkey" FOREIGN KEY ("articleId") REFERENCES public."Article" (id) ON UPDATE CASCADE ON DELETE CASCADE
        # );
        # CREATE INDEX ON public."Node" USING hnsw (embedding vector_l2_ops);

        # Tabel public."Edge" (sesuai skema JS)
        # CREATE TABLE public."Edge" (
        #     id TEXT NOT NULL, -- ID unik untuk Edge
        #     "fromId" TEXT NOT NULL,
        #     "toId" TEXT NOT NULL,
        #     relation TEXT NULL,
        #     label TEXT NULL,
        #     color TEXT NULL,
        #     "articleId" TEXT NOT NULL,
        #     CONSTRAINT "Edge_pkey" PRIMARY KEY (id),
        #     CONSTRAINT "Edge_articleId_fkey" FOREIGN KEY ("articleId") REFERENCES public."Article" (id) ON UPDATE CASCADE ON DELETE CASCADE,
        #     CONSTRAINT "Edge_fromId_fkey" FOREIGN KEY ("fromId") REFERENCES public."Node" (id) ON UPDATE CASCADE ON DELETE CASCADE,
        #     CONSTRAINT "Edge_toId_fkey" FOREIGN KEY ("toId") REFERENCES public."Node" (id) ON UPDATE CASCADE ON DELETE CASCADE
        # );
        # CREATE UNIQUE INDEX IF NOT EXISTS "Edge_fromId_toId_relation_key" ON public."Edge" USING btree ("fromId", "toId", relation);

        # Tabel public.document_chunks (sesuai skema JS)
        # CREATE TABLE public.document_chunks (
        #     id TEXT PRIMARY KEY,
        #     doc_id TEXT NOT NULL, -- FK ke public."Article".id
        #     chunk_text TEXT,
        #     page_number INTEGER,
        #     embedding VECTOR(768),
        #     "createdAt" TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
        #     CONSTRAINT document_chunks_doc_id_fkey FOREIGN KEY (doc_id) REFERENCES public."Article" (id) ON UPDATE CASCADE ON DELETE CASCADE
        # );
        # CREATE INDEX ON public.document_chunks USING hnsw (embedding vector_l2_ops);

        print("Pastikan SEMUA tabel JS (public.User, public.BrainstormingSession, public.Article, public.Node, public.Edge, public.document_chunks) telah dibuat dengan benar, dan foreign keys, serta kolom embedding/indeks telah ditambahkan.")


    async def _get_existing_nodes(self) -> List[Dict[str, Any]]:
        """
        Mengambil semua nodes ARTIKEL yang sudah ada dari tabel public."Node" di Supabase.
        Ambil properti yang dibutuhkan LLM untuk identifikasi relasi.
        """
        try:
            print(f"[kg_processor] Mulai mengambil existing nodes dari tabel public.'Node'...")
            # Ambil semua kolom yang relevan dari tabel "Node"
            response = self.supabase_client.from_('Node').select('id, label, title, att_goal, att_method, att_background, att_future, att_gaps, att_url, type, content, "articleId"').eq('type', 'article').execute()
            print(f"[kg_processor] Respon mentah dari Supabase: {response}")
            if response.data:
                print(f"[kg_processor] Mengambil {len(response.data)} existing nodes dari Supabase: {response.data}")
                return response.data
            else:
                print(f"[kg_processor] Tidak ada data dalam response.data. Mencoba fallback dengan koneksi langsung")
                conn = await self._get_pg_connection()
                # Sesuaikan query ini dengan kolom-kolom tabel public."Node"
                nodes_raw = await conn.fetch('SELECT id, label, title, att_goal, att_method, att_background, att_future, att_gaps, att_url, type, content, "articleId" FROM "Node" WHERE type = \'article\'')
                await conn.close()
                if nodes_raw:
                    # Konversi asyncpg.Record ke dict dan pastikan camelCase fields dikembalikan dengan benar
                    # dict(node) akan mengembalikan nama kolom asli (camelCase)
                    return [dict(node) for node in nodes_raw]
                return []
        except Exception as e:
            print(f"[kg_processor] Error fetching existing nodes: {e}")
            return []
            
    async def process_document(self, full_document_content: str, article_id: str, document_title: str, att_url: Optional[str] = None) -> bool: # <-- Terima att_url
        print(f"Memulai ekstraksi Knowledge Graph untuk dokumen '{document_title}' (Article ID: {article_id}) dengan RAG-driven approach...")

        conn_retrieval = None # Koneksi untuk retrieval chunks
        retrieved_context = ""
        try:
            conn_retrieval = await self._get_pg_connection()

            query_text = f"Judul: {document_title}. Ringkasan awal: {full_document_content[:500]}..."
            query_embedding = await embeddings_for_retrieval.aembed_query(query_text)
            
            # Retrieve chunks dari tabel public.document_chunks
            retrieval_results = await conn_retrieval.fetch("""
                SELECT chunk_text, embedding <-> $1 AS distance
                FROM public.document_chunks
                WHERE doc_id = $2
                ORDER BY distance
                LIMIT 10;
            """, query_embedding, article_id) # Gunakan article_id untuk doc_id

            if retrieval_results:
                retrieved_context = "\n\n".join([r['chunk_text'] for r in retrieval_results])
                print(f"[kg_processor] Berhasil mengambil {len(retrieval_results)} chunks untuk konteks. Isi retrieved_context: {retrieved_context[:500]}...")
            else:
                retrieved_context = full_document_content[:2000]
                print("[kg_processor] Tidak ada chunks yang diambil, menggunakan bagian awal dokumen penuh sebagai fallback. Isi fallback: {full_document_content[:500]}...")

        except Exception as e:
            print(f"[kg_processor] Error saat melakukan retrieval dari document_chunks: {e}")
            retrieved_context = full_document_content[:2000] 
            print("[kg_processor] Retrieval gagal, menggunakan bagian awal dokumen penuh sebagai fallback.")
        finally:
            if conn_retrieval: 
                await conn_retrieval.close()

        # Ambil nodes artikel yang sudah ada untuk konteks LLM dan perbandingan
        existing_article_nodes = await self._get_existing_nodes()
        existing_nodes_json_for_prompt = json.dumps(existing_article_nodes, indent=2) 
        
        format_instruction_main_node = json.dumps(KnowledgeGraphOutput.model_json_schema(), indent=2)

        try:
            print("[kg_processor] Mulai ekstraksi NODE ARTIKEL UTAMA dengan LLM...")
            input_data_main_node = {
                "retrieved_context": retrieved_context,
                "document_id": article_id, # Kirim article_id sebagai document_id
                "document_title": document_title,
                "existing_nodes_json": existing_nodes_json_for_prompt,
                "format_instruction": format_instruction_main_node
            }
            raw_kg_output_message = await self.kg_extraction_chain.ainvoke(input_data_main_node)
            
            raw_kg_output_str = raw_kg_output_message.content
            print("[kg_processor] Output LLM (Node Ekstraksi):", raw_kg_output_str)
            
            json_string_to_parse = raw_kg_output_str.strip().lstrip('```json').rstrip('```').strip()
            
            kg_output_dict = json.loads(json_string_to_parse)
            
            kg_output = KnowledgeGraphOutput.model_validate(kg_output_dict)
            print("[kg_processor] Output KG (Node) tervalidasi.")
            
            print("Ekstraksi Node Artikel Utama oleh LangChain berhasil.")
        except ValidationError as e:
            print(f"[kg_processor] Pydantic Validation Error (Node): {e.errors()}")
            print(f"Raw LLM output (for validation error): {raw_kg_output_str}")
            return False
        except json.JSONDecodeError as e:
            print(f"[kg_processor] Error decoding JSON from LLM output (Node): {e}")
            print(f"Raw LLM output: {raw_kg_output_str}")
            return False
        except Exception as e:
            print(f"Error saat ekstraksi Node Artikel Utama dengan LangChain: {e}")
            return False

        conn_save = None 
        try:
            conn_save = await self._get_pg_connection()

            # --- LOGIKA PENYIMPANAN NODE UTAMA ARTIKEL (Sesuai Skema JS "Node") ---
            main_article_node_data = kg_output.article_node

            # Generate embedding untuk node
            article_embedding_content = main_article_node_data.content # Gunakan 'content' (summary) untuk embedding
            node_embedding = await embeddings_kg_model.aembed_query(article_embedding_content)
            
            # Perhatikan nama kolom CamelCase di sini
            await conn_save.execute("""
                INSERT INTO public."Node" (id, label, title, att_goal, att_method, att_background, att_future, att_gaps, att_url, type, content, "articleId", embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (id) DO NOTHING;
            """, 
                main_article_node_data.id, 
                main_article_node_data.label, # 'Artikel'
                main_article_node_data.title, 
                main_article_node_data.att_goal, 
                main_article_node_data.att_method, 
                main_article_node_data.att_background, 
                main_article_node_data.att_future, 
                main_article_node_data.att_gaps, 
                att_url, # Gunakan att_url dari argumen fungsi
                main_article_node_data.type, # 'article'
                main_article_node_data.content, # Ringkasan
                article_id, # articleId dari public."Article"
                node_embedding
            )
            print(f"Node Artikel Utama '{main_article_node_data.id}' diproses dan disimpan ke public.Node.")

            # --- LOGIKA PEMBUATAN EDGES SEPENUHNYA OLEH LLM ---
            print(f"Mencari dan membuat relasi untuk '{main_article_node_data.id}' dengan prompt terpisah (sepenuhnya oleh LLM)...")

            other_existing_articles = [
                node for node in existing_article_nodes if node['id'] != main_article_node_data.id
            ]

            if not other_existing_articles:
                print("Tidak ada artikel lain di database untuk dibandingkan relasi.")
                return True # Proses berhasil tanpa edges

            final_relations_to_insert = []

            # Format untuk skema output relasi
            format_instruction_rel_verify = json.dumps({
                "type": {"type": "string", "enum": list(self.relation_prompts_map.keys()) + ["null"]},
                "context": {"type": "string"}
            }, indent=2)

            for existing_art_node in other_existing_articles:
                existing_art_id = existing_art_node['id']
                existing_art_title = existing_art_node['title']
                existing_art_details = existing_art_node['extracted_details']
                
                # Cek apakah existing_art_details masih berupa string dan parse jika perlu
                if isinstance(existing_art_details, str):
                    try:
                        existing_art_details = json.loads(existing_art_details)
                    except json.JSONDecodeError as e:
                        print(f"  [kg_processor] WARNING: Gagal mengurai JSON string 'extracted_details' dari node {existing_art_id} (Data Lama?): {e}. Menggunakan dict kosong.")
                        existing_art_details = {} 

                # Iterasi setiap aspek perbandingan
                aspects_to_check = {
                    "background": "SERUPA_LATAR_BELAKANG",
                    "methodology": "SERUPA_METODOLOGI",
                    "purpose": "SERUPA_TUJUAN",
                    "future_research": "SERUPA_PENELITIAN_LANJUT",
                    "research_gap": "SERUPA_GAP_PENELITIAN",
                }
                
                for aspect_key, relation_type in aspects_to_check.items():
                    # Akses detail dari Pydantic object
                    current_article_details_dict = main_article_node_data.extracted_details.dict()
                    
                    # Cek apakah konten aspek tidak None/kosong di kedua artikel
                    # Ambil dari current_article_details_dict dan existing_art_details
                    current_aspect_content = current_article_details_dict.get(aspect_key)
                    existing_aspect_content = existing_art_details.get(aspect_key)

                    if not current_aspect_content or current_aspect_content == "Tidak ditemukan" or not existing_aspect_content or existing_aspect_content == "Tidak ditemukan":
                        print(f"  [kg_processor] Melewatkan perbandingan '{aspect_key}' karena konten kosong/tidak ditemukan di salah satu artikel.")
                        continue

                    try:
                        # Siapkan input untuk prompt verifikasi relasi
                        verification_input = {
                            "article1_id": main_article_node_data.id,
                            # Kirim detail langsung dari Pydantic model atau dict yang sudah ada
                            "article1_details": json.dumps(current_article_details_dict), 
                            "article2_id": existing_art_id,
                            "article2_details": json.dumps(existing_art_details),
                            "potential_relation_type": relation_type,
                            "context_hint": f"Periksa apakah ada kemiripan signifikan pada bagian '{aspect_key}'.",
                            "format_instruction": format_instruction_rel_verify
                        }
                        
                        relation_chain_to_use = self.relation_chains[relation_type]

                        print(f"  [kg_processor] Memeriksa relasi '{relation_type}' antara '{main_article_node_data.id}' dan '{existing_art_id}' berdasarkan '{aspect_key}'...")
                        
                        verification_output = await relation_chain_to_use.ainvoke(verification_input)
                        verified_rel_str = verification_output.content.strip().lstrip('```json').rstrip('```').strip()
                        
                        print(f"  [kg_processor] Output LLM untuk verifikasi {relation_type}: {verified_rel_str}")

                        if not verified_rel_str or not verified_rel_str.strip():
                            print(f"  [kg_processor] Output LLM kosong untuk {relation_type}, melewati.")
                            continue
                        
                        verified_rel_dict = json.loads(verified_rel_str)
                        
                        if verified_rel_dict.get("type") == relation_type:
                            final_relations_to_insert.append({
                                "id": str(uuid.uuid4()), # Generate UUID for edge ID
                                "fromId": main_article_node_data.id,
                                "toId": existing_art_id,
                                "relation": relation_type,
                                "label": relation_type.replace('_', ' ').title(), # Label visual
                                "color": get_relation_color(relation_type), # Warna visual
                                "articleId": article_id, # Article ID dari current processing
                                "context": verified_rel_dict.get("context", f"Ditemukan kemiripan pada {aspect_key} antara '{main_article_node_data.title}' dan '{existing_art_title}'.")
                            })
                            print(f"  [kg_processor] Dikonfirmasi: Relasi {relation_type} antara '{main_article_node_data.id}' dan '{existing_art_id}'.")
                        else:
                            print(f"  [kg_processor] LLM menolak atau tidak mengkonfirmasi relasi {relation_type} antara '{main_article_node_data.id}' dan '{existing_art_id}'. Output LLM type: {verified_rel_dict.get('type', 'N/A')}")

                    except (json.JSONDecodeError, ValidationError) as llm_parse_e:
                        print(f"  [kg_processor] Error parsing LLM output untuk {relation_type}: {llm_parse_e}. Raw Output: {verified_rel_str}")
                    except Exception as llm_e:
                        print(f"  [kg_processor] Error memanggil LLM untuk verifikasi {relation_type}: {llm_e}")
                        print(f"  [kg_processor] Gagal dengan input: article1_id={main_article_node_data.id}, article2_id={existing_art_id}, aspect={aspect_key}. Error: {llm_e}")


            # Insert semua relasi yang sudah diverifikasi ke database
            if final_relations_to_insert:
                for relation_data in final_relations_to_insert:
                    try:
                        # Pastikan source_id dan target_id ada sebelum insert (ulang, untuk safety)
                        # Nama kolom di DB adalah "fromId" dan "toId"
                        source_exists_final = await conn_save.fetchval("""SELECT id FROM public."Node" WHERE id = $1""", relation_data['fromId'])
                        target_exists_final = await conn_save.fetchval("""SELECT id FROM public."Node" WHERE id = $1""", relation_data['toId'])

                        if source_exists_final and target_exists_final:
                            await conn_save.execute("""
                                INSERT INTO public."Edge" (id, "fromId", "toId", relation, label, color, "articleId", context)
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                                ON CONFLICT (id) DO NOTHING;
                            """, 
                                relation_data['id'],
                                relation_data['fromId'], 
                                relation_data['toId'], 
                                relation_data['type'], # 'type' dari Pydantic, 'relation' di DB
                                relation_data['label'], 
                                relation_data['color'], 
                                relation_data['articleId'],
                                relation_data['context']
                            )
                            print(f"Edge '{relation_data['fromId']}' -[{relation_data['type']}]-> '{relation_data['toId']}' diproses dan disimpan ke public.Edge.")
                        else:
                            print(f"Melewatkan edge final '{relation_data['fromId']}' -[{relation_data['type']}]-> '{relation_data['toId']}' karena node tidak ditemukan di public.Node (setelah verifikasi).")
                    except Exception as e:
                        print(f"Error saat menyimpan edge '{relation_data['fromId']}' -[{relation_data['type']}]-> '{relation_data['toId']}': {e}")
                print(f"Total {len(final_relations_to_insert)} edges berhasil disimpan untuk '{main_article_node_data.id}'.")
            else:
                print(f"Tidak ada edges yang ditemukan atau diverifikasi untuk '{main_article_node_data.id}'.")
            
            return True

        except Exception as e:
            print(f"Error menyimpan KG ke Supabase: {e}")
            return False
        finally:
            if conn_save:
                await conn_save.close()


    async def get_graph_data(self, session_id: Optional[str] = None) -> Dict[str, Any]: # Menerima session_id
        nodes_data = []
        edges_data = []
        try:
            # Mengambil dari tabel public."Node"
            # Perhatikan nama kolom CamelCase saat memilih dari DB
            query_nodes = self.supabase_client.from_('Node').select('id, label, title, att_goal, att_method, att_background, att_future, att_gaps, att_url, type, content, "articleId", embedding')
            # Jika session_id diberikan, tambahkan filter
            if session_id:
                # Ini mengasumsikan Anda akan menambahkan kolom "sessionId" ke tabel "Node"
                # agar nodes dapat difilter per sesi/proyek.
                # Contoh: query_nodes = query_nodes.eq('sessionId', session_id)
                print(f"Warning: Filtering by session_id is not yet implemented for Node table. Please add 'sessionId' column to public.Node if needed.")
            
            nodes_response = query_nodes.execute()
            if nodes_response.data:
                nodes_data = nodes_response.data
            
            # Mengambil dari tabel public."Edge"
            # Perhatikan nama kolom CamelCase saat memilih dari DB
            query_edges = self.supabase_client.from_('Edge').select('id, "fromId", "toId", relation, label, color, "articleId", context')
            # Jika session_id diberikan, tambahkan filter
            if session_id:
                # Ini mengasumsikan Anda akan menambahkan kolom "sessionId" ke tabel "Edge"
                # Contoh: query_edges = query_edges.eq('sessionId', session_id)
                print(f"Warning: Filtering by session_id is not yet implemented for Edge table. Please add 'sessionId' column to public.Edge if needed.")

            edges_response = query_edges.execute()
            if edges_response.data:
                edges_data = edges_response.data

            print(f"Berhasil mengambil {len(nodes_data)} nodes dan {len(edges_data)} edges dari Supabase.")
            return {"nodes": nodes_data, "edges": edges_data}
        except Exception as e:
            print(f"Error fetching graph data from Supabase: {e}")
            return {"nodes": [], "edges": []}
