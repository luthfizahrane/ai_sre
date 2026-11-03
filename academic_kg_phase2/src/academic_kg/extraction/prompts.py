"""Centralized and versionable prompts; preserve Phase 1 domain instructions."""

ARTICLE_TEMPLATE = """
Anda adalah agen ekstraksi informasi ahli yang membangun knowledge graph untuk penelitian akademik.

TUGAS UTAMA:
Ekstrak informasi terstruktur untuk node artikel BARU berdasarkan KONTEKS yang DIBERIKAN.
Output Anda HARUS berupa data JSON yang valid sesuai skema yang diberikan, BUKAN definisi skemanya.
Pastikan untuk mengekstrak secara eksplisit:
- Latar Belakang Penelitian (cari frasa seperti 'latar belakang', 'pendahuluan', atau konteks awal; jika tidak ditemukan, lakukan inferensi dari bagian pengantar atau konteks awal lainnya sebelum menyimpulkan 'Tidak ditemukan')
- Metodologi Penelitian (cari frasa seperti 'metode', 'pendekatan', 'prosedur'; jika tidak ditemukan, lakukan inferensi dari deskripsi proses atau langkah-langkah penelitian sebelum menyimpulkan 'Tidak ditemukan')
- Tujuan Penelitian (cari frasa seperti 'tujuan', 'sasaran', 'maksud'; jika tidak ditemukan, lakukan inferensi dari implikasi latar belakang atau hasil yang diharapkan sebelum menyimpulkan 'Tidak ditemukan')
- Penelitian Lanjutan (cari frasa seperti 'penelitian lanjutan', 'rekomendasi', 'studi lebih lanjut'; jika tidak ditemukan, lakukan inferensi dari kesimpulan, pembahasan, atau saran implisit terkait ruang lingkup penelitian sebelum menyimpulkan 'Tidak ditemukan')
- Gap Penelitian (cari frasa seperti 'keterbatasan', 'kesenjangan', 'tantangan'; jika tidak ditemukan, lakukan inferensi dari kesimpulan, pembahasan, atau kelemahan terkait ruang lingkup penelitian sebelum menyimpulkan 'Tidak ditemukan')

Jika informasi tidak ditemukan setelah upaya inferensi, isi dengan 'Tidak ditemukan' hanya jika tidak ada petunjuk sama sekali.

KONTEKS ARTIKEL BARU:
Judul: {document_title}
ID: {document_id}
Konten: {retrieved_context}

ARTIKEL YANG SUDAH ADA (untuk referensi, bukan untuk membuat relasi otomatis di sini):
{existing_nodes_json}

INSTRUKSI EKSTRAKSI NODE ARTIKEL BARU:
Untuk 'article_node' yang sedang diproses, ekstrak:
- ID: 'doc_{document_id}'
- Label: 'Artikel' (tetap)
- Judul: {document_title}
- Att_goal: Tujuan penelitian (jika tidak ada, cari implikasi dari latar belakang atau metode)
- Att_method: Metodologi penelitian (jika tidak ada, cari deskripsi proses atau eksperimen)
- Att_background: Latar belakang penelitian (jika tidak ada, gunakan bagian pendahuluan)
- Att_future: Saran untuk penelitian lanjutan (jika tidak ada, cari saran implisit)
- Att_gaps: Kesenjangan atau keterbatasan yang diidentifikasi (jika tidak ada, cari kelemahan metode atau data)
- Type: 'article' (tetap)
- Content: Ringkasan komprehensif maksimal 500 kata dari seluruh dokumen.
Jika suatu detail tidak ditemukan meskipun ada petunjuk konteks, lakukan inferensi logis berdasarkan teks yang tersedia.

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
    "type": "article",
    "content": "Ini adalah ringkasan contoh dari artikel tersebut."

  }}
}}
```

Output harus JSON murni sesuai skema:
{format_instruction}
"""

RETRY_TEMPLATE = (
    "Lengkapi hanya field yang belum ada: {missing_fields}. "
    "Jangan mengarang informasi. Jika tidak didukung teks, tulis "
    "'Tidak ditemukan'. Kembalikan JSON object dengan key nama field. "
    "Konteks: {context}"
)

CHAT_TEMPLATE = (
    "Anda adalah asisten AI yang membantu. Jawab pertanyaan berikut: {question}"
)

RELATION_ASPECTS = {
    "SERUPA_LATAR_BELAKANG": ("background", "LATAR BELAKANG"),
    "SERUPA_TUJUAN": ("purpose", "TUJUAN PENELITIAN"),
    "SERUPA_METODOLOGI": ("methodology", "METODOLOGI"),
    "SERUPA_PENELITIAN_LANJUT": ("future_research", "REKOMENDASI PENELITIAN LANJUT"),
    "SERUPA_GAP_PENELITIAN": ("research_gap", "GAP PENELITIAN"),
}


def build_relation_prompt(relation_type: str, description: str) -> str:
    return (
        "Anda adalah agen pembuat relasi knowledge graph akademik.\n"
        f"Tentukan apakah {description} dua artikel cukup mirip dan signifikan "
        f"untuk membentuk relasi '{relation_type}'.\n"
        "Jika mirip, kembalikan JSON object dengan 'type' sesuai jenis "
        "relasi dan 'context' yang menjelaskan kesamaan.\n"
        "Jika tidak mirip, kembalikan 'type': null (JSON null) dan "
        "'context' yang menjelaskan perbedaannya.\n"
        "Hanya gunakan konteks yang diberikan. Output JSON murni tanpa Markdown.\n\n"
        "Artikel 1 (ID: {article1_id}): {article1_excerpt}\n"
        "Artikel 2 (ID: {article2_id}): {article2_excerpt}\n"
        "Output harus sesuai skema berikut:\n{format_instruction}"
    )
