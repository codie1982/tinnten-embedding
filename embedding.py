import os,re, json, threading, uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from crawl4ai import AsyncWebCrawler, CacheMode, BrowserConfig, CrawlerRunConfig
import asyncio
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, JsonXPathExtractionStrategy
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai import CrawlerMonitor, DisplayMode, RateLimiter
import nest_asyncio
nest_asyncio.apply()
# Create the Flask app
from dotenv import load_dotenv
import numpy as np
import faiss

INDEX_PATH = "faiss.index"
META_PATH = "meta.json"

URL_REGEX = re.compile(
    r'^(?:http|https)://'  # http:// veya https://
    r'(?:\S+)'             # en az bir karakter
)
# .env dosyasını yükle
load_dotenv()

# Flask uygulamasını oluştur
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Tüm istemcilerden gelen istekleri kabul et

# Ortam değişkeninden model adı oku (bulamazsa default)
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# SentenceTransformer modelini yükle
model = SentenceTransformer(MODEL_NAME)


_lock = threading.Lock()
index = None              # faiss.IndexIDMap2
_meta = {}                # {int_id: {"external_id": str, "text": str|None, "metadata": dict}}
_next_int_id = 1

def _normalize(a: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(a)
    return a

def _save_state():
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in _meta.items()}, f, ensure_ascii=False)
    if index is not None:
        faiss.write_index(index, INDEX_PATH)

def _load_state():
    global index, _meta, _next_int_id
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = {int(k): v for k, v in json.load(f).items()}
    else:
        _meta = {}

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = None

    _next_int_id = (max(_meta.keys()) + 1) if _meta else 1

def _ensure_index(dim: int):
    global index, _meta, _next_int_id
    if index is None:
        base = faiss.IndexFlatIP(dim)    # cosine için IP + normalize
        index = faiss.IndexIDMap2(base)
    elif index.d != dim:
        # Model boyutu değişmiş → temiz kurulum
        base = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(base)
        _meta.clear()
        _next_int_id = 1
        _save_state()



@app.route('/', methods=['GET'])
def index():
    """Root endpoint to confirm the API is running."""
    # Return a simple JSON response to indicate that the server is running
    return jsonify({"message": "Hello, world! The API is running."})


@app.route('/api/v10/llm/vector', methods=['POST'])
def generate_vector(): 
    """Endpoint to convert provided text into a vector."""
    try:
        # Extract JSON data from the incoming request
        data = request.get_json()

        # Check if the data is present and contains the 'text' key
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided in the request body."}), 400
    
        # Retrieve the text to be vectorized
        text = data['text']

        # Validate that the provided text is not empty after stripping whitespace
        if not text.strip():
            return jsonify({"error": "The provided text is empty."}), 400

        # Encode the text using the SentenceTransformer model (CPU) and convert it to a list
        vector = model.encode(text).tolist()

        # Return the generated vector as a JSON response
        return jsonify({"vector": vector})

    except Exception as e:
        # Handle any unexpected errors and return a descriptive message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/v10/llm/scrapper', methods=['POST'])
def run_scraper():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()

        print("➡️ Scraper isteği alındı. URL:", url)
        if not url or not re.match(URL_REGEX, url):
            return jsonify({"error": "Invalid URL format provided."}), 400

        if not url:
            return jsonify({"error": "No valid URL provided."}), 400

        result = asyncio.run(extract_content_from_url(url))

        if result.get("success") is False:
            return jsonify({
                "error": result.get("message", "Unknown error"),
                "status_code": result.get("status_code", 500)
            }), result.get("status_code", 500)

        return jsonify({"crawler": result}), 200

    except Exception as e:
        print("❌ SCRAPER HATASI:", str(e))
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/api/v10/ingest/markdown", methods=["POST"])
def ingest_markdown():
    """
    Girdi:
    {
      "url": "https://site.com/hizmetlerimiz/",
      "raw_markdown": "...",
      "options": { "target_chars": 1100, "overlap_chars": 180, "doc_type": "service" }
    }
    Çıktı: page + chunks (chunk_id + faiss_id ile)
    """
    try:
        data = request.get_json() or {}
        url = (data.get("url") or "").strip()
        raw_md = data.get("raw_markdown") or ""
        opts = data.get("options") or {}
        if not url or not re.match(URL_REGEX, url):
            return jsonify({"success": False, "error": "invalid url"}), 400
        if not raw_md.strip():
            return jsonify({"success": False, "error": "empty markdown"}), 400

        doc_type = (opts.get("doc_type") or "service").lower()
        target = int(opts.get("target_chars") or 1100)
        overlap = int(opts.get("overlap_chars") or 180)

        # 1) clean
        clean_md = clean_markdown(raw_md)
        content_hash = sha1_of(clean_md)
        title = None
        m = re.search(r"(?m)^#\s+(.+)$", clean_md)
        if m: title = m.group(1).strip()

        # 2) chunk
        raw_chunks = chunk_text(clean_md, target=target, overlap=overlap)

        # 3) embed + FAISS add_with_ids
        results = []
        with _lock:
            # embed boyutu öğrenmek için tek örnek encode edelim (boşsa kısa devre)
            if not raw_chunks:
                page = {
                    "url": url, "title": title, "language": "tr",
                    "content_hash": content_hash
                }
                return jsonify({"success": True, "page": page, "chunks": [], "model": {
                    "name": MODEL_NAME
                }, "index": {"name": "main_flat_ip"}})

            # encode toplu:
            texts = [c[0] for c in raw_chunks]
            vecs = model.encode(texts)
            vecs = np.array(vecs, dtype=np.float32)
            if vecs.ndim == 1: vecs = vecs.reshape(1, -1)
            dim = vecs.shape[1]
            _ensure_index(dim)
            _normalize(vecs)

            # FAISS id’leri ardışık verelim
            global _next_int_id
            start_id = _next_int_id
            ids = np.arange(start_id, start_id + vecs.shape[0], dtype=np.int64)
            _next_int_id += vecs.shape[0]

            index.add_with_ids(vecs, ids)

            # _meta sadece basic tutuyor (opsiyonel)
            for i, (txt, s, e, h_path) in enumerate(raw_chunks):
                faiss_id = int(ids[i])
                chunk_id = str(uuid.uuid4())
                _meta[faiss_id] = {
                    "external_id": chunk_id,
                    "text": txt,
                    "metadata": {"doc_type": doc_type, "url": url, "h_path": h_path}
                }
                results.append({
                    "chunk_id": chunk_id,
                    "faiss_id": faiss_id,
                    "url": url,
                    "h_path": h_path,
                    "text": txt,
                    "char_start": s,
                    "char_end": e,
                    "metadata": {
                        "doc_type": doc_type
                    }
                })
            _save_state()

        page = {
            "url": url,
            "title": title,
            "language": "tr",
            "content_hash": content_hash
        }
        return jsonify({
            "success": True,
            "page": page,
            "chunks": results,
            "model": { "name": MODEL_NAME, "dim": int(vecs.shape[1]), "emb_ver": time.strftime("%Y-%m-%d") },
            "index": { "name": "main_flat_ip" }
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"ingest failed: {str(e)}"}), 500
# -------------------------------
# 1) UPSERT: metni/vektörü FAISS'e kaydet
# -------------------------------
@app.route("/api/v10/vector/upsert", methods=["POST"])
def upsert_vector():
    """
    Body örnekleri:
    { "text": "ev boyama ustası", "external_id": "svc-001", "metadata": {"category":"services"} }
    { "vector": [...], "external_id": "svc-raw", "metadata": {...} }
    Opsiyonel: { "id": 42 }  # aynı kaydı güncellemek için
    """
    try:
        p = request.get_json() or {}
        text = p.get("text")
        vec = p.get("vector")
        external_id = p.get("external_id") or str(uuid.uuid4())
        extra_meta = p.get("metadata") or {}
        int_id = p.get("id")

        if vec is None:
            if not text or not text.strip():
                return jsonify({"error": "Provide either 'text' or 'vector'."}), 400
            v = model.encode(text)
        else:
            v = np.array(vec, dtype=np.float32)

        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float32)
        if v.ndim == 1:
            v = v.reshape(1, -1).astype(np.float32)

        dim = v.shape[1]

        with _lock:
            _ensure_index(dim)
            _normalize(v)

            global _next_int_id
            if int_id is None:
                int_id = _next_int_id
                _next_int_id += 1
            else:
                try:
                    index.remove_ids(np.array([int(int_id)], dtype=np.int64))
                except Exception:
                    pass

            index.add_with_ids(v, np.array([int(int_id)], dtype=np.int64))
            _meta[int(int_id)] = {
                "external_id": external_id,
                "text": text if text else _meta.get(int(int_id), {}).get("text"),
                "metadata": extra_meta
            }
            _save_state()

        return jsonify({"id": int(int_id), "external_id": external_id})
    except Exception as e:
        return jsonify({"error": f"upsert failed: {str(e)}"}), 500

# -------------------------------
# 2) SEARCH: metinle ya da vektörle ara/oku
# -------------------------------
@app.route("/api/v10/vector/search", methods=["POST"])
def search_vector():
    """
    Body örnekleri:
    { "text": "iç cephe boyama", "k": 5 }
    { "vector": [...], "k": 10 }

    Opsiyonel basit filtre:
    { "filter": {"metadata.category": "services"} }
    """
    try:
        p = request.get_json() or {}
        text = p.get("text")
        vec = p.get("vector")
        k = int(p.get("k") or 5)
        filt = p.get("filter") or {}

        if vec is None:
            if not text or not text.strip():
                return jsonify({"error": "Provide either 'text' or 'vector'."}), 400
            q = model.encode(text)
        else:
            q = np.array(vec, dtype=np.float32)

        if not isinstance(q, np.ndarray):
            q = np.array(q, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1).astype(np.float32)

        with _lock:
            if index is None or index.ntotal == 0:
                return jsonify({"results": []})

            if index.d != q.shape[1]:
                return jsonify({"error": f"Query dim {q.shape[1]} != index dim {index.d}"}), 400

            _normalize(q)
            scores, ids = index.search(q, k)
            ids = ids[0].tolist()
            scores = scores[0].tolist()

            out = []
            for i, idx in enumerate(ids):
                if idx == -1:
                    continue
                m = _meta.get(int(idx))
                if not m:
                    continue

                # çok basit exact-match metadata filtresi
                if filt:
                    ok = True
                    for key, val in filt.items():
                        if key.startswith("metadata."):
                            sub = key.split(".", 1)[1]
                            if m.get("metadata", {}).get(sub) != val:
                                ok = False; break
                        else:
                            if m.get(key) != val:
                                ok = False; break
                    if not ok:
                        continue

                out.append({
                    "id": int(idx),
                    "score": float(scores[i]),   # cosine (IP sonrası)
                    "external_id": m.get("external_id"),
                    "text": m.get("text"),
                    "metadata": m.get("metadata", {})
                })

        return jsonify({"results": out})
    except Exception as e:
        return jsonify({"error": f"search failed: {str(e)}"}), 500

  # Uygulama ayağa kalkınca bir kez yükle


async def extract_content_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    browser_cfg = BrowserConfig(
        browser_type="chromium",
        headless=True,
        text_mode=True,
        headers=headers
    )

    config = CrawlerRunConfig(
        verbose=True,
        cache_mode=CacheMode.BYPASS,
        check_robots_txt=True
    )

    async with AsyncWebCrawler(config=browser_cfg, verbose=True) as crawler:
        result = await crawler.arun(url=url, config=config)
        if not result.success:
            print(f"❌ CRAWL FAILED for {result.url}: {result.status_code} - {result.error_message}")
            return {
                "success": False,
                "url": result.url,
                "status_code": result.status_code,
                "message": result.error_message or "Crawling failed"
            }
        
        
        product_info = {
            "success": True,
            "url": result.url,
            "data": result.extracted_content,
            "images": result.media.get("images", []),
            "metadata": result.metadata,
            "markdown": result.markdown.raw_markdown if result.markdown else "",
            "clean_markdown": clean_markdown(result.markdown.raw_markdown) if result.markdown else "",
            "markdown": result.markdown if result.markdown else "",
            "internal_links" : result.links.get("internal", []),
            "external_links" : result.links.get("external", [])
        }
        return product_info


def chunk_text(text: str, target=1100, overlap=180):
    """
    Basit başlık+paragraf odaklı chunking.
    Dönen her item: (chunk_text, start, end, h_path)
    """
    # Başlıkları ayır
    parts = re.split(r"(?m)^(#{1,6}\s.+)$", text)
    blocks = []
    for i in range(0, len(parts), 2):
        pre = parts[i]
        hdr = parts[i+1] if i+1 < len(parts) else None
        body = parts[i+2] if i+2 < len(parts) else ""
        if hdr:
            blocks.append((hdr.strip(), body))
        elif pre.strip():
            blocks.append(("# Loose", pre))

    chunks = []
    cursor = 0
    for hdr, body in blocks:
        h_path = [hdr.lstrip("# ").strip()]
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        buf = hdr + "\n"
        start = cursor  # yaklaşık konum
        for p in paras:
            if len(buf) + len(p) + 2 <= target:
                buf += "\n" + p
            else:
                end = start + len(buf)
                chunks.append((buf.strip(), start, end, h_path))
                tail = buf[-overlap:] if overlap > 0 else ""
                buf = (hdr + "\n" + tail + p)[-target:]
                start = end - len(buf)
        if buf.strip():
            end = start + len(buf)
            chunks.append((buf.strip(), start, end, h_path))
            cursor = end
    return chunks

def sha1_of(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def clean_markdown(md: str, one_per_line: bool = True, drop_footers: bool = True) -> str:
    """
    Cleans noisy markdown to plain summary text:
    - [text](url) → "text" (optionally one-per-line)
    - removes bare URLs
    - strips [0] style indices at start of lines
    - optionally drops footer/noise lines (privacy, KVKK, terms, contact, address)
    - deduplicates lines and collapses blanks
    """
    if not isinstance(md, str):
        return ""

    out = md.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Unwrap nested angle-bracket URLs: (<https://…>) → (https://…)
    out = re.sub(r"\(<\s*([^>]+)\s*>\)", r"(\1)", out, flags=re.I)

    # Fix https:/ → https://   (and http:/ → http://)
    out = re.sub(r"\b(https?):\/(?!\/)", r"\1://", out, flags=re.I)

    # Remove [index]-style prefixes at line starts: [0] , [12] …
    out = re.sub(r"^\s*\[\d+\]\s*", "", out, flags=re.M)

    # Remove javascript:void(0) links and hash-only/empty links
    def _rm_js_void(m):
        txt = m.group(1) or ""
        return txt.strip() if txt.strip() else ""
    out = re.sub(r"\[([^\]]*)\]\(\s*javascript:void\(0\)\s*\)", _rm_js_void, out, flags=re.I)

    def _rm_hash_or_empty(m):
        txt = m.group(1) or ""
        return txt.strip() if txt.strip() else ""
    out = re.sub(r"\[([^\]]+)\]\(\s*(#|\s*)\)", _rm_hash_or_empty, out)

    # Remove empty-text links: [](...)
    out = re.sub(r"\[\s*\]\(\s*[^)]+\s*\)", "", out)

    # Convert markdown links to plain text; optionally ensure one-per-line
    def _link_to_text(m):
        txt = m.group(1)
        return f"\n{txt}\n" if one_per_line else txt
    out = re.sub(r"\[([^\]]+)\]\(\s*[^)]+\s*\)", _link_to_text, out)

    # Remove images entirely: ![alt](url)
    out = re.sub(r"!\[[^\]]*\]\(\s*[^)]+\s*\)", "", out)

    # Remove bare URLs (after link conversion)
    out = re.sub(r"\bhttps?://\S+", "", out, flags=re.I)

    # Trim spaces around newlines
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n[ \t]+", "\n", out)

    # Split to lines
    lines = [l.strip() for l in out.split("\n")]

    if drop_footers:
        footer_or_noise = re.compile(
            "|".join([
                r"^gizlilik politikası$",
                r"^kvkk$",
                r"^kullanım koşulları$",
                r"^çerez politikası$",
                r"^privacy policy$",
                r"^terms(?: and conditions)?$",
                r"^cookies?$",
                r"^iletişim(?: bilgilerimiz)?$",
                r"^contact$",
                r"^(tel:|telefon:|phone:|mailto:|e-?posta:)",
                r"(mah\.|mahalle|sokak|cad\.|no:|pk:|kat\b|daire\b|istanbul|ankara|izmir)",
                r"^copyright|^©",
            ]),
            flags=re.I
        )
        lines = [l for l in lines if l and not footer_or_noise.search(l) and len(l) > 2]
    else:
        lines = [l for l in lines if l]

    # Deduplicate (case-insensitive) while preserving order
    seen = set()
    deduped = []
    for l in lines:
        key = l.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(l)

    # Collapse excessive blanks and trim
    result = "\n".join(deduped)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result
try:
    _load_state()
except Exception:
    pass
# Run the Flask app, listening on all available IP addresses (host) at port 5003 with multithreading enabled

if __name__ == "__main__":
    # Ortam değişkenlerinden konfigürasyon al
    env = os.getenv("FLASK_ENV", "production")
    port = int(os.getenv("FLASK_PORT", "5003"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
  
    # Yalnızca development ortamında Flask dev server başlat
    if env == "development":
        app.run(
            host="0.0.0.0",
            port=port,
            debug=debug,
            threaded=True
        )