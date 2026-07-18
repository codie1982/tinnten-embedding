# RAG v2 Roadmap — "Hybrid Contextual RAG v2"

> Durum: uygulama büyük ölçüde tamam (kod + testler). Kalan iş operasyonel
> (deploy config açma + gerçek veriyle ölçüm + pilot). Dosya:satır referansları
> yazıldığı tarihteki koda aittir; zamanla kayabilir.

## 0. Uygulama durumu

| Faz | Kod | Test | Kalan (operasyonel) |
|---|---|---|---|
| 1a — `retrieval_meta` + auth'lu `/config` | ✅ app.py | ✅ | — |
| 1b — eval harness | ✅ scripts/rag_eval.py | ✅ 42 metrik | gerçek seed set (DB erişimi) |
| 2 — `$text` preflight + config yüzeyi | ✅ scripts/rag_preflight.py, .env.example, compose | ✅ 10 | — |
| 3 — firma-bazlı canary (server) | ✅ tinnten-server retrievalCanary.js | ✅ 12 | canary listesini doldur |
| 4 — reranker warmup + bake | ✅ app.py, Dockerfile | ✅ 3 | p95 ölç, model bake+aç |
| 5 — idempotent re-ingest (versiyon+lock) | ✅ mongo_store, content_store, ingest_worker, app.py | ✅ 24 | — |
| 6 — faithful schema + fetcher reindex | ✅ schema_index_text.py, fetcher reindex_documents.py | ✅ 11+10 | reindex'i pilotta çalıştır |
| 7 — source-type `auto` chunking | ✅ ingest_worker.py | ✅ 10 | `CHUNK_STRATEGY=auto` aç + reindex |
| 8 — pilot rollout | — | — | tek firma pilot → yay |

**Kilit çıkış:** RAG kalitesi henüz DEĞİŞMEDİ — tüm anahtarlar hâlâ güvenli varsayılanda (hybrid/reranker kapalı, chunking char). Kod + ölçüm + geri-alma altyapısı hazır; açma operasyonel karar. **Yan kazanç zaten aktif:** Faz 5 canlı fetcher duplicate chunk bug'ını kapattı (kod deploy edilince).

**Yeni env değişkenleri:** `HYBRID_SEARCH_ENABLED`, `RERANKER_MODEL`, `RERANK_TOP_N`, `CHUNK_STRATEGY` (char|structure|**auto**), `INDEX_SCHEMA_CONTENT_ENABLED`, `INDEX_SCHEMA_MIN_CHARS`, `INGEST_LEASE_SECONDS` (embedding); `RAG_HYBRID_CANARY_COMPANIES`, `RAG_RERANK_CANARY_COMPANIES` (tinnten-server). Hepsi güvenli varsayılanlı → deploy davranışı değiştirmez.

## 1. Kapsam ve mimari

`tinnten-embedding` **retrieval servisidir**. İçinde LLM/generation **yoktur** (bağımlılıklar: `sentence-transformers`, `faiss-cpu`, `torch`). Query understanding, answer generation ve verification **orchestration katmanında** (`tinnten-server`) yaşar. Bu ayrım roadmap boyunca korunur.

**İngest hattı** (`workers/ingest_worker.py`):
```
RabbitMQ (content_indexing_queue)
  → içerik yükle (upload / fetcher / S3 / inline text)
  → temizle (html→markdown, cleanup)
  → chunk (char | structure)
  → embed (SentenceTransformer, L2-normalize)
  → faiss_id ayır (Mongo counter) → FAISS'e ekle
  → chunk'ları Mongo'ya yaz (embedding_chunks)
```

**Retrieval hattı** (`app.py:_chunk_search_response`):
```
sorgu → encode → FAISS dense (over-fetch)
  → [hybrid] Mongo $text lexical → RRF füzyon
  → [reranker] cross-encoder
  → title-promotion + doc başına dedup
  → [radius>0] komşu chunk birleştirme
```

**İlgili repolar**: `tinnten-embedding` (retrieval) · `tinnten-fetcher` (crawl + indexleme tetikleme) · `tinnten-server` (arama istemcisi) · `tinnten-docker` (runtime config).

## 2. Mevcut durum: rapor → kod eşleşmesi

| Öneri | Durum | Nerede |
|---|---|---|
| Hybrid BM25+Vector | ✅ Yazılı, **KAPALI** | `_rrf_fuse` app.py:1415, `HYBRID_SEARCH_ENABLED` |
| RRF fusion (k=60) | ✅ Yazılı | app.py:1415 |
| Gerçek BM25 | 🟡 Mongo `$text` (stemming yok) | `text_search_chunks` mongo_store.py:225 |
| Cross-encoder reranker | ✅ Yazılı, **KAPALI** | `_maybe_rerank` app.py:1473, `RERANKER_MODEL` |
| Contextual chunk (heading header) | 🟡 Yazılı, **KAPALI** | `chunk_markdown_structure` chunker.py:136 |
| Contextual `contextSummary` (LLM) | 🔴 Yok | — |
| Parent-child / komşu chunk | 🟡 Radius var, parent yok | app.py:1314, 1383 |
| Section/page/domain summary (RAPTOR-lite) | 🔴 Yok | — |
| Structured metadata (headingPath/pageType/lang) | 🔴 Yok (text'e gömülü) | — |
| Eval veri seti + metrikler | 🔴 Yok | — |
| Query router / rewrite / decomposition | ⚪ Başka servis | orchestration |
| Corrective RAG | ⚪ Başka servis (confidence sinyali burada üretilebilir) | — |
| GraphRAG / ColBERT | 🔴 Yeni, büyük | — |

**Özet**: en güçlü üç özellik (hybrid, reranker, structure chunking) **koda hazır ama hiçbir deploy artefaktında set değil**.

## 3. Doğrulanmış kritik bulgular (cross-repo)

Bunlar plan yazılırken kod üzerinde doğrulandı; roadmap'in şeklini belirlediler.

1. **Config hiçbir yerde set değil.** Env kaynağı `tinnten-docker/docker-compose.yml:220-251` (`env_file: ./tinnten-embedding/.env` + inline `environment:`). `HYBRID_SEARCH_ENABLED`, `RERANKER_MODEL`, `RERANK_TOP_N`, `CHUNK_STRATEGY` **hiçbir** compose/Dockerfile/.env'de yok. `.env` değiştirmek ≠ deploy değişikliği.

2. **Re-ingest idempotent değil — bu CANLI bir bug.** `_chunk_and_embed` (ingest_worker.py:1436-1466) yalnız ekler, eskiyi silmez. Fetcher her değişen sayfayı stabil `information_page_doc_id` ile `POST /content/index`'e yolluyor → **bugün her sayfa güncellemesinde duplicate chunk + orphan vektör** üretiliyor.

3. **"Atomik replace" yanıltıcı.** `replace_embeddings` yalnız FAISS içinde atomiktir (embedding_engine.py:179); **Mongo+FAISS bütünlüğünü kapsamaz**. Naive `Mongo sil → FAISS replace → Mongo ekle` sırası hata/eşzamanlılıkta iki store'u ayrıştırır.

4. **Structure chunking her içerikte fayda vermez.** Fetcher schema extraction başarılıysa **düz `\n`-join, başlıksız** metni indexliyor (`_select_index_content` result_processor.py:1118). Başlık yoksa heading-path avantajı yok — yalnız clean/raw markdown dallarında işe yarar.

5. **Faithful schema reindex MÜMKÜN.** `extracted_content` Mongo'da kalıcı (`crawl_results.extracted.extracted_content`); yalnız >12MB sanitizer'ı düşürür (`mongo_sanitized.steps` izi bırakır). **Extraction deterministik/offline** — kalıcı css/xpath selector'lar; LLM yalnız onboarding'de schema *üretir*. Embedding tarafı bu alanı sadece **istemiyor** (projection'da yok: fetcher_store.py:70-83).

6. **Eval'in kanıtı yok.** Server `hybrid`/`rerank` göndermiyor (`embeddingSearch.service.js:49`) → per-request canary yok. Reranker yüklenemezse **sessizce** eski sıraya düşer (app.py:1482). Yanıtta hangi yolun/modelin çalıştığını gösteren alan yok. `/config` route yok. Ayrıca `_promote_title_matches` reranker'dan **sonra** `(title_match, score)` ile yeniden sıralıyor (app.py:1673) — `rerank_score`'a bakmıyor.

7. **Reranker operasyonel riskler.** Dockerfile model indirmiyor → ~2GB ilk sorguda runtime'da iner; HF cache volume'da kalıcı değil (`embedding_data:/app/data` yalnız FAISS). `--workers 1`, `mem_limit 12g`. Yükleme başarısızsa servis healthy kalır, sessiz degrade olur. **preprod compose'da embedding servisi YOK** → gerçek staging yok.

## 4. Fayda / maliyet (ROI)

| Geliştirme | Fayda | Maliyet / Risk | ROI |
|---|---|---|---|
| Eval harness | Tüm kararları ölçülebilir kılar | Etiketleme emeği | ⭐⭐⭐⭐⭐ |
| Hybrid'i aç | Exact-match recall (ürün kodu, fiyat, özel isim) | +1 Mongo `$text`/sorgu | ⭐⭐⭐⭐⭐ |
| Reranker'ı aç | En büyük precision sıçraması | CPU latency, ~2GB RAM, paketleme | ⭐⭐⭐⭐ |
| Idempotent re-ingest | **Canlı duplicate bug'ını kapatır** | Versiyonlu swap + lock tasarımı | ⭐⭐⭐⭐ |
| Structure chunking | Daha iyi chunk sınırı + context header | Reindex; schema'da faydasız | ⭐⭐⭐ |
| Contextual summary (LLM) | Retrieval hatasında en yüksek düşüş | Chunk başına LLM $ | ⭐⭐⭐ |
| Parent-child | Precision + bağlam | Şema + token | ⭐⭐⭐ |
| Gerçek BM25 (OpenSearch) | Daha iyi lexical | Yeni altyapı + ops | ⭐⭐⭐ |
| Model upgrade (BGE-M3) | Taban relevance | Tam reindex (dim değişir) | ⭐⭐⭐ |
| Router/CRAG/GraphRAG/ColBERT | Multi-hop, global analiz | Yüksek; çoğu başka serviste | ⭐ |

**Beklenen kazanım (yön gösterici, garanti değil):** Anthropic'in kendi değerlendirmesinde Contextual Embeddings + Contextual BM25 retrieval hatasını **-49%**, reranking eklenince **-67%** azaltmıştı — bunlar Anthropic'in kendi test sonuçlarıdır, üst-sınır referansı olarak okunmalıdır. Elastic hybrid arama için RRF önerir. Hybrid + reranker iki-aşamalı yapının tek başına dense'e üstünlüğü bağımsız benchmark'larda da gözlenmiştir.

## 5. Fazlar

Sıra **doğrulama-öncelikli**: ölçemediğimiz şeyi açmıyoruz.

- **Faz 1 — Eval + capability doğrulaması.** `retrieval_meta` (hangi yol/model gerçekten çalıştı) + auth'lu yan-etkisiz `/config`; `scripts/rag_eval.py` (dense / hybrid / hybrid+rerank), seed set (`relevant_urls` + `content_type`), recall@k · nDCG@10 · MRR, **query_type × content_type kırılımı**, ayrı latency koşusu (warmup hariç).
- **Faz 2 — Önkoşul doğrulaması.** Mongo `$text`: `listIndexes` + çıplak + **`company_id` filtreli** sorgu (index hataları yutuluyor). Config katmanlama: `.env.example` · staging (preprod'a embedding servisi) · prod (compose inline) · rollout/rollback.
- **Faz 3 — Hybrid'i canary'de aç.** Tercih: `tinnten-server` payload'ına per-company override (embedding kodu değişmez; request alanı zaten onurlandırılıyor). Global env alternatif.
- **Faz 4 — Reranker: önce operasyon.** Bake→`/opt/huggingface` + `HF_HOME` (volume `/app/data`'yı örter — karıştırma) + **revision pin**; warmup + load-error görünürlüğü; `RERANK_TOP_N`≈20; p95 ölç. Title-promotion reranker'ı ezmesin.
- **Faz 5 — Üretim-güvenli idempotent re-ingest.** `ingest_version` + `active_ingest_version`; versiyon-seçici repo metotları (mevcut `delete_chunks_by_doc` versiyon-kör → kullanılamaz); **lease'li CAS lock** (`update_index_fields` lock semantiği taşımıyor); **önce ekle → CAS swap → eskiyi temizle**; arama yalnız aktif sürümü görür; fault-injection + eşzamanlılık + lease-expiry testleri.
- **Faz 6 — Fetcher-aware reindex.** `domain_subscriptions` + `urls`; **watermark'a dokunma** — enqueue → `202`'yi başarı sayma → `ready` bekle → yalnız başarıda watermark güncelle → başarısızı retry manifestine. Faithful schema metni: projection'a `extracted` + seçim fonksiyonunun port'u + parity testi.
- **Faz 7 — Source-type structure chunking.** Global açma; markdown→`structure`, schema→`char` (veya schema'ya kontrollü `#` başlık üretimi ayrı deney). `content_type` kırılımıyla doğrula.
- **Faz 8 — Pilot → kademeli rollout.** Tek canary firma → doğrula → genişlet. Rollback: env geri al / canary'den çıkar / reranker devre dışı.

## 5b. Config katmanlama ve rollout/rollback

**Runtime config kaynağı** (bu sıra önemli — sonraki öncekini ezer):

| Katman | Yer | Rol |
|---|---|---|
| 1. Örnek/doküman | `tinnten-embedding/.env.example` | Versiyon kontrollü referans. **Deploy'u etkilemez.** |
| 2. Host env dosyası | `tinnten-embedding/.env` (compose `env_file`) | Sunucudaki gerçek değerler (sırlar burada) |
| 3. Compose inline | `tinnten-docker/docker-compose.yml` → `environment:` | **En yüksek öncelik.** Retrieval anahtarları burada. |

**Staging durumu:** `docker-compose.preprod.yml`'de embedding servisi **tanımlı değil** → gerçek bir staging ortamı yok. Bu yüzden kademeli açılış için tercih edilen yol, ayrı ortam değil **firma-bazlı canary** (Faz 3, server override): tek prod servisi, request-seviyesi `hybrid`/`rerank` ile sadece seçili firmalar.

### Rollout (hybrid örneği)

```bash
# 0) ÖN KOŞUL — sessiz arızayı önce ele
python3 scripts/rag_preflight.py --company-id <ID> --base-url http://localhost:5003
#    (1) chunk_text_search index'i var mı  (2) $text çalışıyor mu
#    (3) company_id filtreli $text sonuç dönüyor mu   ← gerçek sorgu şekli
#    FAIL varsa AÇMA: hybrid tek bacakla çalışır ve fark edilmez.

# 1) Taban çizgisini ölç (dense)
python3 scripts/rag_eval.py --config dense --json-out /tmp/baseline.json

# 2) Canary aç (tercih: firma-bazlı, Faz 3) — veya global:
#    docker-compose.yml → HYBRID_SEARCH_ENABLED=true
docker compose up -d tinnten-embedding

# 3) Kanıtla: retrieval_meta gerçekten hybrid mi?
curl -s .../api/v10/config | jq '.hybrid_search_enabled'
python3 scripts/rag_eval.py --json-out /tmp/hybrid.json    # + INTEGRITY satırlarını oku

# 4) Karşılaştır: exact/price query_type'ta recall arttı mı?
```

### Rollback

| Ne | Nasıl | Etki |
|---|---|---|
| Hybrid | `HYBRID_SEARCH_ENABLED=false` + restart | Anında dense'e döner |
| Hybrid (canary) | Firmayı canary listesinden çıkar | Restart bile gerekmez |
| Reranker | `RERANKER_MODEL=` (boşalt) + restart | Füzyon sırası kalır, arama çalışmaya devam eder |
| Structure chunking | `CHUNK_STRATEGY=char` + **reindex** | ⚠️ Geri dönüş reindex ister — diğerleri gibi anlık değil |

**Neden hybrid/reranker rollback'i ucuz:** ikisi de saf okuma yolu, veri şeması değiştirmiyor. **Structure chunking farklı** — indexlenmiş veriyi değiştirir, o yüzden en sona bırakıldı ve pilotla açılır.

### Faz 7-8 — Structure chunking + faithful reindex pilotu (tek firma)

Reindex indexlenmiş veriyi değiştirir → önce TEK canary firmada, job-completion takipli sürücüyle:

```bash
# 1) (embedding) auto chunking + faithful schema metnini AÇ
#    docker-compose.yml → CHUNK_STRATEGY=auto, INDEX_SCHEMA_CONTENT_ENABLED=true
docker compose up -d tinnten-embedding

# 2) (fetcher) önce dry-run — ne reindex edilecek?
python3 scripts/reindex_documents.py --company-id <ID>              # dry-run varsayılan
# 3) gerçek reindex (Faz 5 idempotent → hard-remove GEREKMEZ)
python3 scripts/reindex_documents.py --company-id <ID> --execute
#    her sayfa: enqueue → contentdocuments.index TAZE ready olana kadar bekle
#    (202 başarı sayılmaz); başarısızlar reindex_failures.json'a yazılır

# 4) doğrula: duplicate yok + kalite
#    - doküman başına chunk sayısı beklenenle eşleşmeli (Faz 5 idempotency)
#    - python3 scripts/rag_eval.py  → content_type kırılımında markdown sayfalarda
#      structure kazancı; schema sayfalarda 'auto' char seçtiği için nötr
```

**Pilot kapıları:** (a) reindex sonrası FAISS `ntotal` ≈ önceki (duplicate yok), (b) eval'de gerileme yok, (c) `reindex_failures.json` boş veya kabul edilebilir. Geçerse firma listesini genişlet.

## 6. Karar noktaları

- **KN1 — Reindex metin kaynağı → ÇÖZÜLDÜ.** `/fetcher/pages` DB-read yolu faithful yapılabilir (projection + port + parity testi). Inline yola gerek yok. Markdown fallback şart (`db` storage yoksa / sanitizer düşürmüşse).
- **KN2 — Reranker paketleme.** Öneri: build-time bake, volume'un örtmediği dizine (`/opt/huggingface`), revision pinli.
- **KN3 — Structure chunking kapsamı.** Öneri: global değil, source-type'a göre.
- **KN4 — Title-promotion ↔ reranker sırası.** Öneri: promotion'ı rerank öncesine al veya rerank-aware yap; eval ölçsün.

## 7. Kapsam dışı (sonraki fazlar)

Contextual summary (LLM), structured chunk metadata, parent-child / RAPTOR, gerçek BM25 (OpenSearch/Elastic), embedding modeli upgrade (BGE-M3 / multilingual-e5), query router, Corrective RAG, GraphRAG, ColBERT / multi-vector.

## 8. Bilinen ayrı bulgu

`_load_fetcher_content_from_s3` (`workers/ingest_worker.py`) `s3_key_json`'a `json.loads` uyguluyor; ancak o key artık **gzipped JSONL batch** adresliyor → bu S3 fallback'i ölü görünüyor. Ayrı iş olarak ele alınmalı.
