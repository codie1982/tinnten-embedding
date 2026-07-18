# RAG Eval Runbook

`scripts/rag_eval.py` — retrieval değişikliklerini ölçmek ve **kanıtlamak** için.
Bkz. [docs/RAG_V2_ROADMAP.md](../docs/RAG_V2_ROADMAP.md) (Faz 1).

## Neden var

Hybrid retrieval, cross-encoder reranker ve structure chunking kodda **zaten yazılı ama kapalı**. Açmak ucuz; *işe yaradı mı* bilmek değil. Bu harness her değişikliği ölçülebilir kılar.

Kritik nokta: servis, reranker yüklü değilken ya da `predict` patladığında **hata vermeden** füzyon sırasına düşer. Yani `rerank:true` gönderip "hybrid+rerank ölçtüm" sanabilirsin — gerçekte yalnız hybrid ölçmüş olursun. Harness `retrieval_meta.rerank_applied`'a bakarak bunu yakalar.

## Önkoşullar

- Çalışan bir embedding servisi (Mongo bağlantısı gerekir).
- Auth **varsayılan açık**. İki seçenek:
  - Gerçek token: `--token` veya `EMBEDDING_BEARER_TOKEN`.
  - Yerel: sunucuyu `REQUIRE_KEYCLOAK_AUTH=false` ile başlat. **Yalnız yerel/staging** — prod'da auth kapatma.
- Bağımlılıklar ambient Python'da kurulu (`requests`, `pytest`). Ayrı venv/Docker gerekmiyor.

```bash
# yerel sunucu
REQUIRE_KEYCLOAK_AUTH=false python3 app.py    # :5003
```

⚠️ `.env`'deki `MONGO_URI`'nin nereye baktığını kontrol et — yanlışlıkla prod'a bağlanma.

## Seed set hazırlama

`scripts/rag_eval_queries.json` bir **şablondur** — gerçek sorgu/etiketlerle doldur.

```json
{
  "query": "ücretsiz iade süresi kaç gün",
  "company_id": "<COMPANY_ID>",
  "relevant_urls": ["https://example.com/iade-ve-degisim"],
  "relevant_doc_ids": [],
  "query_type": "exact|price|policy|semantic|conversational",
  "content_type": "schema|clean_markdown|raw_markdown|upload"
}
```

**Etiketleme kuralları:**

1. **Her sorguyu TEK yolla etiketle** — `relevant_urls` **veya** `relevant_doc_ids`. Aynı dokümanı ikisiyle birden etiketlemek onu *iki hedef* sayar; tek bir sonuç ikisini birden "ideal" sırada dolduramayacağı için nDCG haksız yere düşer.
2. **Fetcher sayfaları için `relevant_urls` kullan.** Per-page `doc_id`, `sha1(companyId:url)[:24]` ile türetiliyor — elle etiketlemesi zor, URL pratik. Upload/kütüphane dokümanları için `relevant_doc_ids`.
3. **URL normalizasyonu otomatik**: şema, `www.`, sondaki `/` ve fragment önemsiz. Query string **korunur**.
4. **`content_type` ciddi**: fetcher, schema extraction başarılıysa **düz `\n`-join, başlıksız** metin indexliyor. Structure chunking o sayfalarda heading-path faydası veremez. Bu kırılım olmadan "structure chunking işe yaramadı" yanlış sonucuna varırsın.
5. **Sorgu tiplerini dengele**: hybrid'in asıl kazancı `exact`/`price`'ta (ürün kodu, tutar), dense'inki `semantic`'te.

## Çalıştırma

```bash
# üç config birden (dense / hybrid / hybrid+rerank), tek sunucu
python3 scripts/rag_eval.py \
  --base-url http://localhost:5003 \
  --queries scripts/rag_eval_queries.json \
  --k 10

# JSON çıktısı (koşuları saklayıp kıyaslamak için)
python3 scripts/rag_eval.py --json-out /tmp/eval-$(date +%F).json

# rollout kapısı: reranker gerçekten çalışmadıysa başarısız ol
python3 scripts/rag_eval.py --require-rerank

# latency AYRI koşu (kalite setinden bağımsız, warmup hariç)
python3 scripts/rag_eval.py --latency-runs 30 --latency-warmup 3 --latency-config hybrid
```

`hybrid`/`rerank` **request-seviyesi** parametreler → üç config tek sunucuda, restart/redeploy olmadan ölçülür.

## Çıktıyı okuma

- **Overall**: config × (recall@k, nDCG@k, MRR). `dense` taban çizgin.
- **Breakdown by query_type**: hybrid `exact`/`price`'ta belirgin kazandırmalı.
- **Breakdown by content_type**: structure chunking kararı buraya bakar.
- **INTEGRITY**: `rerank_applied=false` uyarısı görüyorsan **o satırlar hybrid+rerank değil, düz hybrid'dir.** Sayıları öyle raporlama.

## PASS/FAIL

| Exit | Anlam |
|---|---|
| 0 | Geçti |
| 1 | Eşiğin altında (`--fail-under-recall`), sorgu hatası, veya `--require-rerank` iken reranker çalışmadı |
| 2 | Kötü girdi (dosya yok/bozuk, k≤0, preflight/auth başarısız) |

## Latency'yi neden ayrı ölçüyoruz

50–100 sorguluk kalite seti güvenilir bir **p95** için azdır; ayrıca ilk çağrı model yükleme/cache maliyeti öder — hiçbir gerçek kullanıcı isteğinin ödemediği bir bedel. Bu yüzden `--latency-runs` ayrı koşar ve ilk `--latency-warmup` çağrıyı atar.

## Bilinen tuzak: title-promotion reranker'ı ezer

`_promote_title_matches` reranker'dan **sonra** çalışıyor ve ham `score`'a göre sıralıyor — `rerank_score`'a değil. Yani title/heading eşleşmesi olan sorgularda reranker'ın sırası ezilir. Reranker kazanımını ölçerken bunu hesaba kat; düzeltme Faz 4'te planlı (roadmap KN4).

## Testler

```bash
python3 -m pytest tests/test_rag_eval_metrics.py -q     # 42 test, saf metrikler
```

## Regresyon kontrolü

Repo'da **önceden kırık testler var** (`test_categories.py`, `category_store` devre dışı). "Kaç fail" tek başına anlamsız — taban çizgisiyle karşılaştır:

```bash
git stash push -m wip && python3 -m pytest tests/ -q | tail -3   # BASELINE
git stash pop            && python3 -m pytest tests/ -q | tail -3   # DEĞİŞİKLİKLE
```
