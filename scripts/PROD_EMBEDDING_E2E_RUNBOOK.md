# Prod E2E Validation Runbook (Embedding + ingest worker)

This runbook validates that:
- API accepts and queues fetcher indexing requests.
- ingest worker consumes jobs and completes indexing.
- Mongo collections are updated correctly.
- FAISS index file is updated and contains generated `faiss_id` values.

## 1) Preconditions

- `tinnten-embedding` service is running in prod.
- `tinnten-fetcher` has already crawled content for target domain.
- You have:
  - `companyId`
  - `domain`
  - (for pages test) 1+ crawled page URLs under that domain.
- If Keycloak auth is enabled, obtain a valid Bearer token.

## 2) Recommended execution point

Run from inside the embedding container so env/db/path values match runtime.

```bash
cd /Users/nilayerol/developer/tinnten/tinnten-docker

docker compose exec tinnten-embedding bash
cd /app
```

## 3) Domain-level smoke test

This queues `/api/v10/content/index/fetcher`.

```bash
python scripts/prod_embedding_e2e_check.py \
  --base-url http://localhost:5003 \
  --company-id "<COMPANY_ID>" \
  --domain "example.com" \
  --mode domain \
  --page-limit 20
```

If auth is required:

```bash
python scripts/prod_embedding_e2e_check.py \
  --base-url http://localhost:5003 \
  --token "<BEARER_TOKEN>" \
  --company-id "<COMPANY_ID>" \
  --domain "example.com" \
  --mode domain
```

## 4) Selected-pages smoke test

This queues `/api/v10/content/index/fetcher/pages` and validates only requested pages path.

```bash
python scripts/prod_embedding_e2e_check.py \
  --base-url http://localhost:5003 \
  --company-id "<COMPANY_ID>" \
  --domain "example.com" \
  --mode pages \
  --page-url "https://example.com/docs/a" \
  --page-url "https://example.com/docs/b"
```

Alternative JSON input:

```bash
python scripts/prod_embedding_e2e_check.py \
  --base-url http://localhost:5003 \
  --company-id "<COMPANY_ID>" \
  --domain "example.com" \
  --mode pages \
  --page-urls-json '["https://example.com/docs/a", "https://example.com/docs/b"]'
```

## 5) Full regression path (both)

Runs both domain + pages cases in a single run:

```bash
python scripts/prod_embedding_e2e_check.py \
  --base-url http://localhost:5003 \
  --company-id "<COMPANY_ID>" \
  --domain "example.com" \
  --mode both \
  --page-url "https://example.com/docs/a"
```

## 6) Pass / fail criteria

`PASS` for a case means all of these succeeded:
- queue request returned `202`
- `contentdocuments.index.state == completed`
- `embedding_chunks` has `chunkCount > 0` for created `documentId`
- `contentdocumentlogs` contains entries
- FAISS file exists and includes produced `faiss_id` values (or at minimum has positive `ntotal` if id map is unavailable)

On failure, script prints reasons and raw JSON diagnostics.

## 7) Data cleanup behavior

By default, script removes created test documents from FAISS + Mongo using `/api/v10/content/index/remove`.

Use `--keep-data` if you intentionally want to keep artifacts.

## 8) Useful options

- `--timeout 900` and `--poll-interval 5`
- `--mongo-uri`, `--content-db`, `--embed-db`
- `--faiss-path`
- `--fetched-from`, `--fetched-to`
- `--storage-preference "db,s3,disk"`

## 9) Exit codes

- `0`: all requested cases passed.
- `1`: one or more cases failed.
- `2`: invalid input / setup issue (missing required args, Mongo/auth/config problems).
