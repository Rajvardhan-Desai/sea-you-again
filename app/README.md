# MM-MARAS Dashboard

User-facing operational layer on top of the MM-MARAS research model — a daily ingestion → inference → alert pipeline plus a public web dashboard so fishers, coastal authorities, and researchers can act on harmful-algal-bloom (HAB) forecasts in time to prevent stock losses.

For the underlying ML model and its accuracy numbers, see the [root README](../README.md).

---

## What's in here

```
app/
├── api/         FastAPI service (forecast / subscriptions / alerts / admin / playbook)
├── db/          SQLAlchemy 2.x models + Alembic migrations (Postgres + PostGIS)
├── inference/   Shared forward-pass + checkpoint loader + PNG renderer
├── playbook/    YAML action-band playbook surfaced in alerts and on /playbook
├── web/         Next.js 14 dashboard (App Router, Leaflet, TanStack Query)
├── worker/      Daily orchestrator, ingest, alert engine, notifications, scheduler
└── tests/       pytest suite (alert engine + API)
```

---

## Architecture

```
┌──────────┐   daily 02:00 UTC    ┌────────────┐
│scheduler │ ─────────────────▶   │   worker   │
└──────────┘                       │            │
                                   │ 1. ingest  │  CMEMS · ERA5 · GloFAS
                                   │ 2. infer   │  reuses scripts/eval.py forward
                                   │ 3. render  │  PNG overlays per layer × horizon
                                   │ 4. alert   │  geofenced rule per AOI subscription
                                   │ 5. notify  │  email · SMS · in-app
                                   └────┬───────┘
                                        ▼
              ┌────────────────────────────────────────────┐
              │ Postgres+PostGIS  ·  /data/runs/<date>/    │
              └────────────────────┬───────────────────────┘
                                   ▼
   browser ──▶ nginx ──▶  Next.js (web)   ──┐
                    └──▶  FastAPI (api)  ◀──┘   reads runs + artifacts
```

---

## Prerequisites

| Tool | Version |
|---|---|
| Docker Desktop / Engine | ≥ 24 |
| Docker Compose plugin   | ≥ 2.20 |
| Disk                    | ≥ 15 GB free (images + sample run artifacts) |
| RAM                     | ≥ 8 GB recommended (worker keeps the 46.4 M-param model in memory) |

You **do not** need Python or Node installed locally to run the dashboard — everything builds inside containers.

External credentials (only needed for live data ingestion — the dashboard runs fine without them on existing artifacts):

- **Copernicus Marine** (`copernicusmarine` CLI account) — Chl-a + ocean physics
- **CDS / ERA5** (`~/.cdsapirc` API key)        — wind, MSLP, precipitation
- **GloFAS** (CDS API key, same as above)        — freshwater discharge
- **SendGrid SMTP relay** *or* any SMTP server   — email alerts (optional in dev — use Mailpit)
- **Twilio**                                     — SMS alerts (optional)

---

## Quickstart

### 1. Configure the environment

```bash
cp infra/env.example infra/.env
# edit infra/.env — at minimum set ADMIN_TOKEN and SECRET_KEY
```

For local development you can leave the SendGrid / Twilio / CMEMS / ERA5 vars empty — the stack will start, the API will serve, the daily run will only fail on the ingest phase (which you can skip in dev).

To use the bundled Mailpit dev SMTP sink, override these in `infra/.env`:

```ini
SMTP_HOST=mailpit
SMTP_PORT=1025
SMTP_USER=
SMTP_PASS=
```

### 2. Bring up the stack

```bash
docker compose -f infra/docker-compose.yml --env-file infra/.env --profile dev up -d --build
```

The first build takes **5–10 minutes** (PyTorch CPU wheel ~250 MB, GDAL/GEOS apt packages ~210 MB). Subsequent builds use Docker's layer cache and finish in seconds.

### 3. Verify

```bash
docker compose -f infra/docker-compose.yml ps
curl http://localhost:8080/api/health
# → {"status":"ok","version":"3.6.0","latest_run_date":null}
```

### 4. Open the dashboard

| URL | What |
|---|---|
| http://localhost:8080/                | Landing page |
| http://localhost:8080/map             | Forecast map (Chl-a · bloom · ERI · impact) |
| http://localhost:8080/subscribe       | Draw your AOI and subscribe |
| http://localhost:8080/playbook        | Action playbook |
| http://localhost:8080/about           | Model card + accuracy |
| http://localhost:8080/admin/login     | Admin (paste your `ADMIN_TOKEN`) |
| http://localhost:8025/                | Mailpit UI (dev) — view alert emails |
| http://localhost:8000/api/docs        | FastAPI Swagger UI |
| http://localhost:8000/metrics         | Prometheus metrics |

> Ports may differ if you customised `infra/docker-compose.yml`. Check `docker compose ps`.

### 5. Stop / clean up

```bash
docker compose -f infra/docker-compose.yml --profile dev down          # stop, keep data
docker compose -f infra/docker-compose.yml --profile dev down -v       # also wipe DB volume
```

---

## Daily operations

### Trigger a run manually

```bash
# Yesterday's data (default)
docker compose -f infra/docker-compose.yml run --rm worker python -m app.worker.daily_run

# Specific date
docker compose -f infra/docker-compose.yml run --rm worker \
    python -m app.worker.daily_run --date 2026-04-25

# Skip ingest (use already-downloaded raw files in data/raw/<date>/)
docker compose -f infra/docker-compose.yml run --rm worker \
    python -m app.worker.daily_run --date 2026-04-25 --start-phase infer

# Re-evaluate alerts only (no re-inference)
docker compose -f infra/docker-compose.yml run --rm worker \
    python -m app.worker.daily_run --date 2026-04-25 --start-phase alert
```

### Trigger via API (admin)

```bash
curl -X POST http://localhost:8080/api/admin/runs/trigger \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"run_date":"2026-04-25"}'
```

### Watch logs

```bash
docker compose -f infra/docker-compose.yml logs -f worker scheduler api
```

### Inspect a run

Each successful run writes:

```
data/runs/<YYYY-MM-DD>/
├── recon.npz · forecast.npz · bloom.npz · eri.npz · impact.npz
├── overlays/
│   ├── recon_d{1..5}.png · forecast_d{1..5}.png · bloom_d{1..5}.png
│   ├── eri_d{1..5}.png · impact_d{1..5}.png
│   └── colorbars/{layer}.png
└── metadata.json
```

Plus a row in the `runs` Postgres table with `status`, `started_at`, `finished_at`, `inference_metrics` (JSON), and `error_text` if it failed.

---

## Subscribing to alerts (end-user flow)

1. Visit `/map`, draw a polygon over your farm / fishing zone with the AOI tool.
2. Visit `/subscribe`, fill name + email (+ optional phone for SMS), set the bloom-probability threshold, pick channels (`email`, `sms`, `inapp`).
3. A confirmation link is sent (visible in Mailpit during dev).
4. After confirmation, every daily run evaluates the polygon — if bloom probability or ERI inside it exceeds the threshold over the next 5 days, an alert is dispatched with the matching playbook actions inlined.

The **alert rule** is:

```
fire_if  max_over_horizons(bloom_prob[polygon]) > severity_threshold
      OR max_over_horizons(eri_class[polygon]) >= 3
```

Idempotent on `(subscription_id, run_id)` — running the same daily job twice will not double-send.

---

## Admin

```bash
# Sign in (sets httpOnly cookie)
# Open http://localhost:8080/admin/login and paste your ADMIN_TOKEN

# Or via API
curl -X POST http://localhost:8080/api/admin/session \
     -H "Content-Type: application/json" \
     -d "{\"token\":\"$ADMIN_TOKEN\"}" -c admin.cookie

curl http://localhost:8080/api/admin/runs?limit=50 -b admin.cookie
```

Admin pages:

- **`/admin`**                 — last 50 runs, status pills, click-to-retry
- **`/admin/runs/<id>`**       — full run detail (ingest summary + inference metrics + error trace)
- **`/admin/subscriptions`**   — paginated subscriber list

---

## Local development without Docker

```bash
# Backend
python -m venv .venv && source .venv/bin/activate     # or `.\.venv\Scripts\activate` on PowerShell
pip install -e ".[app,dev]"

# Run Postgres+PostGIS in Docker only
docker compose -f infra/docker-compose.yml up -d postgres redis

# DB migrations
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/mmaras alembic upgrade head

# API
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/mmaras \
ADMIN_TOKEN=dev SECRET_KEY=dev-secret-at-least-32-chars-long \
uvicorn app.api.main:app --reload --port 8000

# Frontend
cd app/web
npm ci
npm run dev   # http://localhost:3000
```

---

## Tests

```bash
# Python (alert engine + API smoke tests)
pytest app/tests/ -v

# Frontend (Vitest + React Testing Library)
cd app/web
npm test
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `api` healthcheck fails on cold start | Alembic migration is running; wait 30–60 s. Check `docker compose logs api`. |
| `worker` crashes with `FileNotFoundError: best.pt` | Place a trained checkpoint at `model/checkpoints/best.pt` (or set `CHECKPOINT_FILE` in `.env`). The dashboard only needs it for the inference phase — you can still browse historic runs without it. |
| `getaddrinfo ENOTFOUND api` during web build | Stale Docker layer cache — rebuild with `--no-cache`. Pages that fetch from the API are marked `dynamic = "force-dynamic"`. |
| `502 Bad Gateway` from nginx for `/api/*` on first load | API hasn't passed its healthcheck yet (wait ~40 s start_period). |
| No alerts firing despite obvious bloom in the map | Subscriber must have clicked the magic-link confirm email. Check `subscriptions.confirmed_at IS NOT NULL`. |
| CMEMS / ERA5 download fails in ingest | Check credentials in `infra/.env`; for offline dev, pre-stage NetCDFs into `data/raw/<date>/` and run with `--start-phase infer`. |
| `pip install -e ".[app]"` fails with `ModuleNotFoundError: setuptools.backends` | Update to the merged version of `pyproject.toml` (`build-backend = "setuptools.build_meta"`). |
| Build context > 1 GB / disk pressure | Add `infra/.dockerignore` excluding `data/`, `model/checkpoints/`, `.venv/`, `node_modules/`, `.next/`. |

### Useful one-liners

```bash
# Tail all services
docker compose -f infra/docker-compose.yml logs -f

# Open a Postgres shell
docker compose -f infra/docker-compose.yml exec postgres \
    psql -U postgres -d mmaras

# Inspect Prometheus metrics
curl -s http://localhost:8000/metrics | grep mmaras_

# Test alert dispatch with a synthetic fixture
docker compose -f infra/docker-compose.yml run --rm worker \
    python -m app.worker.daily_run --date 2026-04-25 \
    --use-fixture app/tests/fixtures/fake_batch.npz
```

---

## Configuration reference

All settings flow through `infra/.env` → docker-compose env vars → `app/api/settings.py` (`pydantic-settings`).

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `postgres` / `postgres` / `mmaras` | DB credentials |
| `ADMIN_TOKEN`               | `changeme`                | Bearer token for `/api/admin/*` |
| `SECRET_KEY`                | `changeme`                | Signs subscription confirm + unsubscribe tokens (`itsdangerous`) |
| `SMTP_HOST` / `_PORT` / `_USER` / `_PASS` / `_FROM` | SendGrid defaults | Email transport |
| `TWILIO_SID` / `_TOKEN` / `_FROM` | empty                | SMS transport (skipped if blank) |
| `CMEMS_USERNAME` / `CMEMS_PASSWORD` | empty            | Copernicus Marine ingestion |
| `CDSAPI_KEY`                | empty                     | ERA5 / GloFAS ingestion |
| `CHECKPOINT_FILE`           | `best.pt`                 | Filename inside `model/checkpoints/` |
| `CHECKPOINT_BACKUP_DIR`     | `./data/backups`          | Daily backup target (host path) |
| `PUBLIC_BASE_URL`           | `http://localhost`        | Used in alert email links + `/api/forecast/latest` overlay URLs |

---

## Service map

| Service     | Port (host)         | What |
|-------------|---------------------|------|
| `nginx`     | 8080 / 8443         | Reverse proxy + static overlays |
| `api`       | 8000                | FastAPI |
| `web`       | 3000                | Next.js |
| `postgres`  | 5432                | DB |
| `redis`     | 6379                | Rate limiting + APScheduler jobstore |
| `worker`    | —                   | On-demand inference runs |
| `scheduler` | —                   | APScheduler — daily 02:00 UTC + checkpoint backup 03:00 UTC |
| `mailpit`   | 8025 (UI), 1025 SMTP | Dev email sink (only with `--profile dev`) |

---

## Security notes (single-VM deployment)

- `ADMIN_TOKEN` is a single shared secret — rotate by updating `infra/.env` and `docker compose restart api web`.
- Subscriber confirm/unsubscribe tokens are signed with `SECRET_KEY` (`itsdangerous`, 24 h TTL for confirm, 30 d for unsub).
- Public ports: only `nginx` is exposed beyond `127.0.0.1`. `postgres`, `redis`, `api`, `web` are bound to localhost in `docker-compose.yml`.
- Put nginx behind TLS for production (mount certs at `infra/certs/`).

---

## Further reading

- Plan / design doc — `~/.claude/plans/i-want-to-develop-generic-quail.md`
- Research model README — [../README.md](../README.md)
- FastAPI OpenAPI spec — http://localhost:8000/api/docs (after the stack is up)
