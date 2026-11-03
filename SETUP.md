# Setup, Run & Test

Architecture and design: [README.md](README.md). Demo runbook: [DEMO.md](DEMO.md).

## 1. Requirements

| | |
|---|---|
| Python | 3.12 |
| OS | Windows, macOS or Linux |
| GPU | Not needed — models run on the providers' servers |
| Disk | ~1.5 GB (packages + Playwright's Chromium) |

## 2. Install

macOS / Linux
```bash
cd agentic-content-platform
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m playwright install chromium
cp .env.example .env
```

Windows (PowerShell)
```powershell
cd agentic-content-platform
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt; pip install -e .
python -m playwright install chromium
Copy-Item .env.example .env
```

If security policy blocks `playwright install`, set
`PLAYWRIGHT_BROWSER_CHANNEL=msedge` in `.env` to use the installed Edge.

## 3. Keys (all optional — everything runs offline without them)

| Key | Used for | Without it |
|---|---|---|
| `GEMINI_API_KEY` | Text and vision agents (free: https://aistudio.google.com/app/apikey) | Deterministic fallbacks: offline picture reading, the curated copy bank, measured evaluation |
| `OPENROUTER_API_KEY` | Fallback text model; second image vendor | Gemini only |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | AI images (daily free allowance) | No supporting visuals in brochures, no hero image in blogs — placeholders are never printed |
| `HF_TOKEN` | Hugging Face images, only with `IMAGE_GEN_PROVIDER=huggingface` | — |

Never commit `.env` or share a folder that contains it. If a token has ever
been shared, revoke it and create a new one.

Check the machine:
```bash
python scripts/preflight.py
```
It checks Python, packages, the browser, keys, fonts, storage, and makes a
real brochure end to end, offline.

## 4. Run

```bash
streamlit run streamlit_app.py          # http://localhost:8501
uvicorn src.api.main:app --reload       # http://localhost:8000/docs
```

Brochure from the command line (the demo property is included):
```bash
python scripts/generate_brochure.py --property-id e128 --template "Luxury - Landscape"
python scripts/generate_brochure.py --property-id e128 --orientation portrait
```

Your own property:
```bash
python scripts/ingest_property.py --source ./photos --property-id green-acres \
    --name "Green Acres" --city Pune --category Mid-Tier \
    --config "2 BHK|680 sq ft|Rs 52 Lakh" --config "3 BHK|940 sq ft" \
    --phone "+91 98765 43210" --floorplan plans/2bhk.png --masterplan site.png
python scripts/generate_brochure.py --property-id green-acres
```

Blog:
```bash
python scripts/ingest_blog.py --blog-id kokapet --topic "Is Kokapet a good place to buy a home?" \
    --facts "Kokapet is about 4 km from the Financial District" --cta-text "Book a site visit"
python scripts/generate_blog.py --blog-id kokapet
```

## 5. Test

```bash
python -m pytest -q
```
The brochure tests under `tests/brochure/` and `tests/integration/` render
real brochures in Chromium (skipped if Playwright's Chromium is missing).
Tests never touch the network: models are refused, image generation is the
offline provider, research tools are off.

Every format of a property's category, offline, with a page-by-page report:
```bash
python scripts/bench_brochure.py data/properties/brigade-citrine --all
```

The three reference properties (SOBHA Neopolis — Luxury, Brigade Citrine —
Mid-Tier, Mahindra Happinest Kalyan — Affordable) are included under
`data/properties/`. To rebuild them from the original photo folders:
```bash
python scripts/seed_reference_properties.py "../Property Photos"
```

## 6. VS Code

Open the folder, select the `.venv` interpreter, and use **Run and Debug**:
Streamlit, FastAPI, preflight, the bench, the brochure and blog CLIs, tests.

## 7. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_*_MODEL` | see `.env.example` | One model per role |
| `OPENROUTER_MODEL` | `openrouter/free` | Fallback text model |
| `IMAGE_GEN_PROVIDER` | `auto` | `auto` (Cloudflare, then OpenRouter) / `cloudflare` / `openrouter` / `huggingface` / `mock` / `none` |
| `PLAYWRIGHT_TIMEOUT_SECONDS` | 300 | Per render |
| `PLAYWRIGHT_BROWSER_CHANNEL` | — | `msedge` / `chrome` for a system browser |
| `DATABASE_PATH`, `DATA_DIR` | `./data/...` | Storage |
| `MLFLOW_ENABLED`, `MLFLOW_TRACKING_URI` | on, `sqlite:///mlflow.db` | Traces (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) |

## 8. Known limitations

- A brochure renders three to seven times in Chromium while it is measured
  and repaired; allow one to three minutes.
- **Without a Gemini key, photographs need a subject to be placed well.**
  The offline reader knows a picture's subject only from its upload slot or
  its file name ("pool.jpg", "master-bedroom.png"); "Picture7.jpg" is placed
  as a general photograph. Either name files descriptively, record subjects
  in `images/roles.json` (`"Picture7.jpg": "photo:landscape"`, as
  `scripts/seed_reference_properties.py` does), or add a free Gemini key so
  the Vision Agent reads them.
- **Copy without a key is composed, not authored.** The offline writer
  builds project-specific copy from the form's facts in each category's
  voice; it cannot know anything the form does not say. A free Gemini key
  lets a model rewrite it (every field passes the same fact guard).
- **Source images set the print quality.** Nothing is upscaled; a
  photograph or plan that prints below 150 dpi is reported in the record's
  notes and the evaluation (e.g. 700 px floor plans).
- Pages can sit outside the category's open-space range when a project has
  few photographs; the evaluation reports it page by page.
- Scanned PDFs among supporting documents yield no text (no OCR).
- Market figures in blogs come only from facts you supply or research that
  could be verified; nothing is estimated silently.
- English only.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Executable doesn't exist` (Playwright) | `python -m playwright install chromium`, or set `PLAYWRIGHT_BROWSER_CHANNEL=msedge` |
| `Playwright rendering timed out` | Fewer or smaller photos, or raise `PLAYWRIGHT_TIMEOUT_SECONDS` |
| Gemini `429` / quota | Free-tier limit; wait, or rely on the OpenRouter fallback |
| No AI images | Image provider not configured — see section 3 |
| "We couldn't finish this brochure" | Details are in the terminal log; most often a picture that cannot be opened |
