# Agentic Content Platform — Real Estate

Two solutions behind one Streamlit app, a CLI and a small REST API:

* **Brochure generation** — a print-ready PDF brochure in one of nine
  templates (Affordable, Mid-Tier, Luxury × A4 landscape, A4 portrait,
  16:9 digital), made from a short form and the property's pictures.
* **Blog generation** — a researched, website-style article delivered as a
  PDF (with Markdown and HTML alongside).

Both run as LangGraph workflows of specialised agents. Every model call has
a deterministic fallback, so both solutions produce a complete result with
no API key at all; the result records which steps fell back. Every result is
evaluated (an automated quality score) and can be approved or sent back by a
person.

Installation, keys and running: **[SETUP.md](SETUP.md)**. A demo runbook:
**[DEMO.md](DEMO.md)**.

---

## Using it

```bash
streamlit run streamlit_app.py
```

**Brochure.** One form, most important first — only the essentials are
required (*):

| # | Field | |
|---|---|---|
| 1 | Project name | * |
| 2 | City, locality | * city |
| 3 | Property type | |
| 4 | Category — Affordable, Mid-Tier or Luxury | * |
| 5 | Starting price / price range | |
| 6 | Configurations, carpet areas (and prices) — one row each | * one row |
| 7 | Key selling points | |
| 8 | Amenities and facilities | |
| 9 | Target audience | |
| 10 | Developer name and logo | |
| 11 | Phone / email / website and call to action | * phone or email |

Pictures go in their own slots — project photographs (at least one), floor
plans, master plan, location map, logo — so a drawing is always known to be
a drawing. Optional extras (tagline, possession, RERA number, project size,
nearby places) print only when given. Category and orientation choose the
template. There is no spreadsheet upload.

**Blog.** A topic (required) and, optionally, location, audience, tone,
primary keyword, length, key points, facts you can vouch for, project/brand
and a call to action, and images.

**The result** is shown in four tabs: **Final PDF**, **Evaluation**,
**Workflow**, **Human feedback**. Errors are explained in plain
words; technical detail goes to the log.

From the command line:

```bash
python scripts/ingest_property.py --source ./photos --property-id green-acres \
    --name "Green Acres" --city Pune --category Mid-Tier --config "2 BHK|680 sq ft" \
    --phone "+91 98765 43210" --floorplan plans/2bhk.png
python scripts/generate_brochure.py --property-id green-acres --template "Mid-Tier - Portrait"
python scripts/generate_blog.py --blog-id <id>
```

REST: `uvicorn src.api.main:app` — `POST /generate/brochure`
(`property_id`, optional `template_name` or `orientation`),
`POST /generate/blog`, `GET /generations/{id}`, `POST /generations/{id}/review`,
and the same for `/blogs`.

---

## The nine templates

Category × format. Print is **A4** (297 × 210 mm landscape, 210 × 297 mm
portrait) with a print-ready twin carrying 3 mm bleed and crop marks; the
**digital** variant is 16:9, exported as a PDF and as 1920 × 1080 px page
images (available from the API and scripts; the UI offers Landscape and
Portrait only). Every page is composed on a **12-column grid** with a ~4 mm gutter
and a 4 pt baseline, and authored at printed size (mm and pt).

| Category | Type (heading / body) | Scale (H1 · H2 · body) | Margins | Identity |
|---|---|---|---|---|
| Affordable | Plus Jakarta Sans ExtraBold / Inter | 24 · 14 · 9.6 pt | 12 mm | White and warm grey, deep slate, brand crimson, forest green, amber. Rounded 8 px cards, duotone icons, price and distance badges. Compact and value-led. |
| Mid-Tier | Playfair Display / Outfit | 28 · 15 · 10.2 pt | 16 / 18 mm | Soft sand and warm stone, espresso, Citrine bronze, sage, ochre. Square blocks, bronze hairlines and offset frames, numbered distinctions. Editorial and lifestyle-led. |
| Luxury | Cormorant Garamond / Jost Light | 36 · 17 · 10.6 pt | 22 / 25 mm | Alabaster and limestone, obsidian, Aegean navy, champagne-gold ornament. Sharp corners, ultra-fine gold rules, generous space. Architecture-led; no prices. |

The design tokens live in `src/agents/brochure/catalog.py` (type, margins,
gutter, radius, white-space target) and `palettes.py` (colours by job, every
type pairing contrast-checked to WCAG by a test). A `brand_color` in a
property.json replaces the accent only where it stays legible.

Positioning (`strategy.py`) decides what each category says and in what
order: Affordable leads with price, savings and the homes; Mid-Tier with the
project's story and distinctions (sustainability, privacy); Luxury with the
narrative, the architecture and the leisure, and never prints a price.

---

## How a brochure is made

```
form ─► property folder (property.json, images/, images/roles.json)
            │
property_intelligence   the fact ledger: every fact the brochure may state
vision                  what each picture shows (a subject given at upload wins;
                        offline: the upload slot, the pixels, the file name)
asset_prep              drawings trimmed of blank canvas, logos trimmed with their
                        transparency kept, the category's grade on photographs
visual_enrichment       optional supporting AI visuals for real gaps — never the
                        property, never the cover
creative_director       one of the category's contrast-checked palettes
art_director            the book planned from the material: pages, layouts, pictures
copywriter              every page written from the facts, in the category's voice
render_qa               rendered in Chromium, measured, repaired by whole sentences
critic                  the evaluation (a model's judgement, held to the measurements)
```

### The content model (facts.py)

Raw form fields are never poured into a layout. They are turned into the
structures a brochure is designed from: "1 Bed + 1 Toilet … 4 Bed + 4
Toilet" becomes **1, 2, 3 & 4 BHK Residences** with a schedule of
"3 BHK · 2 baths", "3 BHK + Study · 3 baths"; "1 BHK-A" becomes "1 BHK ·
Type A"; areas and prices are tidied ("1,611 – 1,915 sq ft",
"₹29.95 – 31.15 Lakh"); each selling point is split into a headline, a
detail and its key figure; amenity lines become zones ("Kids' Zone: …"),
never split inside a number ("1,200+ trees"); places are ordered by what a
buyer weighs first, with "Proximity" never printed as a distance.

### Pages from the material (planner.py, layouts.py)

* **Scoped to the project.** Only the pages the material supports, in the
  category's order: cover, overview, distinctions, sustainability figures
  (when the project states them), features, amenities, interiors,
  masterplan, residences, plans, location, closing. No contents page; no
  generic "How to book" page — the closing page carries the call to action
  and every way to act on it. A project with few photographs gets a short
  book, not a stretched one.
* **Each selling point on exactly one page** — the distinctions page, the
  sustainability page, the homes page (Affordable's payment benefits sit
  beside the prices) or the location page.
* **Pictures chosen by subject and shape.** Every photograph is used once;
  near-duplicates are detected perceptually and printed once. A photograph
  goes to the slot that crops it least, and a wide render switches a page to
  a band composition rather than being cut in half. Crops keep tower crowns
  and rooflines (the focal point is set high for architecture).
  Resolution is checked at the printed size and anything below 150 dpi is
  reported; nothing is upscaled.
* **Drawings are whole and large.** The masterplan has its own page; plan
  cards take the drawing's own proportions; where an Affordable project has
  unit plans and no interior photographs, the plans sit beside the price
  schedule. Locations without a map get a distance diagram drawn from the
  stated distances (labelled as not to scale).

### Words from the facts (copywriter.py, voice.py)

Offline, the copy is composed from **frames** in each category's voice
filled only with the project's own facts — its name, place, figures,
amenities, audience and selling points. Several frames exist for every job
and a stable hash of the project's name picks among them, so two projects in
one category do not read alike. Frames that imply a feature (a pool, gardens,
columns) are used only when the project has it. With a Gemini key (free tier)
one book-level call rewrites titles, standfirsts, running copy and card
headlines; each field passes the same guard or the offline field is kept.

The guard rejects any sentence that states a number not in the property's
own data, describes a picture, repeats a title or sentence used elsewhere,
or uses a word outside the category's register.

### Measured, then repaired (qa.py, brochure_audit.js, production.py)

Each render is audited in Chromium: copy past its box or off the page,
units/prices/phone numbers broken across lines, type below 6 pt, a typeface
that did not load (so the ₹ sign can never fall back to a missing glyph),
images that did not load, repeated text, effective resolution, and each
page's open space against its category's range. Repairs work by whole
sentences and optional elements — one sentence fewer or more, a standfirst,
a figure, a footnote — **never by changing type size**. A repair that does
not reduce an overflow is reverted and not tried again.

---

## How a blog is made

```
research ─► planner ─► visual ─► writer ─► fact_check ─► editor ─► seo ─► critic ─► render
                                     ▲                                     │
                                     └──── revise (targeted feedback) ─────┘
```

* **Research** looks things up (place, nearby amenities, journey distances,
  news, competing coverage) when tools are enabled; what could not be found
  is recorded, never guessed.
* **Planner** picks an article type (locality guide, buyer's guide,
  comparison, investment analysis, explainer, how-to, market update) and an
  outline suited to the reader.
* **Writer** writes to that outline; **fact check** matches every checkable
  claim against the brief and the research; **editor** normalises
  formatting (no collapsed bullets, no stray bold, no placeholder links) and
  removes unsupported claims; **SEO** writes title, meta description and
  slug from the real keyword.
* **Critic** scores five weighted parameters — Content quality (30%),
  Relevance (20%), Structure & readability (20%), Audience engagement (15%),
  SEO optimisation (15%) — each with a one-line reason, from measurements
  and, when available, a model judge. The blog score is their weighted sum.
  Parameters below target are sent back to the writer as specific
  instructions. Without a model judge, the parameters that need a reader
  are capped and say so. Names, weights and descriptions live in
  `configs/blog/evaluation.yaml`; the critic prompt, the scoring and the UI
  all read them from there.
* **Render** lays the article out as a website-style PDF: headings, a
  contents box, tables, callouts, a verification box, FAQ and call to
  action. Figures are labelled by source (supplied, researched, estimate).

---

## Evaluation and review

* **Workflow** — every agent step, its status and time.
* **Evaluation** — brochure: content, visual design, factual consistency,
  layout integrity, image integrity, average white space, and pass/attention
  for each measured check. Blog: the five parameters, each with its weight,
  score and one-line reason, and how the weighted overall was calculated.
  A model judge can never rate a brochure's design above
  what the measurements allow.
* **Human feedback** — approve or request a revision, with a comment;
  stored with the generation.

---

## Project structure

```
streamlit_app.py                 the UI
src/api/main.py                  REST API
src/workflows/brochure.py        brochure LangGraph workflow
src/workflows/blog.py            blog LangGraph workflow
src/agents/brochure/
    catalog.py                   the nine templates and the design system (sheets, type, margins)
    strategy.py                  category positioning
    facts.py                     the content model: raw form fields made print-ready
    planner.py                   which pages, which pictures in which slots
    layouts.py                   the page compositions on the 12-column grid
    copywriter.py, voice.py      copy from the facts in each category's voice, with the guard
    qa.py, production.py         render-measure-repair loop
    palettes.py, topics.py, image_plan.py, book.py
    property_intelligence.py, vision.py, vision_offline.py,
    visual_enrichment.py, creative_director.py, critic.py
src/agents/blog/                 blog agents
src/agents/image/                image gap analysis and generation
src/data/property_form.py        the brochure form -> property folder
src/data/ingestion.py            reading a property folder
src/rendering/                   HTML -> PDF (Playwright), audit script, icons, fonts
src/evaluation/                  scoring, brochure checks, blog rubric
templates/brochure/              base.html, macros, brochure.css, themes/<category>.css, pages/
templates/blog/                  article templates
configs/                         prompts, rubric, model settings
data/properties/                 demo properties: SOBHA Neopolis, Brigade Citrine,
                                 Mahindra Happinest Kalyan (scripts/seed_reference_properties.py), e128
scripts/                         CLI, ingest, preflight, bench
tests/                           unit, agent, brochure (incl. Chromium) and integration tests
```

---

## Development

```bash
python -m pytest -q                                   # all tests
python scripts/bench_brochure.py data/properties/brigade-citrine --all   # every format, offline
python scripts/preflight.py                           # is this machine ready?
```

`bench_brochure.py` renders a property in any template with no API key and
prints, page by page, the open space, each text box's fill and any defect the audit found.
