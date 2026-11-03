# Demo Runbook

## The day before

```bash
python scripts/preflight.py
```
Fix every `FAIL`; `WARN` lines are degraded features (no key, no images),
not blockers. Then make the exact brochure and blog you plan to show and
keep the PDFs — if anything misbehaves live, show the saved one.

Have ready: 8–15 photographs (2,000 px or more on the long edge), the floor
plans and master plan as separate files, a logo, the configurations with
carpet areas, the amenities, a few selling points, and a phone or email.

## Timings

| Step | Typical |
|---|---|
| Property intelligence, vision | 30–90 s with a key; seconds offline |
| Planning and copy | a few seconds |
| Render, measure, repair (3–6 renders) | 1–3 min |
| Critic | 10–30 s |

Start the generation, then talk over it.

## A 12-minute demo

1. **The form (2 min).** Category and format (A4 landscape or A4 portrait)
   choose the template. Only six things are required; everything else is used only
   when given. Pictures go in their own slots — a floor plan is a floor plan.
2. **Generate (talk over it, 3 min).** Explain the order: pictures are
   placed first, every one whole and once; then the copy is written to the
   room left; then the book is rendered, measured and repaired.
3. **Final PDF (3 min).** Walk the book: the cover, the overview with its
   photograph, amenities beside the photographs that show them, the homes
   with their plans, how to book. Then switch category on the same property
   to show the positioning change: the affordable book leads with price,
   the luxury one never prints it.
4. **Evaluation and Workflow (2 min).** The measured checks (no overflow,
   no type over photographs, no broken units, fonts and ₹ loaded, open space
   per page) and every agent step with its time.
5. **Human feedback (1 min).** Approve or request a revision.
6. **Blog (1 min).** A topic and two facts; the website-style PDF, figures
   labelled by source.

## Do not demo

- **That AI images show the property.** They never do: supporting visuals
  are mood and context only, never the cover, and never captioned as the
  building. Show the uploaded photographs as the property.
- **That the system will invent a figure.** It will not: a price, area or
  distance that is not in the form or the supplied facts is rejected from
  the copy. If a figure is missing, add it to the form rather than asking
  the system to "fill it in".
- **A model-quality claim without a key.** Without `GEMINI_API_KEY` the
  copy comes from the curated copy bank and the evaluation is measured
  only; the Evaluation tab says so ("judged by: measured").

## Fallbacks

| Live problem | Say / do |
|---|---|
| No key or quota spent | The workflow falls back step by step and still delivers; the evaluation says it was measured only |
| Rendering slow | It renders several times to measure itself; show the saved PDF meanwhile |
| Something fails | Show the saved PDF; the details are in the terminal log |
