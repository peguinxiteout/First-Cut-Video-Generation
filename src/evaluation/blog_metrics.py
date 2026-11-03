"""Deterministic measurements of a blog post.

Every number the rubric (blog_rubric.py) scores, and every check the
Editor's validator reports, is computed here, once — paragraph length
distribution, sentence length, repetition, heading structure, list/table
presence, keyword placement and density, placeholders, untraceable
figures, bold density, local-entity coverage, stray geography, sales
language. Measurements are reproducible and cannot be argued with, which
is why they carry most of the score and an LLM's opinion carries less.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.agents.blog import article, blog_config, brief as brief_mod
from src.agents.blog.normaliser import (
    block_kind, blocks_of, find_placeholders, markup_leaks, split_sentences, words,
)
from src.data.schemas import BlogContent, BlogInput, FactCheckResult, ResearchPack

_STOP = frozenset("""a an the and or but of to in on at for with by from as is are was were be been it its
this that these those you your we our they their he she his her i me my not no so if then than
can will would should could may might do does did has have had about into over under up down out
more most less very just also only such there here what which who whom when where why how vs versus""".split())


@dataclass
class Measurement:
    values: dict[str, float] = field(default_factory=dict)
    details: dict[str, list[str]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.values.get(key, 0.0)

    def detail(self, key: str) -> list[str]:
        return self.details.get(key, [])


# ---------------------------------------------------------------------------
# Small reusable checks (also used by the Editor's validator)
# ---------------------------------------------------------------------------

def paragraphs(markdown: str) -> list[str]:
    return [b for b in blocks_of(markdown) if block_kind(b) == "text"]


def keyword_hits(text: str, keyword: str) -> int:
    """Occurrences of a keyword, allowing its words a little slack ("homes
    in Shela, Ahmedabad" counts for "homes in shela ahmedabad")."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", keyword.lower()) if t not in _STOP]
    if not tokens:
        return 0
    pattern = r"\b" + r"\W+(?:\w+\W+){0,2}?".join(re.escape(t) for t in tokens) + r"\b"
    return len(re.findall(pattern, text.lower()))


def keyword_density(text: str, keyword: str) -> float:
    """Percent of words taken up by the keyword's content words."""
    total = max(1, len(words(text)))
    tokens = max(1, len([t for t in re.findall(r"[a-z0-9]+", keyword.lower()) if t not in _STOP]))
    return round(100 * keyword_hits(text, keyword) * tokens / total, 2)


def keyword_present(text: str, keyword: str) -> bool:
    """All the keyword's content words appear (any order) — for titles and
    headings, where an exact phrase would force awkward wording."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", keyword.lower()) if t not in _STOP]
    lowered = text.lower()
    return bool(tokens) and all(re.search(r"\b" + re.escape(_stem(t)), lowered) for t in tokens)


def _stem(token: str) -> str:
    """Crude, but enough for "homes" to find "home" and "buyers" "buyer"."""
    for suffix in ("ies", "es", "s"):
        if len(token) > 4 and token.endswith(suffix):
            token = token[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    return token[:6]


def promotional_terms(text: str) -> list[str]:
    lowered = text.lower()
    return [t for t in blog_config.editorial().get("promotional_terms", [])
            if re.search(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", lowered)]


def prohibited_claims(text: str) -> list[str]:
    lowered = text.lower()
    return [t for t in blog_config.editorial().get("prohibited_claims", [])
            if re.search(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", lowered)]


def generic_headings(headings: list[str]) -> list[str]:
    generic = {g.lower() for g in blog_config.editorial().get("generic_headings", [])}
    generic |= {e["label"].lower() for e in blog_config.sections().values()}
    return [h for h in headings if h.strip().lower().rstrip(":?") in generic]


def allowed_geography(blog_input: BlogInput, research: Optional[ResearchPack] = None) -> str:
    """Every place name the post may mention: the brief's, and anything
    research found (nearby places, journeys, background, news)."""
    parts = [blog_input.topic, blog_input.location or "", *blog_input.local_facts, *blog_input.key_points,
             *blog_input.custom_sections, blog_input.project_name or "", blog_input.brand_description or ""]
    if research is not None:
        parts += [research.place_name, research.background, *(n.title for n in research.news),
                  *(p.name for p in research.nearby), *(j.destination for j in research.journeys),
                  *(r.title + " " + r.note for r in research.official_sources)]
    return " ".join(p for p in parts if p)


def stray_places(text: str, allowed_text: str) -> list[str]:
    """Known cities the post mentions that the brief and research never
    did — the "irrelevant geographic references" failure."""
    lowered, allowed = text.lower(), allowed_text.lower()
    found = []
    for city in blog_config.editorial().get("cities", {}):
        if re.search(r"\b" + re.escape(city) + r"\b", lowered) and not re.search(
                r"\b" + re.escape(city) + r"\b", allowed):
            # "navi mumbai" allowed covers "mumbai" only if mumbai itself is allowed.
            found.append(city)
    return sorted(set(found))


_DISTANCE = re.compile(r"(\d+(?:\.\d+)?)\s?(km|min)\b", re.I)
# Only landmarks a place has ONE of — two schools at different distances
# are not a contradiction; two distances to "the airport" are.
_LANDMARK = re.compile(r"\b(airport|railway station|ring road|city cent(?:re|er)|bus stand|bus terminus)\b", re.I)


def _units_of_text(text: str) -> list[str]:
    """Sentences of running text, and each table row on its own."""
    out: list[str] = []
    for block in blocks_of(text):
        if block_kind(block) == "table":
            out.extend(row.replace("|", " ") for row in block.split("\n")[2:])
        elif block_kind(block) == "list":
            out.extend(article.plain(item) for item in block.split("\n"))
        else:
            out.extend(split_sentences(article.plain(block).replace("\n", " ")))
    return out


def inconsistent_figures(text: str) -> list[str]:
    """The same landmark given different distances or times in different
    places ("18 km to the airport" ... "22 km from the airport"). Each
    figure is paired with the NEAREST landmark in its sentence or row."""
    seen: dict[tuple[str, str], set[str]] = {}
    for unit_text in _units_of_text(text):
        landmarks = [(m.start(), m.group(1).lower().rstrip("s")) for m in _LANDMARK.finditer(unit_text)]
        if not landmarks:
            continue
        for fig in _DISTANCE.finditer(unit_text):
            pos = fig.start()
            dist, landmark = min((abs(pos - p), name) for p, name in landmarks)
            if dist > 60:
                continue
            value = fig.group(1)
            value = value.rstrip("0").rstrip(".") if "." in value else value
            seen.setdefault((landmark, fig.group(2).lower()), set()).add(value)
    return [f"{landmark}: " + " vs ".join(f"{v} {unit}" for v in sorted(values))
            for (landmark, unit), values in seen.items() if len(values) > 1]


def implausible_units(text: str) -> list[str]:
    """Units that cannot be right: a 900 km 'nearby' metro, a price per
    sq ft in crore, a percentage over 100."""
    out = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s?km\b", text, re.I):
        if float(m.group(1).replace(",", "")) > 300:
            out.append(m.group(0))
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s?(?:min|minutes)\b", text, re.I):
        if float(m.group(1).replace(",", "")) > 600:
            out.append(m.group(0))
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s?%", text):
        if float(m.group(1).replace(",", "")) > 100:
            out.append(m.group(0))
    for m in re.finditer(r"(?:crore|cr)\s+per\s+sq\.?\s?(?:ft|m)", text, re.I):
        out.append(m.group(0))
    for m in re.finditer(r"₹\s?(\d[\d,]*)\s+per\s+sq\s?ft", text):
        value = float(m.group(1).replace(",", ""))
        if value < 500 or value > 200000:
            out.append(m.group(0))
    return out


def ngram_repetition(text: str, n: int = 4) -> float:
    """Share of word n-grams that occur three or more times."""
    tokens = [t for t in re.findall(r"[a-z0-9₹]+", article.plain(text).lower())]
    grams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    grams = [g for g in grams if sum(1 for w in g.split() if w not in _STOP) >= 2]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c >= 3)
    return repeated / len(grams)


def near_duplicate(a: str, b: str, threshold: float = 0.8) -> bool:
    ta = set(_trigrams(a))
    tb = set(_trigrams(b))
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / max(1, min(len(ta), len(tb))) >= threshold


def _trigrams(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)]


def duplicate_sentences(sentences: list[str]) -> list[str]:
    dupes, kept = [], []
    for s in sentences:
        if len(s.split()) < 6:
            continue
        if any(near_duplicate(s, k) for k in kept):
            dupes.append(s)
        else:
            kept.append(s)
    return dupes


# Hooks that read as filler: a definition, a cliché, a throat-clear.
_GENERIC_HOOK = re.compile(
    r"^(?:in today's|in the (?:ever|fast)|when it comes to|are you (?:looking|thinking|planning)|"
    r"real estate is|buying a (?:home|house|property) is (?:a|one)|investing in real estate|"
    r"welcome to|in this (?:blog|article|post)|the real estate (?:market|sector) (?:is|has)|"
    r"[a-z ]{0,30} is (?:a|an) (?:popular|emerging|prime|thriving|vibrant))", re.I)

_DECISIVE = re.compile(r"\b(?:suits?|best for|choose|go for|worth (?:it|considering)|our verdict|"
                       r"the verdict|if you|we recommend|recommend|look elsewhere|makes sense|"
                       r"a strong fit|a weaker fit|skip)\b", re.I)


# ---------------------------------------------------------------------------
# The full measurement
# ---------------------------------------------------------------------------

def measure(content: BlogContent, blog_input: BlogInput, research: Optional[ResearchPack] = None,
            fact_check: Optional[FactCheckResult] = None, evidence: Optional[dict] = None,
            outline_text: str = "") -> Measurement:
    from src.agents.blog import fact_check as fact_check_mod

    limits = blog_config.limits()
    m = Measurement()
    v, d = m.values, m.details
    ctx = brief_mod.context(blog_input)
    full = article.full_markdown(content)
    body = article.body_markdown(content)
    plain_body = article.plain(body)
    all_words = words(article.plain(full))
    v["word_count"] = len(all_words)
    v["section_count"] = len(content.sections)

    # --- paragraphs & sentences ------------------------------------------
    paras = paragraphs(body)
    lengths = sorted(len(words(p)) for p in paras) or [0]
    max_words = limits.get("paragraph_max_words", 70)
    long_paras = [p for p in paras if len(words(p)) > max_words
                  or len(split_sentences(p)) > limits.get("paragraph_max_sentences", 4)]
    v["paragraphs"] = len(paras)
    v["avg_paragraph_words"] = round(sum(lengths) / max(1, len(paras)), 1)
    v["p90_paragraph_words"] = lengths[min(len(lengths) - 1, int(0.9 * len(lengths)))]
    v["long_paragraphs"] = len(long_paras)
    v["long_paragraph_share"] = round(len(long_paras) / max(1, len(paras)), 3)
    d["long_paragraphs"] = [p[:80] for p in long_paras]

    sentences = [s for p in paras for s in split_sentences(p)]
    sentences += [article.plain(item) for b in blocks_of(body) if block_kind(b) == "list"
                  for item in b.split("\n") if len(item.split()) > 8]
    sent_lengths = [len(words(s)) for s in sentences] or [0]
    long_limit = limits.get("sentence_long_words", 30)
    v["sentences"] = len(sentences)
    v["avg_sentence_words"] = round(sum(sent_lengths) / max(1, len(sentences)), 1)
    v["long_sentence_share"] = round(sum(1 for n in sent_lengths if n > long_limit) / max(1, len(sentences)), 3)
    d["long_sentences"] = [s[:90] for s in sentences if len(words(s)) > long_limit]

    # --- repetition --------------------------------------------------------
    dupes = duplicate_sentences(sentences)
    v["duplicate_sentences"] = len(dupes)
    d["duplicate_sentences"] = [s[:90] for s in dupes]
    running = "\n\n".join(b for b in blocks_of(body) if block_kind(b) in ("text", "list", "quote"))
    v["repeated_ngram_share"] = round(ngram_repetition(running), 3)
    openers = Counter(" ".join(s.lower().split()[:2]) for s in sentences if len(s.split()) > 3)
    top_opener = openers.most_common(1)[0] if openers else ("", 0)
    v["repeated_opener_share"] = round(top_opener[1] / max(1, len(sentences)), 3)
    d["repeated_opener"] = [top_opener[0]] if top_opener[1] >= 4 else []

    # --- structure ----------------------------------------------------------
    blocks = [b for s in content.sections for b in blocks_of(s.body_markdown)]
    v["lists"] = sum(1 for b in blocks if block_kind(b) == "list")
    v["tables"] = sum(1 for b in blocks if block_kind(b) == "table")
    v["checklist_items"] = sum(len(re.findall(r"(?m)^\s*[-*]\s+\[ \]", b)) for b in blocks)
    v["ordered_lists"] = sum(1 for b in blocks if re.match(r"^\s*\d+[.)]\s", b))
    v["callouts"] = sum(1 for b in blocks if block_kind(b) == "quote")
    v["subheadings"] = sum(1 for b in blocks if block_kind(b) == "heading")
    v["faqs"] = len(getattr(content, "faqs", []) or [])
    v["takeaways"] = len(getattr(content, "key_takeaways", []) or [])
    headings = [s.heading for s in content.sections]
    generic = generic_headings(headings)
    v["generic_headings"] = len(generic)
    d["generic_headings"] = generic
    v["question_headings"] = sum(1 for h in headings if h.strip().endswith("?"))
    v["avg_heading_words"] = round(sum(len(h.split()) for h in headings) / max(1, len(headings)), 1)
    v["long_headings"] = sum(1 for h in headings if len(h.split()) > limits.get("heading_max_words", 12))
    v["duplicate_headings"] = len(headings) - len({h.lower() for h in headings})
    v["empty_sections"] = sum(1 for s in content.sections if len(words(s.body_markdown)) < 20)
    v["has_intro"] = float(len(words(content.intro_markdown)) >= 20)
    v["has_conclusion"] = float(len(words(content.conclusion_markdown)) >= 20)
    v["conclusion_decisive"] = float(bool(_DECISIVE.search(content.conclusion_markdown or "")))
    first_sentence = (split_sentences(article.plain(content.intro_markdown)) or [""])[0]
    v["hook_generic"] = float(bool(_GENERIC_HOOK.search(first_sentence.strip())) or not first_sentence)
    from src.evaluation.blog_deterministic import uncovered_sections
    missing = uncovered_sections(full, blog_input, outline_text)
    v["missing_sections"] = len(missing)
    d["missing_sections"] = missing

    # --- formatting hygiene -------------------------------------------------
    rendered_like = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", plain_body)
    rendered_like = re.sub(r"\*\*([^*]+)\*\*", r"\1", rendered_like)
    leaks = markup_leaks(rendered_like)
    v["markup_leaks"] = len(leaks)
    d["markup_leaks"] = leaks
    no_leadins = re.sub(r"(?m)^(\s*(?:[-*+]|\d+[.)])\s+(?:\[ \]\s+)?)\*\*[^*]{1,60}\*\*", r"\1", body)
    bold_spans = re.findall(r"\*\*([^*]+)\*\*", no_leadins)
    bold_words = sum(len(b.split()) for b in bold_spans)
    v["bold_spans"] = len(bold_spans)
    v["bold_share"] = round(bold_words / max(1, len(words(plain_body))), 3)
    placeholders = find_placeholders(full)
    v["placeholders"] = len(placeholders)
    d["placeholders"] = placeholders[:6]

    # --- accuracy -----------------------------------------------------------
    evidence = evidence or fact_check_mod.evidence_for(blog_input, research)
    untraceable = fact_check_mod.untraceable_figures(article.plain(full), evidence)
    v["untraceable_figures"] = len(untraceable)
    d["untraceable_figures"] = sorted(set(untraceable))[:8]
    v["unsupported_claims"] = len(fact_check.unsupported) if fact_check else 0
    d["unsupported_claims"] = list(fact_check.unsupported[:5]) if fact_check else []
    inconsistent = inconsistent_figures(full)
    v["inconsistent_figures"] = len(inconsistent)
    d["inconsistent_figures"] = inconsistent
    implausible = implausible_units(full)
    v["implausible_units"] = len(implausible)
    d["implausible_units"] = implausible
    figures_total = len(fact_check_mod.figures_in(article.plain(full)))
    v["figures"] = figures_total
    facts = [f for f in blog_input.local_facts if f.strip()]
    used = 0
    for fact in facts:
        nums = fact_check_mod.numbers_in(fact)
        key_words = [w for w in re.findall(r"[a-z]{5,}", fact.lower()) if w not in _STOP][:3]
        if (nums and nums <= fact_check_mod.numbers_in(full)) or (not nums and key_words
                                                                   and all(w in full.lower() for w in key_words)):
            used += 1
    v["brief_facts"] = len(facts)
    v["brief_fact_coverage"] = round(used / len(facts), 3) if facts else 1.0
    research_facts = bool(research and (research.nearby or research.journeys or research.news
                                         or research.official_sources))
    v["has_specifics"] = float(bool(facts) or research_facts)
    v["unverified_admissions"] = len(re.findall(r"\b(?:could not|couldn't|unable to) (?:verify|find)\b",
                                                plain_body, re.I))
    v["verification_language"] = float(bool(re.search(
        r"\b(?:verify|check|confirm|according to|source|as of|estimate|announced|measured|"
        r"supplied by|registered)\b", plain_body, re.I)))

    # --- local relevance -----------------------------------------------------
    loc = ctx.location
    if loc.known:
        names = [n for n in {loc.short, loc.city, *ctx.options} if n]
        hits = sum(keyword_hits(plain_body, n) for n in names)
        v["location_mentions"] = hits
        v["location_mention_rate"] = round(100 * hits / max(1, len(words(plain_body))), 2)
    else:
        v["location_mentions"] = 0
        v["location_mention_rate"] = 0
    research_names = [p.name for p in (research.nearby if research else [])]
    v["research_entities_used"] = sum(1 for n in research_names if n.lower() in full.lower())
    stray = stray_places(article.plain(full), allowed_geography(blog_input, research))
    v["stray_places"] = len(stray)
    d["stray_places"] = stray

    # --- reader focus, tone, marketing ----------------------------------------
    you = len(re.findall(r"\b(?:you|your|you're|yours)\b", plain_body, re.I))
    v["you_per_100_words"] = round(100 * you / max(1, len(words(plain_body))), 2)
    audience_words = [w for w in re.findall(r"[a-z]{4,}", ctx.audience_label.lower()) if w not in _STOP]
    v["audience_mentions"] = sum(len(re.findall(r"\b" + re.escape(w[:6]), full.lower())) for w in audience_words)
    promo = promotional_terms(full)
    v["promotional_terms"] = len(promo)
    d["promotional_terms"] = promo
    prohibited = prohibited_claims(full)
    v["prohibited_claims"] = len(prohibited)
    d["prohibited_claims"] = prohibited
    from src.evaluation.blog_deterministic import banned_phrases_used
    banned = banned_phrases_used(full)
    v["stock_phrases"] = len(banned)
    d["stock_phrases"] = banned
    brand = (getattr(blog_input, "brand_name", "") or "").strip()
    v["brand_mentions"] = len(re.findall(re.escape(brand), full, re.I)) if brand else 0
    competitors = blog_config.editorial().get("competitor_domains", [])
    cited = [r.url for r in getattr(content, "references", []) or []]
    links = re.findall(r"\]\((https?://[^)]+)\)", full) + cited
    comp = [u for u in links if any(c in u.lower() for c in competitors)]
    comp += [c for c in competitors if "." in c and re.search(re.escape(c), plain_body, re.I)
             or "." not in c and re.search(r"\b" + re.escape(c) + r"\b", plain_body, re.I)]
    v["competitor_mentions"] = len(comp)
    d["competitor_mentions"] = comp
    v["references"] = len(getattr(content, "references", []) or [])
    v["official_references"] = sum(1 for r in getattr(content, "references", []) or [] if r.kind == "official")

    # --- CTA ------------------------------------------------------------------
    v["cta_present"] = float(len(words(content.cta_markdown or "")) >= 6)
    cta_text = (content.cta_markdown or "") + " " + (content.cta_heading or "")
    v["cta_placeholder"] = float(bool(find_placeholders(cta_text)))

    # --- SEO ------------------------------------------------------------------
    keyword = (getattr(content, "primary_keyword", "") or ctx.primary_keyword or "").strip()
    first_100 = " ".join(words(article.plain(content.intro_markdown))[:100])
    v["keyword_in_title"] = float(keyword_present(content.title, keyword)) if keyword else 0
    v["keyword_in_intro"] = float(keyword_present(first_100, keyword)) if keyword else 0
    v["keyword_in_heading"] = float(any(keyword_present(h, keyword) for h in headings)) if keyword else 0
    meta_title = getattr(content, "meta_title", "") or ""
    v["keyword_in_meta"] = float(keyword_present(content.meta_description + " " + meta_title, keyword)) if keyword else 0
    v["keyword_density"] = keyword_density(article.plain(full), keyword) if keyword else 0
    secondaries = list(getattr(content, "secondary_keywords", []) or []) or list(blog_input.secondary_keywords)
    v["secondary_coverage"] = round(sum(1 for k in secondaries if keyword_present(full, k)) / len(secondaries), 3) \
        if secondaries else 1.0
    v["meta_title_length"] = len(meta_title)
    v["meta_description_length"] = len(content.meta_description or "")
    slug = getattr(content, "slug", "") or ""
    v["slug_ok"] = float(bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+){1,9}", slug)) and len(slug) <= 75)
    v["tags"] = len(getattr(content, "tags", []) or [])
    v["seo_keywords"] = len(content.seo_keywords or [])
    return m
