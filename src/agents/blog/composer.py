"""The deterministic writer: a complete, structured article from the brief
and the research, with no model.

This is not a placeholder. It is what runs whenever the model is
unreachable — no key, quota exhausted, provider outage — which on a free
tier is often, and it is what runs in this sandbox. So it has to produce
a post someone would publish:

  * it follows the archetype's section plan (a locality guide, a buyer's
    guide, a comparison... each shaped for its search intent);
  * every section is built around the brief's own facts and the research,
    each fact stated with its status and source, and each tied to WHY it
    matters for this reader (configs/blog/archetypes.yaml `audiences`);
  * the formats are the planned ones — fact tables, checklists, numbered
    steps, a decision table — with short paragraphs throughout;
  * it NEVER writes a number of its own. Every figure in the output is a
    fact's own text. Generic guidance is written without figures (no
    invented percentages, distances or timelines), and where something
    could not be verified the post says so and tells the reader where to
    check instead.

Each slot generator returns markdown; the Editor normalises and validates
it exactly like model output, so both paths meet the same bar.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from src.agents.blog import archetypes as archetypes_mod, blog_config, brief as brief_mod, facts as facts_mod
from src.agents.blog.facts import CATEGORY_TITLES, Fact
from src.data.schemas import (
    BlogContent, BlogFAQ, BlogImageAsset, BlogInput, BlogOutline, BlogReference, BlogSection,
    BlogSectionPlan, ResearchPack,
)


def regulator(ctx: brief_mod.BriefContext) -> tuple[str, str]:
    """(name, url) of the body to verify registration with. A URL only
    when the curated registry has one — never a guessed address."""
    portals = blog_config.editorial().get("rera_portals", {}) or {}
    state = ctx.location.state
    if state in portals:
        return portals[state]["name"], portals[state].get("url", "")
    if state and ctx.location.india:
        return f"the {state} RERA portal", ""
    if ctx.location.india:
        return "your state's RERA portal", ""
    return "the local property regulator's public register", ""


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


_COMMON_OPENERS = frozenset("""a an the about around within listed most many under over near just only
approximately roughly nearly expected starting from is are some all both each new existing upcoming proposed
planned announced more less several few main its their our your this that these those under-construction
ready""".split())


def _lc(text: str) -> str:
    """Lower-case the first letter only when the first word is an ordinary
    word ("Several schools", "Within 3 km") — never a name ("Shela Lake
    Garden", "Ahmedabad Metro") or an acronym."""
    if not text:
        return text
    first = text.split()[0].strip(",.:;").lower()
    if first in _COMMON_OPENERS:
        return text[:1].lower() + text[1:]
    return text


class Draft:
    """One article being composed. Holds the brief-derived context and
    remembers which facts and sentences have been used, so the post does
    not repeat itself — repetition was a recurring evaluation finding."""

    def __init__(self, blog_input: BlogInput, research: Optional[ResearchPack], outline: BlogOutline,
                 images: list[BlogImageAsset]):
        self.bi = blog_input
        self.research = research
        self.outline = outline
        self.images = images
        self.ctx = brief_mod.context(blog_input)
        self.archetype = outline.archetype or archetypes_mod.classify(blog_input, self.ctx)
        self.facts = facts_mod.ordered(facts_mod.collect(blog_input, research), self.priorities)
        self.cat = facts_mod.by_category(self.facts)
        self.used_facts: set[int] = set()
        self.reg_name, self.reg_url = regulator(self.ctx)
        self.verdict_lines: list[str] = []
        self.takeaways: dict[str, str] = {}
        self.slots = {p.slot for p in outline.sections}
        self.used_why: set[str] = set()
        self.publisher = (blog_input.brand_name or "").strip() or "the publisher"

    # --- context shortcuts -------------------------------------------------
    @property
    def priorities(self) -> list[str]:
        return list(brief_mod.audience_profile(self.bi).get("priorities", []))

    @property
    def place(self) -> str:
        return self.ctx.location.short or "the area"

    @property
    def place_known(self) -> bool:
        return self.ctx.location.known

    @property
    def audience(self) -> str:
        return self.ctx.audience_label

    @property
    def india(self) -> bool:
        return self.ctx.location.india

    def why(self, category: str) -> str:
        return self.ctx.audience.get("why", {}).get(category, "")

    def why_once(self, category: str) -> str:
        """The audience reason for a category, the first time only — the
        same sentence twice reads as padding."""
        text = self.why(category)
        if not text or text in self.used_why:
            return ""
        self.used_why.add(text)
        return text

    @property
    def regulator_md(self) -> str:
        return f"[{self.reg_name}]({self.reg_url})" if self.reg_url else self.reg_name

    def facts_in(self, *categories: str, unused: bool = False, limit: int = 6) -> list[Fact]:
        out = []
        for c in categories:
            for f in self.cat.get(c, []):
                if unused and id(f) in self.used_facts:
                    continue
                out.append(f)
        return out[:limit]

    def use(self, fact: Fact) -> Fact:
        self.used_facts.add(id(fact))
        return fact

    def has(self, *categories: str) -> bool:
        return any(self.cat.get(c) for c in categories)

    @staticmethod
    def bullet(fact: Fact, why: str = "") -> str:
        if fact.status == "Reported":            # a headline: announced, attributed, not asserted
            line = f"- **In the news:** {fact.label} ({fact.source}, {fact.value})."
        elif fact.origin == "research":
            line = f"- **{fact.label}:** {fact.value.rstrip('.')} ({fact.status.lower()}: {fact.source})."
        else:
            line = f"- **{fact.label}:** {fact.value.rstrip('.')}."
        if why:
            line += f" {why}"
        return line

    def status_cell(self, fact: Fact) -> str:
        if fact.origin == "brief":
            return "Announced (supplied)" if fact.status == "Announced" else "Supplied"
        return f"{fact.status} — {fact.source}"

    def tabled_elsewhere(self, fact: Fact) -> bool:
        """True when a dedicated table section will show this fact."""
        return (fact.category == "connectivity" and "connectivity" in self.slots) or \
            (fact.category == "price" and "pricing" in self.slots)

    def infra_names(self) -> list[str]:
        return [f.label for f in self.cat.get("infrastructure", [])]

    # =======================================================================
    # Intro, takeaways, conclusion, FAQ, CTA
    # =======================================================================

    def intro(self) -> str:
        a = self.archetype
        place, aud = self.place, self.audience
        kw = self.ctx.keyword_display
        concern = self.ctx.audience.get("concern", "")
        infra = self.has("infrastructure")
        if a == "locality_guide":
            trade = ("what works in the area today against infrastructure that is still being built"
                     if infra else "price and space against commute and everyday convenience")
            first = (f"If you are considering {kw}, the real question is not whether {place} is “good”. "
                     f"It is whether it fits the way you live — and the trade-off is usually {trade}.")
        elif a == "comparison":
            opts = " and ".join(self.ctx.options[:2]) or "these options"
            first = (f"Choosing between {opts} is rarely about which is better in general. "
                     f"It is about which one fits your budget, your commute and your timeline.")
        elif a == "investment_analysis":
            first = (f"An investment in {place} stands or falls on three questions: what drives demand, "
                     "what it costs to hold, and how easily you can sell when you want to.")
        elif a == "explainer":
            first = (f"{self.ctx.subject} sounds like paperwork, but for {aud} it decides real money: "
                     "what you are actually paying for and what you can hold a developer to.")
        elif a == "how_to":
            first = (f"{self.ctx.subject} is mostly sequencing and paperwork. Get the order wrong and you "
                     "pay for it in delays, repeat visits and avoidable fees.")
        elif a == "market_update":
            first = (f"Headlines about {place} property move faster than the market itself. This update "
                     "separates what has been announced from what has actually changed.")
        else:
            first = (f"For {aud}, the hardest part of {_lc(self.ctx.subject)} is {concern}. "
                     "Most costly mistakes happen before booking, not after.")
        second = ("This guide sets out what is known, labels where each fact comes from, "
                  "and flags what you should verify yourself before you commit.")
        if self.facts:
            supplied = sum(1 for f in self.facts if f.origin == "brief")
            measured = len(self.facts) - supplied
            basis = []
            if supplied:
                basis.append("figures supplied with this brief")
            if measured:
                basis.append("public data such as maps and news reports")
            second = (f"Below, you will find the facts that matter for {aud}, drawn from "
                      f"{' and '.join(basis)} — each labelled by source — followed by the trade-offs "
                      "and a checklist to use before you book.")
        else:
            second = ("We could not verify local figures for this article, so instead of numbers to rely on, "
                      "it gives you a way to judge the decision and the questions to ask before you book.")
        return f"{first}\n\n{second}"

    def key_takeaways(self) -> list[str]:
        """Answer first, then the strongest point, the main trade-off, the
        money rule and the one check — in that order."""
        order = ["verdict", "strength", "tradeoff", "money", "returns", "checklist"]
        out = [v for k, v in self.takeaways.items() if k.startswith("option:")]
        if out:
            out.append("Choose on your own commute and all-in budget, not on an area's reputation.")
        out += [self.takeaways[k] for k in order if self.takeaways.get(k)]
        out.append(f"Verify registration and approvals on {self.reg_name} before paying any booking amount."
                   if self.india else "Verify registration and approvals with the regulator before paying anything.")
        return list(dict.fromkeys(t for t in out if t))[:5]

    def conclusion(self) -> tuple[str, str]:
        a, place, aud = self.archetype, self.place, self.audience
        phrases = {"social": "schools close by", "connectivity": "good road links",
                   "amenity": "green space nearby", "price": "a clear price range to compare",
                   "market": "proven rental demand"}
        seen_cats = [c for c in self.priorities if self.cat.get(c) and c in phrases]
        strengths = [phrases[c] for c in seen_cats[:2]]
        infra = self.infra_names()
        if a == "comparison" and self.ctx.options:
            heading = "The verdict: how to choose"
            body = [f"Neither {self.ctx.options[0]} nor {self.ctx.options[1]} wins for everyone. "
                    "Choose the one that shortens the commute you make most often and fits your all-in budget, "
                    "then let project quality decide between shortlisted homes.",
                    "If the two are close, the safer pick is the one with the clearer paperwork and the "
                    "better delivery record, not the one with the bigger promises."]
        elif a == "investment_analysis":
            heading = f"The verdict: is {place} worth your capital?" if self.place_known else "The verdict"
            body = [f"{_cap(place)} makes sense for investors who can hold through a full cycle and who have "
                    "checked rents and resale prices independently. It is a weaker fit if the plan only works "
                    "when announced projects arrive on time.",
                    "Start with the demand you can verify today; treat everything else as upside."]
        elif a in ("explainer", "how_to"):
            heading = "The bottom line"
            body = [f"For {aud}, the practical rule is simple: rely on the official record, get every "
                    "commitment in writing, and check before you pay rather than after.",
                    "If you do only one thing, check your agreement against the official record before you pay. "
                    "Where they do not match, stop and get it corrected in writing before the next step."]
        else:
            heading = f"The verdict: is {place} right for you?" if self.place_known else "The verdict"
            good = " and ".join(strengths) if strengths else "the everyday basics they need"
            body = [f"{_cap(place)} suits {aud} who want {good}"
                    + (f", and who can wait for {infra[0]} to open" if infra else "") + ". "
                    "It is a weaker fit if you need every convenience on day one or a quick, certain resale.",
                    "Shortlist two or three projects, run the checklist above on each, and let the paperwork — "
                    "not the brochure — make the final call."]
        self.verdict_lines = body
        if a not in ("explainer", "how_to", "comparison"):
            first = body[0].split(". ")[0].rstrip(".")
            self.takeaways["verdict"] = first + "."
        return heading, "\n\n".join(body)

    def faqs(self) -> list[BlogFAQ]:
        a, place, aud = self.archetype, self.place, self.audience
        reg = self.regulator_md
        out: list[BlogFAQ] = []
        if a in ("locality_guide", "market_update", "investment_analysis") and self.place_known:
            out.append(BlogFAQ(
                question=f"Is {place} a good choice for {aud}?",
                answer_markdown=("It can be, if its strengths match your priorities. Weigh the trade-offs above "
                                 "against your commute and budget, and visit at peak hour before deciding.")))
        if a == "comparison" and self.ctx.options:
            out.append(BlogFAQ(
                question=f"Which is better, {self.ctx.options[0]} or {self.ctx.options[1]}?",
                answer_markdown=("The better option is the one that fits your commute, budget and timeline. "
                                 "Use the decision guide above, then compare specific projects on paperwork "
                                 "and all-in cost.")))
        if self.india:
            out.append(BlogFAQ(
                question="How do I check whether a project is RERA-registered?",
                answer_markdown=(f"Search for the project name or registration number on {reg}. Check that the "
                                 "registered details — developer, carpet area and completion date — match your "
                                 "agreement.")))
        conn = self.cat.get("connectivity", [])
        if conn and a not in ("explainer", "comparison", "how_to"):
            f = conn[0]
            out.append(BlogFAQ(
                question=f"How well connected is {place}?" if self.place_known else "How should I judge connectivity?",
                answer_markdown=(f"One reference point: {_lc(f.label)} — {_lc(f.value.rstrip('.'))}. "
                                 "Distances on a map are a starting point; time the trips you will make "
                                 "every day before you decide.")))
        if a in ("buyers_guide", "explainer", "how_to"):
            out.append(BlogFAQ(
                question="What costs should I budget for beyond the listed price?",
                answer_markdown=("Ask for an all-inclusive cost sheet. It should cover taxes, registration, "
                                 "parking, maintenance deposits and any club or location charges.")))
        if a == "investment_analysis":
            out.append(BlogFAQ(
                question="What rental yield can I expect?",
                answer_markdown=("Only verified local rents can answer that. Ask agents for actual agreed rents on "
                                 "comparable homes, not asking rents, and calculate the yield on your all-in cost.")))
        return out[:4]

    def cta(self) -> tuple[str, str]:
        return default_cta(self.bi, self.archetype, self.ctx)

    # =======================================================================
    # Slot generators
    # =======================================================================

    def at_a_glance(self, plan: BlogSectionPlan) -> str:
        facts = [f for f in self.facts if not self.tabled_elsewhere(f)][:10]
        if len(facts) < 2:
            facts = self.facts[:10]
        if not facts:
            rows = [("Registration and approvals", self.regulator_md),
                    ("Commute times", "Your own trip at peak hour"),
                    ("Prices", "Recent registered sales and builder cost sheets"),
                    ("Upcoming infrastructure", "The implementing authority's official updates")]
            table = "| What to find out | Where to check |\n|---|---|\n" + "\n".join(
                f"| {a} | {b} |" for a, b in rows)
            return ("Here is where to find each answer yourself, rather than relying on estimates.\n\n" + table)
        lines = ["| What | Detail | Source |", "|---|---|---|"]
        for fact in facts:
            lines.append(f"| {fact.label} | {fact.value} | {self.status_cell(fact)} |")
        supplied = any(f.origin == "brief" for f in facts)
        lead = (f"These are the facts that matter most for {self.audience}, ordered by what usually decides "
                "the purchase."
                + (f" Figures marked “Supplied” come from {self.publisher}; others name their public source."
                   if supplied else " Each row names its public source."))
        note = ""
        if self.has("infrastructure"):
            note = ("\n\n> **Note:** Items marked as announced or under construction are plans, not operating "
                    "facilities. Price in what exists today.")
        elif any(f.origin == "brief" for f in facts):
            note = ("\n\n> **Verify:** Supplied figures are a starting point. Confirm them against the project's "
                    "registration details and your cost sheet before you pay.")
        return f"{lead}\n\n" + "\n".join(lines) + note

    def strengths(self, plan: BlogSectionPlan) -> str:
        facts, labels = [], set()
        for f in self.facts:
            key = re.sub(r"[^a-z0-9]", "", f.label.lower())
            if f.category in ("social", "connectivity", "employment", "amenity", "price", "project", "market") \
                    and f.status != "Reported" and key not in labels:
                facts.append(f)
                labels.add(key)
        facts = facts[:5]
        if facts:
            lead = (f"Here is where {self.place} has a genuine case for {self.audience}. "
                    "Each point is tied to a fact you can check and to why it matters.")
            bullets = []
            for f in facts:
                bullets.append(self.bullet(self.use(f), self.why_once(f.category)))
            strongest = facts[0]
            self.takeaways["strength"] = f"Strongest point for {self.audience}: {_lc(strongest.text.rstrip('.'))}."
            return lead + "\n\n" + "\n".join(bullets)
        crit = self.priorities[:4]
        lead = (f"Without verified local figures, judge {self.place}'s strengths on the points that matter "
                f"most to {self.audience}:")
        bullets = [f"- **{CATEGORY_TITLES[c]}:** {self.why(c)}" for c in crit if self.why(c)]
        return lead + "\n\n" + "\n".join(bullets)

    def connectivity(self, plan: BlogSectionPlan) -> str:
        facts = self.facts_in("connectivity", unused=False, limit=8)
        parts = []
        if facts:
            parts.append(f"How long daily trips take shapes life in {self.place} more than any amenity. "
                         "Here is what is on record:")
            rows = ["| Destination | Distance or time | Basis |", "|---|---|---|"]
            for f in facts:
                self.use(f)
                rows.append(f"| {f.label} | {f.value} | {self.status_cell(f)} |")
            parts.append("\n".join(rows))
            parts.append("Map distances and free-flow times are best cases. The number that matters is your "
                         "door-to-door time at the hour you actually travel.")
        else:
            parts.append(f"We could not verify travel distances for {self.place}, so test connectivity directly "
                         "rather than trusting a brochure map:")
            parts.append("\n".join([
                "1. List the trips you make most often: work, school, family.",
                "2. Make each trip at the time you would normally travel, in both directions.",
                "3. Repeat once on a rainy or busy day if you can — that is the commute you will remember."]))
        infra = [f for f in self.cat.get("infrastructure", []) if re.search(r"metro|rail|road|bridge|flyover|"
                                                                            r"expressway|corridor|airport", f.text, re.I)]
        if infra:
            parts.append(f"> **Plan for today:** {infra[0].label} is {_lc(infra[0].value.rstrip('.'))}. Until it "
                         "opens, plan your commute around the roads that exist now.")
        why = self.why_once("connectivity")
        if why:
            parts.append(why)
        return "\n\n".join(parts)

    def daily_life(self, plan: BlogSectionPlan) -> str:
        known = self.facts_in("social", "amenity", limit=6)
        fresh = [f for f in known if id(f) not in self.used_facts]
        parts = [f"Everyday routines — school runs, a doctor's visit, groceries — are where {self.place} "
                 "either works for you or wears you down."]
        if fresh:
            parts.append("\n".join(self.bullet(self.use(f), self.why_once(f.category)) for f in fresh))
        if known and not fresh:
            nouns = {"social": "schools", "amenity": "green space and amenities"}
            covered = " and ".join(dict.fromkeys(nouns[f.category] for f in known if f.category in nouns))
            parts.append(f"The {covered} noted above cover the basics on paper. What a map cannot tell you is "
                         "how they work day to day, so check these in person:")
        elif known:
            parts.append("Before you shortlist, check what a map cannot tell you:")
        else:
            parts.append(f"We could not verify specific schools, hospitals or shops near {self.place}. "
                         "Check these yourself before shortlisting:")
        parts.append("\n".join([
            "- **Schools:** admission policies and transport routes, not just distance.",
            "- **Healthcare:** the nearest hospital with emergency care, and a clinic for routine visits.",
            "- **Daily shopping:** whether groceries and pharmacies are walkable or need a drive."]))
        parts.append("> **Tip:** Visit on a weekday morning and again after dark. Traffic, lighting and noise "
                     "tell you more than a weekend show-flat visit.")
        return "\n\n".join(parts)

    def pricing(self, plan: BlogSectionPlan) -> str:
        prices = self.facts_in("price", unused=False, limit=6)
        parts = []
        if prices:
            parts.append("Prices on record for the homes in this brief:")
            rows = ["| Home or item | Price | Source |", "|---|---|---|"]
            for f in prices:
                self.use(f)
                rows.append(f"| {f.label} | {f.value} | {self.status_cell(f)} |")
            parts.append("\n".join(rows))
            parts.append("Treat listed prices as the start of the conversation. Two homes with similar "
                         "headline prices can differ sharply once usable area and extra charges are included.")
            self.takeaways["money"] = ("Compare homes on carpet-area price and the all-in cost sheet, not the "
                                       "headline price." if self.india else
                                       "Compare homes on usable-area price and the all-in cost, not the headline price.")
        else:
            parts.append("Listed prices are only part of what you pay. Before comparing projects, put each one "
                         "on the same basis:")
        area_basis = ("Ask for the price per sq ft of **carpet area** — the usable floor area developers must "
                      "disclose under RERA — rather than super built-up area." if self.india else
                      "Compare the price per unit of usable floor area, not the gross or marketed area.")
        parts.append(area_basis)
        items = (["Taxes on the purchase, such as GST on under-construction homes",
                  "Stamp duty and registration charges (check the current state rates)",
                  "Parking, club membership and preferential location charges",
                  "Maintenance deposit and the first year's maintenance",
                  "Home-loan processing fees and legal fees",
                  "Interiors and furnishing"] if self.india else
                 ["Purchase taxes and registration or transfer fees",
                  "Parking and service charges",
                  "Maintenance or owners' association fees",
                  "Loan arrangement and legal fees",
                  "Interiors and furnishing"])
        rows = ["| Cost to budget for | Ask for it in writing? |", "|---|---|"]
        rows += [f"| {i} | Yes — in the cost sheet |" for i in items[:-1]] + [f"| {items[-1]} | Your own estimate |"]
        parts.append("\n".join(rows))
        return "\n\n".join(parts)

    def tradeoffs(self, plan: BlogSectionPlan) -> str:
        bullets = []
        first_takeaway = ""
        for f in [f for f in self.cat.get("infrastructure", []) if f.status != "Reported"][:2]:
            first_takeaway = first_takeaway or f"{f.label} is not operating yet"
            bullets.append(f"- **{f.label} is not operating yet.** It is {_lc(f.value.rstrip('.'))}. Timelines for "
                           "public projects often slip, so do not let it be the reason the numbers work.")
        if self.has("connectivity"):
            bullets.append("- **Short on the map is not always short at peak hour.** Measured distances do not "
                           "include traffic, signals or parking — test the trip yourself.")
        if self.has("price"):
            bullets.append("- **A wide price range hides real differences.** Projects in the same area can differ "
                           "in usable area, specification and extra charges.")
        bullets.append("- **Newer areas have less resale history.** If most homes nearby are recent launches, "
                       "there are fewer past sales to tell you what a fair price is.")
        bullets.append("- **Construction next door.** Ask what is planned on neighbouring plots; years of dust, "
                       "noise and heavy vehicles affect daily life and rental appeal.")
        aud = self.ctx.audience.get("id")
        if aud == "families":
            bullets.append("- **Proximity does not guarantee a school seat.** Check admission policies before "
                           "treating a nearby school as a reason to buy.")
        elif aud == "investors":
            bullets.append("- **Liquidity is not guaranteed.** Selling quickly can mean accepting a discount, "
                           "especially where many similar homes are on the market.")
        elif aud == "nri":
            bullets.append("- **Distance makes oversight harder.** Construction quality and paperwork are harder "
                           "to monitor from abroad without a trusted local representative.")
        lead = (f"No location is right for everyone. These are the trade-offs {self.audience} should weigh "
                f"honestly before committing to {self.place}:" if self.place_known else
                f"These are the trade-offs {self.audience} should weigh honestly:")
        if not first_takeaway:
            phrase = bullets[0].split("**")[1].rstrip(".")
            first_takeaway = phrase[:1].lower() + phrase[1:]
        self.takeaways["tradeoff"] = f"Main trade-off: {first_takeaway}."
        return lead + "\n\n" + "\n".join(bullets[:5])

    def future(self, plan: BlogSectionPlan) -> str:
        infra = self.facts_in("infrastructure", unused=False, limit=5)
        parts = []
        if infra:
            parts.append(f"Some of what will shape {self.place} over the next few years is not open yet. "
                         "Here is the status of each item on record:")
            parts.append("\n".join(self.bullet(f) if f.status == "Reported" else
                                   f"- **{f.label}:** {f.value.rstrip('.')} "
                                   f"({'as supplied' if f.origin == 'brief' else f.source})." for f in infra))
            parts.append("Announced is not the same as operational. Price in what exists today, and treat "
                         "anything still under construction as upside rather than as the reason to buy.")
        else:
            parts.append(f"We found no confirmed infrastructure announcements for {self.place} to report here. "
                         "That is not proof that nothing is planned — it means nothing could be verified.")
        parts.append("For timelines, rely on the implementing authority's official updates — the metro rail "
                     "corporation, the municipal body or the development authority — rather than marketing "
                     "material.")
        return "\n\n".join(parts)

    def who_it_suits(self, plan: BlogSectionPlan) -> str:
        rows: list[tuple[str, str, str]] = []
        social, conn, price = self.cat.get("social"), self.cat.get("connectivity"), self.cat.get("price")
        infra = self.cat.get("infrastructure")
        if self.archetype == "comparison" and self.ctx.options:
            a, b = self.ctx.options[:2]
            rows = []
            for need, pattern in (("Need metro access today", r"metro|rail"),
                                  ("Want to be near a major tech park or office hub", r"tech park|it park|corridor|"
                                                                                     r"office|employ|sez")):
                hits = {}
                for opt in (a, b):
                    match = [f for f in facts_mod.mentioning(self.facts, opt) if re.search(pattern, f.text, re.I)]
                    if match:
                        hits[opt] = match[0]
                positive = [o for o, f in hits.items() if not _NEGATIVE.search(f.text)]
                if len(positive) == 1:
                    winner = positive[0]
                    rows.append((need, winner, about(hits[winner], winner)))
            rows += [("Have a fixed workplace", "Whichever is closer at peak hour", "Commute time costs you every day"),
                    ("Have school-age children", "The one with better school access", "School runs shape the week"),
                    ("Need to move in soon", "Whichever has ready homes", "Under-construction homes add delay risk"),
                    ("Are buying to rent out", "The one with deeper rental demand", "Demand decides vacancy"),
                    ("Have a tight budget", "Compare all-in cost per carpet area", "Headline prices mislead")]
            rows = rows[:6]
            lead = f"Use your own situation to choose between {a} and {b}:"
            table = "| If you... | Lean towards | Why |\n|---|---|---|\n" + "\n".join(f"| {x} | {y} | {z} |" for x, y, z in rows)
            return lead + "\n\n" + table
        if self.archetype == "investment_analysis":
            rows = [("Long-term holder", "A reasonable fit" if (conn or social) else "Possible, after checks",
                     "Time smooths out delivery delays and price swings"),
                    ("Rental-income seeker", "Check first", "Verify agreed rents on comparable homes"),
                    ("Short-term trader", "A weak fit", "Resale takes time and transaction costs are high"),
                    ("First-time investor", "Proceed with care", "Keep a cash buffer for vacancies and repairs")]
            table = ("| Investor profile | Fit | Why |\n|---|---|---|\n"
                     + "\n".join(f"| {x} | {y} | {z} |" for x, y, z in rows))
            return f"Whether {self.place} suits you depends on what you want the money to do:\n\n{table}"
        rows.append(("Want short school and clinic runs", "A good fit" if social else "Check first",
                     social[0].sentence.rstrip(".") if social else "Verify schools and clinics nearby"))
        rows.append(("Commute daily to a fixed workplace", "Test it first",
                     conn[0].sentence.rstrip(".") if conn else "Time the trip at peak hour"))
        rows.append(("Need metro or rail access now",
                     "Not yet" if infra else "Check first",
                     f"{infra[0].label} is {_lc(infra[0].value.rstrip('.'))}" if infra else "Confirm what operates today"))
        rows.append(("Are working to a fixed budget", "Compare carefully" if price else "Check first",
                     "Compare all-in cost, not headline price"))
        rows.append(("Need to move in within months", "Shortlist ready homes", "Under-construction homes add delay risk"))
        table = ("| If you... | " + _cap(self.place) + " is... | Why |\n|---|---|---|\n"
                 + "\n".join(f"| {x} | {y} | {z} |" for x, y, z in rows))
        return (f"Whether {self.place} works depends on your priorities. Find your situation below:\n\n"
                + table)

    def due_diligence(self, plan: BlogSectionPlan) -> str:
        if self.archetype == "how_to":
            return self._process_checklist()
        items = []
        if self.india:
            items.append(f"Look up the project on {self.regulator_md} and match the registration number to your agreement.")
            items.append("Confirm the carpet area and the price per sq ft of carpet area in writing.")
        else:
            items.append("Check the project's registration and approvals with the local regulator.")
            items.append("Confirm the usable floor area and the price per unit of usable area in writing.")
        items += [
            "Ask for the approved building plan and check your unit against it.",
            "Have a property lawyer review the title documents and an encumbrance certificate."
            if self.india else "Have a property lawyer review the title and any charges on the land.",
            "Get an all-inclusive cost sheet before paying any booking amount.",
            "For under-construction homes, check the payment schedule is linked to construction milestones.",
            "For ready homes, ask for the occupancy certificate." if self.india else
            "For ready homes, ask for the completion or occupancy certificate.",
            "Check which banks have approved the project for home loans.",
        ]
        for f in self.cat.get("infrastructure", [])[:1]:
            items.append(f"Check the current status of {f.label} with the implementing authority, not the sales team.")
        aud = self.ctx.audience.get("id")
        extra = {
            "families": "Visit shortlisted schools and confirm admission policies and transport routes.",
            "investors": "Ask two or three local agents for agreed rents on comparable homes before assuming a yield.",
            "nri": "Arrange a trusted local representative for site visits and document collection.",
            "first_time_buyers": "Get a home-loan eligibility estimate from your bank before you shortlist.",
            "retirees": "Check lift backup, step-free access and the nearest emergency care.",
        }.get(aud)
        if extra:
            items.append(extra)
        if self.archetype not in ("explainer", "how_to"):
            items.append("Make your daily commute at peak hour, and visit the site after dark.")
        checklist = "\n".join(f"- [ ] {i}" for i in items[:10])
        lead = ("Run every shortlisted home through this checklist. Anything you cannot verify is a reason "
                "to pause, not a detail to sort out later.")
        if self.archetype in ("explainer", "how_to"):
            where = [("Registration and the declared carpet area" if self.india else "Registration and declared areas",
                      _cap(self.regulator_md)),
                     ("What you are promised", "Your agreement for sale — have a lawyer read it"),
                     ("Approved plans", "The local development authority or municipal body"),
                     ("What you will pay", "The all-inclusive cost sheet and every receipt")]
            table = "| What to check | Where to check it |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in where)
            lead = (f"Start with the official record, then work through the checklist.\n\n{table}\n\n"
                    "Then, for every home you shortlist:")
        self.takeaways["checklist"] = "Run the due-diligence checklist on every shortlisted project before you pay."
        return f"{lead}\n\n{checklist}"

    def _process_checklist(self) -> str:
        where = [("The current procedure and fees", "The registering authority's official website or office"),
                 ("Registration details of the project" if self.india else "Registration details",
                  _cap(self.regulator_md)),
                 ("What you are agreeing to", "The draft agreement or deed — have a lawyer read it")]
        table = "| What to confirm | Where |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in where)
        items = ["Identity and address proof for every party, with copies.",
                 "The draft agreement or deed, checked against the property papers.",
                 "Previous title documents and recent tax receipts for the property.",
                 "Proof that any duty or fee due has been paid, with reference numbers.",
                 "Approvals for the building, such as the occupancy certificate where it applies.",
                 "Photographs and any witness details the office asks for."]
        self.takeaways["checklist"] = "Confirm the current document list with the authority before your appointment."
        return ("Confirm the exact list with the authority before your appointment. Start here:\n\n" + table
                + "\n\nA typical set of documents and checks:\n\n" + "\n".join(f"- [ ] {i}" for i in items))

    def steps(self, plan: BlogSectionPlan) -> str:
        points = [p.strip().rstrip(".") for p in self.bi.key_points if p.strip()]
        if self.archetype == "how_to" and len(points) >= 3:
            steps = [f"**{_cap(p)}.**" for p in points[:8]]
            return "Follow the steps in this order:\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        if self.archetype == "how_to":
            steps = [("Confirm the current procedure.", "Rules and fees change; check the official authority's "
                      "website or office before you start."),
                     ("Gather your documents.", "Collect identity, address and property papers, with copies."),
                     ("Check every detail against the record.", "Names, areas and survey numbers must match exactly."),
                     ("Pay fees through official channels only.", "Keep every receipt and reference number."),
                     ("Track the application.", "Follow up in writing and keep a record of each response.")]
        else:
            steps = [("Fix your all-in budget.", "Start from your loan eligibility and savings, then subtract "
                      "taxes, registration and interiors."),
                     ("Shortlist locations by daily life.", "Rank areas by the commute and the schools or clinics "
                      "you will actually use."),
                     ("Shortlist registered projects.", f"Keep only projects you can find on {self.regulator_md}."
                      if self.india else "Keep only projects with verifiable registration and approvals."),
                     ("Visit at the hours that matter.", "Go at peak traffic and after dark, not only on a weekend."),
                     ("Compare cost sheets like-for-like.", "Put every option on carpet-area price plus all charges."),
                     ("Get the paperwork reviewed.", "A lawyer's review of title and agreement costs far less "
                      "than a dispute."),
                     ("Book, then track milestones.", "Pay against construction stages and keep every receipt.")]
        body = "\n".join(f"{i}. **{t}** {d}" for i, (t, d) in enumerate(steps, 1))
        return ("Most of the risk in this decision is in the order you do things. Work through it in this sequence:"
                f"\n\n{body}")

    def common_mistakes(self, plan: BlogSectionPlan) -> str:
        if self.archetype == "explainer":
            items = [("Relying on the brochure.", "Marketing material is not the legal record; the registered "
                      "details and your agreement are."),
                     ("Not getting it in writing.", "A verbal assurance is hard to enforce. Ask for every "
                      "commitment in the agreement."),
                     ("Assuming the rules are the same everywhere.", "Details vary by state and change over time; "
                      "check the current position for your location." if self.india else
                      "Details vary by jurisdiction and change over time."),
                     ("Skipping advice on complex cases.", "Joint ownership, inherited land or resale chains are worth "
                      "a lawyer's review.")]
        elif self.archetype == "how_to":
            items = [("Starting before the papers are ready.", "Missing documents are the most common cause of "
                      "repeat visits."),
                     ("Paying intermediaries without receipts.", "Pay official fees through official channels only."),
                     ("Ignoring small mismatches.", "A spelling or area mismatch can stall the whole process."),
                     ("Not tracking the application.", "Follow up in writing and keep reference numbers.")]
        else:
            items = [("Budgeting on the headline price.", "Budget on the all-in cost sheet, including taxes and charges."),
                     ("Skipping the registration check.", f"Check the project on {self.reg_name} yourself."
                      if self.india else "Check registration and approvals yourself."),
                     ("Buying on announced infrastructure.", "Price in what operates today; treat the rest as upside."),
                     ("Trusting a weekend visit.", "Visit at peak hour and after dark before deciding."),
                     ("Stretching the loan to the limit.", "Leave room for interest-rate rises and a cash buffer.")]
        body = "\n".join(f"- **{m}** {fix}" for m, fix in items)
        first, fix = items[0]
        self.takeaways.setdefault("tradeoff", f"Most common mistake: {first[:1].lower() + first[1:].rstrip('.')}. {fix}")
        return f"These mistakes come up again and again — each one has a simple fix:\n\n{body}"

    def comparison_table(self, plan: BlogSectionPlan) -> str:
        opts = self.ctx.options[:3]
        if len(opts) < 2:
            return self.at_a_glance(plan)
        cells: dict[str, dict[str, list[str]]] = {}
        for opt in opts:
            option_facts = facts_mod.mentioning(self.facts, opt)
            for f in option_facts:
                cells.setdefault(f.category, {}).setdefault(opt, []).append(about(f, opt))
                self.use(f)
            ranked = facts_mod.ordered(option_facts, ["employment", "connectivity", "infrastructure", "social",
                                                      "price", "amenity", "market", "project"])
            if ranked:
                self.takeaways[f"option:{opt}"] = f"{opt}: {_lc(about(ranked[0], opt))}."
        parts = [f"Neither {opts[0]} nor {opts[1]} is better in general. Put them side by side on the "
                 "criteria that actually change the decision:"]
        if cells:
            header = "| Criterion | " + " | ".join(opts) + " |"
            rows = [header, "|" + "---|" * (len(opts) + 1)]
            for cat, per in cells.items():
                rows.append(f"| {CATEGORY_TITLES.get(cat, cat)} | " +
                            " | ".join("; ".join(per.get(o, [])) or "Not verified" for o in opts) + " |")
            parts.append("\n".join(rows))
            if any("Not verified" in r for r in rows):
                parts.append("Cells marked “Not verified” were not covered by the facts available for this "
                             "article — check them before you decide.")
        criteria = [("Commute to your workplace", "You pay for it every working day", "Time it at peak hour"),
                    ("Price per sq ft of carpet area" if self.india else "Price per unit of usable area",
                     "Headline prices hide area differences", "Ask for cost sheets"),
                    ("Schools and healthcare", "They shape daily life and resale demand", "Visit and check admissions"),
                    ("Infrastructure status", "Announced is not operating", "Check the authority's updates"),
                    ("Developer delivery record", "Delays cost rent and EMI together",
                     f"Check past projects on {self.reg_name}" if self.india else "Check past projects")]
        rows = ["| What to compare | Why it matters | How to check |", "|---|---|---|"]
        rows += [f"| {a} | {b} | {c} |" for a, b, c in criteria]
        parts.append("\n".join(rows))
        return "\n\n".join(parts)

    def option_profiles(self, plan: BlogSectionPlan) -> str:
        parts = []
        any_facts = False
        for opt in self.ctx.options[:3]:
            facts = facts_mod.mentioning(self.facts, opt)
            parts.append(f"### {opt}")
            if facts:
                any_facts = True
                parts.append("\n".join(f"- **{CATEGORY_TITLES.get(f.category, 'Fact')}:** {about(f, opt)}."
                                       for f in facts[:4]))
            else:
                parts.append(f"No verified facts specific to {opt} were available for this article. Start with "
                             "the commute you would make most often and the registration status of the projects "
                             "you are shown.")
        if not any_facts:
            parts.insert(0, "Both options can work. What separates them for you is how each one handles your "
                            "commute, your budget and your timeline.")
        return "\n\n".join(parts)

    def pros_cons(self, plan: BlogSectionPlan) -> str:
        if self.archetype == "comparison" and self.ctx.options:
            parts = []
            for opt in self.ctx.options[:3]:
                facts = facts_mod.mentioning(self.facts, opt)
                cons = [f for f in facts if f.category == "infrastructure" or _NEGATIVE.search(f.text)]
                pros = [f for f in facts if f not in cons and f.category != "price"]
                pro_text = "; ".join(about(f, opt) for f in pros[:2]) or "Not verified for this article"
                con_text = "; ".join(about(f, opt) for f in cons[:2]) or "Check the delivery record and the peak-hour commute"
                parts.append(f"| {opt} | {pro_text} | {con_text} |")
            return ("Each option's case, based on the facts available:\n\n| Option | In its favour | "
                    "Watch out for |\n|---|---|---|\n" + "\n".join(parts))
        pros = [f for f in self.facts if f.category in ("social", "connectivity", "amenity", "price")][:4]
        cons = [f for f in self.cat.get("infrastructure", [])][:2]
        pro_lines = [f"- {f.label}: {_lc(f.value.rstrip('.'))}" for f in pros] or \
            [f"- Fits {self.audience} who prioritise {CATEGORY_TITLES[c].lower()}" for c in self.priorities[:2]]
        con_lines = [f"- {f.label} is {_lc(f.value.rstrip('.'))}, not yet operating" for f in cons] + [
            "- Less resale history than established areas",
            "- Construction activity nearby"]
        return ("### Pros\n\n" + "\n".join(pro_lines) + "\n\n### Cons\n\n" + "\n".join(con_lines[:4]))

    def demand_drivers(self, plan: BlogSectionPlan) -> str:
        facts = self.facts_in("connectivity", "social", "market", "amenity", limit=5)
        parts = []
        if facts:
            parts.append(f"Demand for homes in {self.place} rests on things tenants and buyers can see. "
                         "These are the drivers on record:")
            parts.append("\n".join(self.bullet(self.use(f), self.why(f.category)) for f in facts))
        else:
            parts.append(f"No verified demand indicators for {self.place} were available for this article. "
                         "Look for these before you invest:")
            parts.append("\n".join(["- **Employment within commuting range.** Tenants follow jobs.",
                                    "- **Operating transport links.** Not announced ones.",
                                    "- **Schools and hospitals.** They widen the pool of family tenants.",
                                    "- **Limited competing supply.** Many similar launches nearby slow rentals and resale."]))
        if not self.has("market"):
            parts.append(f"> **Note:** No verified rental or price-growth data was supplied for {self.place}. "
                         "Treat any yield or appreciation figure you are quoted as a claim to check.")
        return "\n\n".join(parts)

    def investment_returns(self, plan: BlogSectionPlan) -> str:
        market = self.facts_in("market", unused=False, limit=4)
        parts = ["Your return has two parts: the rent you collect while you hold, and the price you get when you "
                 "sell. Both depend on things you can check before buying."]
        if market:
            parts.append("What is on record:")
            parts.append("\n".join(self.bullet(self.use(f)) for f in market))
        parts.append("| Holding horizon | What matters most |\n|---|---|\n"
                     "| Short | Entry price and resale liquidity — transaction costs eat short-term gains |\n"
                     "| Medium | Rental demand and delivery on time |\n"
                     "| Long | Infrastructure that actually opens and the area's maturity |")
        parts.append("> **Important:** No one can guarantee appreciation or rental income. Be wary of anyone who "
                     "promises either.")
        self.takeaways["returns"] = "Calculate any yield on your all-in cost, using agreed rents rather than asking rents."
        return "\n\n".join(parts)

    def context(self, plan: BlogSectionPlan) -> str:
        a = self.archetype
        parts = []
        if a == "explainer":
            parts.append(f"In plain terms, {_lc(self.ctx.subject)} is about what you are legally buying and what "
                         "you can hold the seller to. The marketing version and the legal version are not always "
                         "the same.")
            legal = self.facts_in("legal", "price", "project", limit=3)
            if legal:
                parts.append("What the facts supplied for this article say:")
                parts.append("\n".join(self.bullet(self.use(f)) for f in legal))
        elif a == "how_to":
            parts.append(f"Before you start, be clear about three things: who has to be present, which documents "
                         f"are needed, and which office or portal handles {_lc(self.ctx.subject)} where you are.")
            parts.append("Requirements and fees change, so confirm the current procedure with the official "
                         "authority before you begin.")
        else:
            concern = self.ctx.audience.get("concern", "")
            parts.append(f"For {self.audience}, the difficulty is {concern}. The listed price, the brochure and "
                         "the site visit each show only part of the picture.")
            parts.append("The way through is to separate what you can verify from what you are being told — "
                         "and to make each decision in the right order.")
        if self.bi.key_points:
            parts.append("Points to keep in mind throughout:")
            parts.append("\n".join(f"- {_cap(p.strip().rstrip('.'))}." for p in self.bi.key_points[:5]))
        return "\n\n".join(parts)

    def why_it_matters(self, plan: BlogSectionPlan) -> str:
        why = self.ctx.audience.get("why", {})
        bullets = ["- **Your money.** It changes what you are really paying per unit of usable space.",
                   "- **Your protection.** What is registered and written down is what you can enforce.",
                   "- **Your comparisons.** Projects are only comparable when measured the same way."]
        if why.get("legal"):
            bullets.append(f"- **For {self.audience}.** {why['legal']}")
        return (f"{self.ctx.subject} matters to {self.audience} in three practical ways:\n\n" + "\n".join(bullets))

    def how_it_works(self, plan: BlogSectionPlan) -> str:
        points = [p.strip().rstrip(".") for p in self.bi.key_points if p.strip()]
        if len(points) >= 3:
            return "In practice it works like this:\n\n" + "\n".join(
                f"{i}. {_cap(p)}." for i, p in enumerate(points[:7], 1))
        steps = ["Find the official definition or rule that applies to your purchase.",
                 "Compare it with what the brochure or sales team told you.",
                 "Check the figures in your agreement and cost sheet against the official record.",
                 "Raise any mismatch in writing before you pay the next instalment."]
        return "In practice, use it as a check at each stage of the purchase:\n\n" + "\n".join(
            f"{i}. {s}" for i, s in enumerate(steps, 1))

    def market_drivers(self, plan: BlogSectionPlan) -> str:
        facts = self.facts_in("market", "infrastructure", "price", limit=6)
        if facts:
            return (f"These are the signals on record for {self.place}, with their sources:\n\n"
                    + "\n".join(self.bullet(self.use(f)) for f in facts)
                    + "\n\nHeadlines report what was announced; check the source before treating any of them as settled.")
        return (f"No verified market data for {self.place} was available for this article. Watch these "
                "signals instead:\n\n"
                "- **Registered sales prices** rather than asking prices.\n"
                "- **New launches nearby**, which add competing supply.\n"
                "- **Infrastructure that opens**, not infrastructure that is announced.\n"
                "- **Rental demand** from people who work nearby.")

    def what_it_means(self, plan: BlogSectionPlan) -> str:
        rows = [("Buying to live in", "Focus on daily life and paperwork; market timing matters less over a long stay."),
                ("Buying to invest", "Check agreed rents and resale depth before relying on growth."),
                ("Planning to sell", "Price against registered sales, not against the most optimistic listing.")]
        return ("What this means depends on why you are in the market:\n\n| If you are... | What it means |\n"
                "|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows))

    def custom(self, plan: BlogSectionPlan) -> str:
        """A section in the author's own words: the facts that bear on it,
        what they do and do not show, and how to test the claim."""
        text = plan.purpose or plan.heading
        lowered = text.lower()
        words = set(re.findall(r"[a-z]{4,}", lowered)) - {"why", "what", "this", "that", "with", "from", "becoming",
                                                         "major", "real", "estate"}
        cats = {c for c, pats in blog_config.editorial().get("fact_categories", {}).items()
                if any(re.search(p, lowered) for p in pats)}
        investment = bool(re.search(r"invest|hub|growth|demand|potential|appreciat|return", lowered))
        if investment:
            cats |= {"market", "infrastructure", "connectivity", "price"}
        if re.search(r"compar|nearby|alternative|versus|vs", lowered):
            cats |= {"price", "project"}
        matched = [f for f in self.facts
                   if f.category in cats or words & set(re.findall(r"[a-z]{4,}", f.text.lower()))][:5]
        parts = []
        if matched:
            fresh = [f for f in matched if id(f) not in self.used_facts]
            seen = [f for f in matched if id(f) in self.used_facts]
            parts.append("Questions like this deserve evidence rather than optimism.")
            if seen:
                parts[-1] += (" Of the facts covered above, these bear on it: "
                              + ", ".join(_lc(f.label) for f in seen[:4]) + ".")
            if fresh:
                parts.append("Also on record:")
                parts.append("\n".join(self.bullet(self.use(f)) for f in fresh))
            infra = [f for f in matched if f.category == "infrastructure"]
            solid = [f for f in matched if f.category != "infrastructure"]
            if infra and solid:
                parts.append("Read them together. What exists today supports the case; the projects still under "
                             "construction would strengthen it, but only once they open.")
            elif infra:
                parts.append("Most of this case rests on projects that are not open yet. That makes it a forecast, "
                             "not a fact — weigh it accordingly.")
            else:
                parts.append("These support the case, but a handful of facts is not a trend. Look for the broader "
                             "evidence below before relying on it.")
        else:
            parts.append("We could not verify figures that answer this directly, so rather than repeat claims, "
                         "here is how to judge it yourself.")
        if investment:
            tests = ["- **Jobs within commuting range.** Demand follows employment; ask which employers are nearby.",
                     "- **Infrastructure that is operating.** Count what is open, not what is announced.",
                     "- **Registered sale prices over time.** Asking prices and brochure claims are not evidence.",
                     "- **New supply.** Many similar launches at once can hold prices and rents back."]
        else:
            tests = ["- **The source of each claim.** Official authorities first, marketing material last.",
                     "- **Like-for-like comparisons.** The same type of home, the same area basis, the same date.",
                     "- **What has actually happened.** Completed projects and registered sales, not plans."]
        parts.append("To test it yourself, look for:\n\n" + "\n".join(tests))
        return "\n\n".join(parts)

    # =======================================================================

    def generator(self, slot: str) -> Callable[[BlogSectionPlan], str]:
        return getattr(self, slot, None) or self.custom


_NEGATIVE = re.compile(r"\b(?:no|not|lacks?|without|yet to|under construction|limited|congest\w*|far from|"
                       r"delay\w*|poor)\b", re.I)


def about(fact: Fact, option: str) -> str:
    """A fact restated about an option, without the option's own name:
    "Whitefield has an operational metro" -> "Has an operational metro"."""
    text = fact.text.strip().rstrip(".")
    trimmed = re.sub(r"^" + re.escape(option) + r"(?:'s)?\s+", "", text, flags=re.I)
    if trimmed != text:
        trimmed = re.sub(r"^(?:is|are)\s+", "", trimmed, flags=re.I)
    return _cap(trimmed)


def default_cta(blog_input: BlogInput, archetype: str = "",
                ctx: Optional[brief_mod.BriefContext] = None) -> tuple[str, str]:
    """(heading, body) for a call to action built only from the brief —
    never a contact detail in the text (real ones are shown beside it)."""
    ctx = ctx or brief_mod.context(blog_input)
    archetype = archetype or archetypes_mod.classify(blog_input, ctx)
    text = (blog_input.cta_text or "").strip().rstrip(".")
    brand = (blog_input.brand_name or "").strip()
    project = (blog_input.project_name or "").strip()
    place = ctx.location.short or ""
    if text:
        heading = _cap(text)
    elif archetype in ("explainer", "how_to"):
        heading = "Get your paperwork checked before you commit"
    elif project or place:
        heading = f"See {project or place} for yourself"
    else:
        heading = "Take the next step with confidence"
    if brand:
        body = (f"{brand} can walk you through the documents and the numbers for "
                f"{project or 'your shortlist'}, so you can verify everything above in person.")
    elif text:
        body = f"{_cap(text)} — and bring this checklist so you can verify everything above in person."
    elif archetype in ("explainer", "how_to"):
        body = ("Before you sign, take your agreement and cost sheet to a property lawyer and check both "
                "against the registered details.")
    else:
        body = ("Arrange a site visit at the time of day you would normally travel, and bring this checklist "
                "with you.")
    return heading, body


# ---------------------------------------------------------------------------
# References — only from where facts actually came from
# ---------------------------------------------------------------------------

def build_references(blog_input: BlogInput, research: Optional[ResearchPack],
                     ctx: Optional[brief_mod.BriefContext] = None) -> list[BlogReference]:
    ctx = ctx or brief_mod.context(blog_input)
    refs: list[BlogReference] = []
    competitors = blog_config.editorial().get("competitor_domains", [])

    from src.agents.blog.normaliser import find_placeholders

    def add(ref: BlogReference) -> None:
        if ref.url and (any(c in ref.url.lower() for c in competitors) or find_placeholders(ref.url)):
            return
        if ref.url and any(r.url == ref.url for r in refs):
            return
        refs.append(ref)

    name, url = regulator(ctx)
    if ctx.location.india or url:
        add(BlogReference(title=name[:1].upper() + name[1:], url=url, kind="official",
                          note="Where to verify project registration, approvals and the developer's record"))
    if research is not None:
        for ref in research.official_sources:
            add(ref.model_copy(update={"kind": "official"}))
        if research.background_source:
            add(BlogReference(title="Background on the area (Wikipedia)", url=research.background_source,
                              kind="background", note="General background, not a source for figures"))
        for item in research.news[:4]:
            if item.url:
                add(BlogReference(title=item.title, url=item.url, kind="news",
                                  note=f"{item.source}{', ' + item.published if item.published else ''}"))
        if research.nearby:
            add(BlogReference(title="OpenStreetMap", url="https://www.openstreetmap.org", kind="research",
                              note="Nearby places and straight-line distances"))
        if research.journeys:
            add(BlogReference(title="OSRM road routing (Project OSRM)", url="https://project-osrm.org",
                              kind="research", note="Road distances and free-flow travel times"))
    for ref in brief_mod.reference_links(blog_input):
        add(ref)
    if any(f.strip() for f in blog_input.local_facts):
        publisher = (blog_input.brand_name or "").strip() or "the publisher"
        add(BlogReference(title=f"Facts supplied by {publisher}", kind="brief",
                          note="Project and local figures provided with this article's brief"))
    return refs


# ---------------------------------------------------------------------------
# The whole article
# ---------------------------------------------------------------------------

def compose(blog_input: BlogInput, outline: BlogOutline, images: list[BlogImageAsset],
            research: Optional[ResearchPack] = None) -> BlogContent:
    draft = Draft(blog_input, research, outline, images)
    plans = outline.sections or [BlogSectionPlan(heading=h, purpose=outline.section_briefs.get(h, ""),
                                                 slot="custom") for h in outline.section_headings]
    sections: list[BlogSection] = []
    # The glance table first, so later sections know which facts it showed.
    bodies: dict[int, str] = {}
    order = sorted(range(len(plans)), key=lambda i: 0 if plans[i].slot in ("at_a_glance", "comparison_table") else 1)
    for i in order:
        plan = plans[i]
        bodies[i] = draft.generator(plan.slot)(plan).strip()
    image_ids = [img.image_id for img in images]
    for i, plan in enumerate(plans):
        body = bodies[i]
        if not body:
            continue
        image_id = None
        if image_ids and plan.format in ("prose", "list") and len(sections) % 2 == 1:
            image_id = image_ids.pop(0)
        sections.append(BlogSection(heading=plan.heading, body_markdown=body, image_id=image_id,
                                    format=plan.format, slot=plan.slot))
    if image_ids:
        for section in sections:
            if not section.image_id and image_ids and section.format != "table":
                section.image_id = image_ids.pop(0)
    conclusion_heading, conclusion = draft.conclusion()
    cta_heading, cta = draft.cta()
    content = BlogContent(
        title=outline.working_title or archetypes_mod.working_title(blog_input, draft.archetype, draft.ctx),
        intro_markdown=draft.intro(),
        sections=sections,
        conclusion_heading=conclusion_heading,
        conclusion_markdown=conclusion,
        cta_heading=cta_heading,
        cta_markdown=cta,
        key_takeaways=draft.key_takeaways(),
        faqs=draft.faqs(),
        archetype=draft.archetype,
        primary_keyword=outline.primary_keyword or draft.ctx.primary_keyword,
        secondary_keywords=list(outline.secondary_keywords),
        writer_mode="fallback",
    )
    return _fit_length(content, blog_input.target_word_count, plans)


def _fit_length(content: BlogContent, target: Optional[int], plans: list[BlogSectionPlan]) -> BlogContent:
    """Trim optional material when the brief asked for a short post:
    FAQ first, then optional sections nobody requested, never a requested
    or closing section."""
    if not target:
        return content
    from src.agents.blog.article import full_markdown, plain

    def count(c: BlogContent) -> int:
        return len(plain(full_markdown(c)).split())

    limit = target * 1.3
    if count(content) > limit and content.faqs:
        content.faqs = content.faqs[:2]
    if count(content) > limit:
        content.faqs = []
    protected = {p.heading for p in plans if p.covers or p.slot in ("who_it_suits", "due_diligence", "custom")}
    for section in reversed(list(content.sections)):
        if count(content) <= limit or len(content.sections) <= 4:
            break
        if section.heading not in protected:
            content.sections.remove(section)
    return content
