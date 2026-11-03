/* The rendered-page audit: what a reader would see on every page.

   Run inside Chromium by _playwright_worker.py after fonts and images have
   loaded. Returns plain data; src/agents/brochure/qa.py decides what is a
   defect and how to repair it.

   Per page:
     boxes        each text box: how far its copy runs past the box
                  (overflow_mm, also sideways), and how much of the box the
                  copy fills (fill: 0..1, pinned footers counted)
     broken       text that must stay on one line (.keep: a unit name, a
                  price, an area, a phone number) but wrapped
     images       loaded or not; natural size and printed size, so the
                  effective resolution can be checked
     small_text   type set below 6pt
     outside      anything drawn past the trim
     whitespace   the share of the frame that carries neither picture nor type
   Globally:
     fonts        whether each family the stylesheet asks for has loaded
     texts        titles and paragraphs, for the repeated-text check
*/
function auditBrochure(doc, win) {
  const PX_PER_MM = 96 / 25.4;
  const mm = (px) => Math.round((px / PX_PER_MM) * 100) / 100;
  const out = { pages: [], texts: [], fonts: {}, page_rects: [] };

  const families = new Map();
  const rootStyle = win.getComputedStyle(doc.documentElement);
  [["--f-head", "--w-head"], ["--f-body", "--w-body"], ["--f-num", "--w-body"]].forEach(([v, w]) => {
    const first = (rootStyle.getPropertyValue(v) || "").split(",")[0].replace(/['"]/g, "").trim();
    const weight = (rootStyle.getPropertyValue(w) || "400").trim();
    if (first && !families.has(first)) families.set(first, weight);
  });
  families.forEach((weight, f) => {
    // Latin text only: a family's extended-Latin face (which declares the
    // currency block) loads only when such a glyph is set, and the rupee sign
    // itself is drawn by the stack's Inter where a family has none
    // (tests/brochure/test_fonts.py checks the stacks).
    try { out.fonts[f] = doc.fonts.check(`${weight} 12px "${f}"`, "Abc1"); } catch (e) { out.fonts[f] = false; }
  });

  const pages = Array.from(doc.querySelectorAll("section.page"));
  pages.forEach((page, index) => {
    const pr = page.getBoundingClientRect();
    out.page_rects.push({ index: index + 1, x: pr.left + win.scrollX, y: pr.top + win.scrollY, w: pr.width, h: pr.height });
    const info = {
      page: parseInt(page.dataset.page || index + 1, 10), kind: page.dataset.kind || "",
      boxes: [], broken: [], images: [], small_text: [], outside: [], over_image: [], whitespace: 0,
    };

    // --- copy set over a photograph or drawing (the cover sets its name over
    // its photograph by design; nowhere else may type touch a picture) -------
    if (info.kind !== "cover") {
      const pics = Array.from(page.querySelectorAll("figure.slot")).map((f) => f.getBoundingClientRect());
      page.querySelectorAll(".box[data-fit]").forEach((box) => {
        const walk = doc.createTreeWalker(box, NodeFilter.SHOW_TEXT);
        let tn;
        while ((tn = walk.nextNode())) {
          if (!(tn.textContent || "").trim()) continue;
          const range = doc.createRange();
          range.selectNodeContents(tn);
          const hit = Array.from(range.getClientRects()).some((r) => r.width > 1 && pics.some((p) =>
            Math.min(r.right, p.right) - Math.max(r.left, p.left) > 1.5 &&
            Math.min(r.bottom, p.bottom) - Math.max(r.top, p.top) > 1.5));
          if (hit) { info.over_image.push((tn.textContent || "").trim().slice(0, 40)); break; }
        }
      });
    }

    // --- text boxes -------------------------------------------------------
    page.querySelectorAll(".box[data-fit]").forEach((box) => {
      const br = box.getBoundingClientRect();
      let bottom = br.top, right = br.left, top = br.bottom, pinned = 0;
      const all = box.querySelectorAll("*");
      all.forEach((el) => {
        const r = el.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) return;
        const cs = win.getComputedStyle(el);
        if (cs.position === "absolute" || cs.visibility === "hidden") return;
        bottom = Math.max(bottom, r.bottom);
        right = Math.max(right, r.right);
        top = Math.min(top, r.top);
      });
      // content height without the free space a pinned footer leaves above itself
      const foot = box.querySelector(":scope > .foot, :scope > .legal");
      let natural = 0;
      Array.from(box.children).forEach((child) => {
        const r = child.getBoundingClientRect();
        const cs = win.getComputedStyle(child);
        natural += r.height + parseFloat(cs.marginTop || 0) * (child === foot ? 0 : 1) + parseFloat(cs.marginBottom || 0);
      });
      const grid = win.getComputedStyle(box).display === "grid";
      let fill;
      if (grid) {
        // side-by-side columns: the tallest column decides
        const cols = {};
        Array.from(box.children).forEach((child) => {
          const r = child.getBoundingClientRect();
          const key = Math.round(r.left);
          cols[key] = Math.max(cols[key] || 0, r.bottom - br.top);
        });
        fill = Math.max(...Object.values(cols), 0) / Math.max(1, br.height);
      } else {
        fill = natural / Math.max(1, br.height);
      }
      info.boxes.push({
        name: box.dataset.fit,
        overflow_mm: Math.max(0, mm(bottom - br.bottom), mm(br.top - top)),
        overflow_x_mm: Math.max(0, mm(right - br.right)),
        fill: Math.round(fill * 1000) / 1000,
        h_mm: mm(br.height), w_mm: mm(br.width),
      });
    });

    // --- type that must not break -------------------------------------------
    page.querySelectorAll(".keep").forEach((el) => {
      const rects = Array.from(el.getClientRects());
      const lines = new Set(rects.map((r) => Math.round(r.top)));
      if (lines.size > 1) info.broken.push((el.textContent || "").trim().slice(0, 60));
    });
    // a word split across lines (hyphenation is off, so only by force)
    page.querySelectorAll("h1, h2, h3, p, li, dt, dd, td, span").forEach((el) => {
      const cs = win.getComputedStyle(el);
      if (cs.overflowWrap === "anywhere" || cs.wordBreak === "break-all") info.broken.push("forced: " + (el.textContent || "").slice(0, 40));
    });

    // --- images ----------------------------------------------------------
    page.querySelectorAll("img[data-img]").forEach((img) => {
      const r = img.getBoundingClientRect();
      info.images.push({
        id: img.dataset.img, kind: img.dataset.kind || "image", loaded: img.complete && img.naturalWidth > 0,
        nat_w: img.naturalWidth, nat_h: img.naturalHeight, w_mm: mm(r.width), h_mm: mm(r.height),
      });
    });

    // --- small type, and anything past the trim ---------------------------------
    const walker = doc.createTreeWalker(page, NodeFilter.SHOW_TEXT);
    const seen = new Set();
    let node;
    const textRects = [];
    while ((node = walker.nextNode())) {
      const text = (node.textContent || "").trim();
      if (!text) continue;
      const el = node.parentElement;
      if (!el || seen.has(el)) continue;
      seen.add(el);
      const cs = win.getComputedStyle(el);
      if (cs.display === "none" || cs.visibility === "hidden") continue;
      const pt = parseFloat(cs.fontSize) * 0.75;
      if (pt < 5.95 && !el.closest("svg")) info.small_text.push(`${pt.toFixed(1)}pt: ${text.slice(0, 30)}`);
      const range = doc.createRange();
      range.selectNodeContents(el);
      Array.from(range.getClientRects()).forEach((r) => {
        if (r.width < 1) return;
        textRects.push(r);
        if (r.left < pr.left - 0.5 || r.right > pr.right + 0.5 || r.top < pr.top - 0.5 || r.bottom > pr.bottom + 0.5) {
          info.outside.push(text.slice(0, 40));
        }
      });
    }

    // --- white space: the frame on a 3mm grid -------------------------------------
    const ms = win.getComputedStyle(doc.documentElement);
    const mt = parseFloat(ms.getPropertyValue("--m-top")) * PX_PER_MM || 40;
    const mr = parseFloat(ms.getPropertyValue("--m-right")) * PX_PER_MM || 40;
    const mb = parseFloat(ms.getPropertyValue("--m-bottom")) * PX_PER_MM || 40;
    const ml = parseFloat(ms.getPropertyValue("--m-left")) * PX_PER_MM || 40;
    const fx0 = pr.left + ml, fy0 = pr.top + mt, fx1 = pr.right - mr, fy1 = pr.bottom - mb;
    const solid = [];
    page.querySelectorAll(".slot, .card, .zone, .fig, .stat, .rings, table, .chip, .price-badge").forEach((el) => solid.push(el.getBoundingClientRect()));
    const step = 3 * PX_PER_MM;
    let cells = 0, used = 0;
    for (let y = fy0; y < fy1; y += step) {
      for (let x = fx0; x < fx1; x += step) {
        cells++;
        const cx = x + step / 2, cy = y + step / 2;
        const hit = solid.some((r) => cx >= r.left && cx <= r.right && cy >= r.top && cy <= r.bottom)
          || textRects.some((r) => cx >= r.left - step / 2 && cx <= r.right + step / 2 && cy >= r.top - step / 3 && cy <= r.bottom + step / 3);
        if (hit) used++;
      }
    }
    info.whitespace = cells ? Math.round((1 - used / cells) * 1000) / 1000 : 0;

    page.querySelectorAll("h1, h2, h3, .lead, .body p, .card p").forEach((el) => {
      const text = (el.textContent || "").replace(/\s+/g, " ").trim();
      if (text.length >= 14) out.texts.push({ page: info.page, text: text, tag: el.tagName.toLowerCase() });
    });
    out.pages.push(info);
  });
  return out;
}
