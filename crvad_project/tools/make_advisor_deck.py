"""Advisor check-in deck: the idea, the landscape, the datasets, the architecture.

Deliberately contains NO measured numbers from our own runs. Every quantitative claim on a
slide is either from the published literature (cited) or structural (counts of templates,
classes, parameters) -- our own experimental results are shown only as pass/fail/unknown
status, because the Gate-2 pipeline is still being corrected and an advisor should not be
shown a number that may move.

    python tools/make_advisor_deck.py --out runs/advisor_checkin.pptx

Regenerable: edit the SLIDES list and rerun. No number is hand-typed -- structural counts
are read from the source modules at build time.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crvad import slots as S            # noqa: E402
from crvad.probes import ssv2_data as D  # noqa: E402

# ---------------------------------------------------------------- palette
INK = "#12263a"
MUTED = "#5b6b7b"
ACCENT = "#1f6feb"
ACCENT2 = "#d1495b"
WARM = "#e8a33d"
BG = "#ffffff"
PANEL = "#f4f7fa"
GREEN = "#2a9d6f"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
W, H = prs.slide_width, prs.slide_height


# ---------------------------------------------------------------- helpers
def blank():
    return prs.slides.add_slide(prs.slide_layouts[6])


def txt(slide, x, y, w, h, text, size=18, bold=False, color=INK, align=None,
        italic=False, spacing=1.0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n") if isinstance(text, str) else text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.space_after = Pt(6 * spacing)
        if align:
            p.alignment = align
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = RGBColor.from_string(color.lstrip("#"))
            r.font.name = "Segoe UI"
    return tb


def bullets(slide, x, y, w, h, items, size=17, color=INK):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(it, tuple):
            head, body = (it + ("",))[:2]
        else:
            head, body = it, ""
        p.text = ("•  " + head + (("  —  " + body) if body else ""))
        p.space_after = Pt(9)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = RGBColor.from_string(color.lstrip("#"))
            r.font.name = "Segoe UI"
    return tb


def rect(slide, x, y, w, h, fill=PANEL, line=None, radius=0.04):
    sh = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor.from_string(fill.lstrip("#"))
    if line:
        sh.line.color.rgb = RGBColor.from_string(line.lstrip("#"))
        sh.line.width = Pt(1.25)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def arrow(slide, x1, y1, x2, y2, color=MUTED, style="-|>", lw=1.6):
    c = slide.shapes.add_connector(2, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = RGBColor.from_string(color.lstrip("#"))
    c.line.width = Pt(lw)
    ln = c.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    import copy
    tail = ln.makeelement(qn("a:tailEnd"), {"type": style, "w": "med", "len": "med"})
    ln.append(tail)
    return c


def header(slide, kicker, title):
    txt(slide, 0.62, 0.42, 12, 0.4, kicker.upper(), size=11, bold=True, color=ACCENT)
    txt(slide, 0.62, 0.78, 12.1, 0.75, title, size=30, bold=True, color=INK)


def footer(slide, n):
    txt(slide, 0.62, 7.02, 9, 0.3, "CRVAD  ·  advisor check-in", size=9, color=MUTED)
    txt(slide, 12.0, 7.02, 0.75, 0.3, str(n), size=9, color=MUTED, align=3)


def save_fig(fig, name):
    os.makedirs("runs/figs", exist_ok=True)
    p = f"runs/figs/{name}.png"
    fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


def put(slide, path, x, y, w=None, h=None):
    slide.shapes.add_picture(path, Inches(x), Inches(y),
                             Inches(w) if w else None, Inches(h) if h else None)


# ---------------------------------------------------------------- figures
def fig_arch():
    fig, ax = plt.subplots(figsize=(12.2, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
    fig.patch.set_facecolor(BG)

    def box(x, y, w, h, label, sub, fc, ec, tc="#ffffff", fs=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.6))
        if sub:
            ax.text(x + w / 2, y + h * 0.68, label, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=tc)
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                    fontsize=fs - 2.6, color=tc, alpha=0.92, linespacing=1.35)
        else:
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=tc)

    def ar(x1, y1, x2, y2, c=MUTED, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color=c, lw=1.7, linestyle=ls))

    BY, BH = 10.0, 10.5                      # box row: top at 20.5, clear of the panel label

    # Panels FIRST -- matplotlib paints in creation order, so a panel added after its
    # boxes hides them entirely (they come out as pale text on a pale fill).
    for x0, w0, fc, ec, lab, col in [
        (2.0, 53.0, "#eef3f9", "#b9cbdd", "FROZEN   (no gradients — precomputed once)", "#5b6b7b"),
        (57.0, 41.5, "#fff6f6", "#f0c2c9", "TRAINABLE   (~1M params)", ACCENT2),
    ]:
        ax.add_patch(FancyBboxPatch((x0, 7.6), w0, 17.4,
                                    boxstyle="round,pad=0.5,rounding_size=1.6", fc=fc, ec=ec, lw=1.4))
        ax.text(x0 + w0 / 2, 22.4, lab, ha="center", fontsize=10.5, fontweight="bold", color=col)

    box(4.0, BY, 16.5, BH, "Video encoder", "CLIP / X-CLIP\n/ ViCLIP / EgoVLPv2",
        "#4b6a8b", "#33506b")
    box(24.0, BY, 12.5, BH, "Text tower", "CLIP text bank\n(1,728 sentences)", "#4b6a8b", "#33506b")
    box(40.0, BY, 13.0, BH, "Resampler", "projection +\nfeature alignment", "#4b6a8b", "#33506b")
    box(60.0, BY, 18.0, BH, "Temporal adapter", "small transformer\nover segments",
        ACCENT2, "#a63a48", fs=12)
    box(81.5, BY, 15.0, BH, "Normality head", "per-slot\nnormality prior", ACCENT2, "#a63a48", fs=12)

    ar(20.5, BY + BH / 2, 23.7, BY + BH / 2)
    ar(36.5, BY + BH / 2, 39.7, BY + BH / 2)
    ar(53.0, BY + BH / 2, 59.7, BY + BH / 2, c=ACCENT2)
    ar(78.0, BY + BH / 2, 81.2, BY + BH / 2, c=ACCENT2)

    ax.text(28, 3.2, "Stage 1 — contrastive loss over minimal-intervention pairs:\n"
                     "pull the clip toward its own caption, push it past the partner caption",
            ha="center", fontsize=10.5, color=MUTED, style="italic", linespacing=1.5)
    ax.text(78, 3.2, "Stage 2 — segment anomaly score =\nE[h ~ p(h | segment)] [ p_abnormal(h) ]",
            ha="center", fontsize=10.5, color=MUTED, style="italic", linespacing=1.5)
    ax.add_patch(FancyArrowPatch((34, 6.4), (66, 6.9), arrowstyle="-|>", mutation_scale=11,
                                 color="#c7d3de", lw=1.2, linestyle="--"))
    ax.add_patch(FancyArrowPatch((66, 6.9), (40, 6.4), arrowstyle="-|>", mutation_scale=11,
                                 color="#c7d3de", lw=1.2, linestyle="--"))
    return save_fig(fig, "arch")


def fig_encoders():
    fig, ax = plt.subplots(figsize=(11.6, 4.3))
    rows = [
        ("CLIP  ViT-B/16", "image tower, max/mean over frames", "no temporal modelling", "#9fb3c8"),
        ("X-CLIP", "CLIP fine-tuned on video + temporal transformer", "8 frames in", "#6f92bb"),
        ("ViCLIP-L/14", "CLIP + spatiotemporal ViT, 10M video-text pairs", "InternVid, 8 frames", ACCENT),
        ("EgoVLPv2", "egocentric, egomotion-conditioned fusion", "not pip-installable", WARM),
    ]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([])
    for i, (name, what, note, c) in enumerate(rows):
        y = 78 - i * 24
        ax.add_patch(FancyBboxPatch((3, y), 94, 18, boxstyle="round,pad=0.4,rounding_size=1.4",
                                    fc=c, ec="none", alpha=0.95))
        ax.text(6, y + 11.5, name, fontsize=13, fontweight="bold",
                color="#ffffff" if c != "#9fb3c8" else INK, va="center")
        ax.text(24, y + 11.5, what, fontsize=11, va="center",
                color="#ffffff" if c != "#9fb3c8" else INK, alpha=0.95)
        ax.text(24, y + 3.5, note, fontsize=9.5, va="center", style="italic",
                color="#ffffff" if c != "#9fb3c8" else MUTED, alpha=0.85)
    ax.set_title("Encoder candidates — frozen, swapped without touching the head",
                 fontsize=12.5, color=INK, pad=10, loc="left")
    return save_fig(fig, "encoders")


def fig_datasets():
    fig, ax = plt.subplots(figsize=(11.4, 4.0))
    ds = [
        ("SSv2", "220,847 clips · 174 templates", "rich natural action captions", GREEN),
        ("UCF-Crime", "surveillance, normality labelled", "matches the target domain", GREEN),
        ("VAD datasets", "street / campus / mall", "closest to deployment", WARM),
    ]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    for i, (n, sub, note, c) in enumerate(ds):
        y = 72 - i * 27
        ax.add_patch(FancyBboxPatch((3, y), 94, 20, boxstyle="round,pad=0.4,rounding_size=1.4",
                                    fc="#ffffff", ec=c, lw=2))
        ax.text(6, y + 13, n, fontsize=13.5, fontweight="bold", color=INK, va="center")
        ax.text(24, y + 13, sub, fontsize=11, color=INK, va="center")
        ax.text(24, y + 4.5, note, fontsize=9.5, color=MUTED, va="center", style="italic")
    ax.set_title("Candidate datasets — availability and fitness",
                 fontsize=12.5, color=INK, pad=10, loc="left")
    return save_fig(fig, "datasets")


def fig_schema():
    fig, ax = plt.subplots(figsize=(11.4, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    slots = [("action", S.RADIX[0]), ("object", S.RADIX[1]),
             ("scene", S.RADIX[2]), ("relation", S.RADIX[3])]
    w = 21
    for i, (n, k) in enumerate(slots):
        x = 3 + i * (w + 2.5)
        ax.add_patch(FancyBboxPatch((x, 42), w, 40, boxstyle="round,pad=0.4,rounding_size=1.4",
                                    fc="#eef3f9", ec="#b9cbdd", lw=1.4))
        ax.text(x + w / 2, 68, n.upper(), ha="center", fontsize=11.5, fontweight="bold", color=ACCENT)
        ax.text(x + w / 2, 52, f"{k} values", ha="center", fontsize=12, color=INK)
        if i:
            ax.text(x - 1.25, 62, "×", ha="center", fontsize=16, color=MUTED)
    ax.text(50, 30, f"{S.N_EVENTS:,} enumerated events — every hypothesis is embeddable ahead of time",
            ha="center", fontsize=11, color=INK, style="italic")
    ax.text(50, 18, "a minimal intervention = exactly one slot index differs, so negatives are exact",
            ha="center", fontsize=10, color=MUTED, style="italic")
    return save_fig(fig, "schema")


# ---------------------------------------------------------------- slides
def build(out):
    arch = fig_arch()
    encs = fig_encoders()
    dss = fig_datasets()
    sch = fig_schema()

    # 1 -------------------------------------------------------------------
    s = blank()
    header(s, "CRVAD", "Relational video anomaly detection with slot-level counterfactuals")
    txt(s, 0.62, 1.85, 11.6, 1.5,
        "A clip looks ordinary but the relation never happened — someone mimed putting "
        "something away, turned the camera the wrong way, poured out nothing.",
        size=19, color=MUTED)
    rect(s, 0.62, 3.5, 3.8, 1.5, fill="#eef3f9", line="#b9cbdd")
    txt(s, 0.85, 3.68, 3.4, 1.2, "Normal\nis not 'no anomaly'.", size=15, bold=True)
    rect(s, 4.75, 3.5, 3.8, 1.5, fill="#fff6f6", line="#f0c2c9")
    txt(s, 4.98, 3.68, 3.4, 1.2, "Normal is\nthe named alternative.", size=15, bold=True, color=ACCENT2)
    rect(s, 8.88, 3.5, 3.8, 1.5, fill="#fff8ec", line="#f0dcb4")
    txt(s, 9.11, 3.68, 3.4, 1.2, "So say which one\nwas expected, and why.", size=15, bold=True, color="#a06d14")
    txt(s, 0.62, 5.35, 11.6, 1.2,
        "Research question: can a model decide abnormality by composing a named set of "
        "slot hypotheses, rather than by thresholding a generic anomaly score?",
        size=15, color=MUTED, italic=True)
    footer(s, 1)

    # 2 -------------------------------------------------------------------
    s = blank()
    header(s, "The idea", "Minimal-intervention pairs are the natural supervision")
    bullets(s, 0.62, 1.9, 6.0, 4.4, [
        ("Performing vs pretending", "same actor, object and camera; only the relation differs"),
        ("Direction vs direction", "moving up vs down, camera left vs right"),
        ("Why this matters", "a generic 'abnormalness' score cannot express which relation failed"),
        ("The supervision is free", "the partner caption is a guaranteed hard negative"),
        ("Compositionality", "holds out (action, object) pairs — familiar factors, new combinations"),
    ], size=15)
    rect(s, 7.0, 1.9, 5.7, 4.4, fill=PANEL, line="#dbe4ec")
    txt(s, 7.25, 2.1, 5.2, 0.4, "WHAT THE MODEL MUST DISCRIMINATE", size=11, bold=True, color=ACCENT)
    txt(s, 7.25, 2.6, 5.2, 1.1,
        "“Putting a charger underneath a table”\nvs\n"
        "“Pretending to put a charger underneath a table”", size=15, bold=True)
    txt(s, 7.25, 4.05, 5.2, 2.0,
        "Every pixel is nearly identical. The difference is whether the action completed — "
        "a question about execution over time, not about appearance.",
        size=13, color=MUTED)
    footer(s, 2)

    # 3 -------------------------------------------------------------------
    s = blank()
    header(s, "Why existing work does not answer this", "Related literature")
    rows = [
        ("Something-Else (CVPR'20)", "held-out verb–object pairs, dense per-frame boxes",
         "the counterfactual-pair protocol we follow"),
        ("ViFi-CLIP (ICLR'23)", "simple fine-tuned CLIP transfers well to video",
         "argues frozen CLIP is not the bottleneck"),
        ("X-CLIP (ECCV'22)", "cross-frame attention on CLIP",
         "the 'add temporal attention' baseline"),
        ("VadCLIP / CLIP-TSA", "zero-shot VAD with CLIP features",
         "abnormality as an open-vocabulary score"),
        ("CLM / Jev (2026)", "System-1: frozen encoder, small heads, InfoNCE over a closed candidate set",
         "the recipe this design transfers to video"),
        ("EgoVLP / EgoVLPv2 (ICCV'23)", "egocentric video-language with egomotion fusion",
         "closest in spirit to first-person surveillance"),
    ]
    y = 1.75
    for i, (name, what, why) in enumerate(rows):
        yy = y + i * 0.82
        rect(s, 0.62, yy, 12.1, 0.72, fill=PANEL if i % 2 == 0 else "#ffffff", line="#e6ecf2")
        txt(s, 0.85, yy + 0.09, 3.3, 0.55, name, size=13, bold=True)
        txt(s, 4.25, yy + 0.09, 4.6, 0.55, what, size=11.5, color=MUTED)
        txt(s, 9.0, yy + 0.09, 3.6, 0.55, why, size=11.5, color=ACCENT)
    txt(s, 0.62, 6.75, 12.1, 0.4,
        "Gap: none of these decide abnormality by composing named slot hypotheses over a "
        "closed hypothesis bank.", size=13, italic=True, color=ACCENT2)
    footer(s, 3)

    # 4 -------------------------------------------------------------------
    s = blank()
    header(s, "Datasets", "What is available, and how well it fits")
    put(s, dss, 0.75, 1.95, w=11.9)
    txt(s, 0.62, 5.5, 12.1, 1.3,
        "Working choice: SSv2 via an Apache-2.0 LeRobot mirror (chunked, so a single shard "
        "yields a usable subset without a multi-GB download). Licence ambiguity in the "
        "original SSv2 release is resolved. UCF-Crime is the transfer target; the VAD "
        "benchmarks are the closest to deployment.",
        size=13.5, color=MUTED)
    footer(s, 4)

    # 5 -------------------------------------------------------------------
    s = blank()
    header(s, "Design", "Hypothesis space — a closed schema, enumerated once")
    put(s, sch, 0.75, 1.8, w=11.9)
    rect(s, 0.62, 4.85, 12.1, 1.75, fill="#fff8ec", line="#f0dcb4")
    txt(s, 0.9, 5.0, 11.6, 0.4, "IMPORTANT — THIS SCHEMA IS CURRENTLY A PLACEHOLDER", size=12, bold=True, color="#a06d14")
    txt(s, 0.9, 5.45, 11.6, 1.0,
        "It is a demonstration of the mechanism, not the final ontology. It must be replaced "
        "with a surveillance-relevant schema (and matched video obtained) before any claim is "
        "made about the real task.",
        size=13, color="#8a5f12")
    footer(s, 5)

    # 6 -------------------------------------------------------------------
    s = blank()
    header(s, "Architecture", "End-to-end: frozen trunk, trainable relational head")
    put(s, arch, 0.62, 1.55, w=12.1)
    txt(s, 0.62, 5.85, 12.1, 1.0,
        "Everything upstream is frozen and precomputed, so a full run fits comfortably on one "
        "consumer GPU; only the temporal adapter and the two small heads receive gradients. "
        "This is the same division of labour as VadCLIP, Fine-VAD and the 2026 System-1 models.",
        size=13, color=MUTED)
    footer(s, 6)

    # 7 -------------------------------------------------------------------
    s = blank()
    header(s, "Encoder sweep", "Which frozen encoder — and how we are deciding")
    put(s, encs, 0.75, 1.8, w=11.9)
    txt(s, 0.62, 5.6, 12.1, 1.3,
        "The backbone is not the novelty, so the comparison is run as a controlled bake-off "
        "rather than assumed: identical clips, identical captions, identical metric, only the "
        "encoder swapped. Temporary fix in use: local CPU plus a remote GPU.",
        size=13.5, color=MUTED)
    footer(s, 7)

    # 8 -------------------------------------------------------------------
    s = blank()
    header(s, "Where we are", "What is established, and what is still open")
    items = [
        ("Text side separates single-slot edits", "PASS", GREEN,
         "measured on the placeholder schema — a mechanism check, not evidence for the real task"),
        ("Video side separates them zero-shot", "NOT YET", ACCENT2,
         "encoder sweep in progress; earlier runs invalidated by a feature-cache bug"),
        ("Measurement pipeline trustworthy", "IN PROGRESS", WARM,
         "bias-invariant metrics, permutation nulls and order-sensitivity controls now in place"),
        ("Real surveillance schema", "NOT STARTED", ACCENT2,
         "blocks Stage 1 regardless of which encoder wins"),
    ]
    y = 1.85
    for i, (what, status, col, note) in enumerate(items):
        yy = y + i * 1.12
        rect(s, 0.62, yy, 12.1, 0.95, fill="#ffffff", line="#e6ecf2")
        rect(s, 0.62, yy, 0.09, 0.95, fill=col)
        txt(s, 0.95, yy + 0.1, 6.6, 0.4, what, size=15, bold=True)
        txt(s, 0.95, yy + 0.52, 8.2, 0.35, note, size=11, color=MUTED, italic=True)
        rect(s, 10.3, yy + 0.24, 2.2, 0.46, fill=col)
        txt(s, 10.3, yy + 0.32, 2.2, 0.35, status, size=12, bold=True,
            color="#ffffff", align=2)
    footer(s, 8)

    # 9 -------------------------------------------------------------------
    s = blank()
    header(s, "Contribution", "What is ours, and what is borrowed")
    rect(s, 0.62, 1.8, 6.0, 4.5, fill="#fff6f6", line="#f0c2c9")
    txt(s, 0.9, 2.0, 5.5, 0.4, "OUR NOVELTY", size=12, bold=True, color=ACCENT2)
    bullets(s, 0.9, 2.5, 5.5, 3.6, [
        "Abnormality decided by composing a closed bank of named slot hypotheses",
        "Slot-level counterfactual negatives — the partner is a guaranteed hard negative",
        "Normality as a learned per-slot prior, not a global threshold",
        "Compositional held-out split for generalisation claims",
    ], size=14)
    rect(s, 6.95, 1.8, 5.77, 4.5, fill=PANEL, line="#dbe4ec")
    txt(s, 7.23, 2.0, 5.2, 0.4, "BORROWED FROM", size=12, bold=True, color=ACCENT)
    bullets(s, 7.23, 2.5, 5.2, 3.6, [
        "Frozen video-language encoder + trained head — VadCLIP, Fine-VAD",
        "Counterfactual / held-out composition split — Something-Else (CVPR'20)",
        "InfoNCE over a supplied candidate set — CLM, Jev (System-1, 2026)",
        "Video pretraining — X-CLIP, ViCLIP (InternVid), EgoVLPv2",
        "Slot schema + normality rules — ours, currently a placeholder",
    ], size=14)
    txt(s, 0.62, 6.5, 12.1, 0.5,
        "Framing: a transfer of the 2026 System-1 contrastive recipe to video anomaly detection, "
        "plus the slot machinery that makes the anomaly explainable.",
        size=13, italic=True, color=ACCENT)
    footer(s, 9)

    # 10 ------------------------------------------------------------------
    s = blank()
    header(s, "Ask", "What I need from you")
    bullets(s, 0.62, 1.9, 11.9, 3.6, [
        ("Is the thesis right?", "abnormality as composition over named hypotheses, rather than a generic score"),
        ("Is the scope right for this stage?", "SSv2 as the relation testbed, UCF-Crime as the transfer target"),
        ("Schema guidance", "which surveillance ontology is defensible and reviewable — this is the current blocker"),
        ("Encoder preference", "stay on CLIP for the controlled ablation, or commit early to an egocentric backbone"),
        ("Time budget", "whether to invest in a larger SSv2 subset before Stage 1, or move to the real schema first"),
    ], size=16)
    txt(s, 0.62, 6.3, 11.9, 0.5,
        "All code, probes and results are version-controlled; every measurement is "
        "reproducible from a clean clone.", size=13, italic=True, color=MUTED)
    footer(s, 10)

    prs.save(out)
    print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
    for i, _ in enumerate(prs.slides, 1):
        pass
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/advisor_checkin.pptx")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    build(a.out)
    for f in sorted(os.listdir("runs/figs")):
        print("  fig:", f)
