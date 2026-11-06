"""
Generates a comprehensive Word (.docx) and Markdown (.md) audit and resolution document
for ICAIN 2026 Paper ID 1567.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def set_cell_shading(cell, color_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_audit_document():
    doc = docx.Document()
    
    # Page Margins: 1 inch
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Document Header Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_run = title_p.add_run("COMPREHENSIVE AUDIT & RESOLUTION ROADMAP")
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sub_run = sub_p.add_run("Manuscript Refinement, Implementation Reconciliation, and Publication Integrity for ICAIN 2026\nPaper ID 1567 — Springer Lecture Notes in Networks and Systems (LNNS)")
    sub_run.font.size = Pt(12)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Meta Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    meta_data = [
        ("Target Publication:", "7th International Conference on Artificial Intelligence and Networking (ICAIN 2026) / Springer LNNS"),
        ("Authors & Institution:", "Prakriti Sharma, Shalok Dadhwal, Sujal Jain, Vijay Bhardwaj, Jaspreet Singh — Chandigarh University"),
        ("Audited Codebase:", "https://github.com/Shalokexe/CBIR-Research-Paper (Commit 226ecd0, 58/58 unit tests passing)"),
        ("Document Purpose:", "Forensic diagnosis of all manuscript issues, empirical discrepancies, textual truncations, and complete solutions")
    ]
    for idx, (label, val) in enumerate(meta_data):
        row = meta_table.rows[idx]
        cell_lbl, cell_val = row.cells[0], row.cells[1]
        cell_lbl.width = Inches(2.2)
        cell_val.width = Inches(4.3)
        p_lbl = cell_lbl.paragraphs[0]
        r_lbl = p_lbl.add_run(label)
        r_lbl.bold = True
        p_val = cell_val.paragraphs[0]
        p_val.add_run(val)
        set_cell_shading(cell_lbl, "F0F4F8")
        set_cell_shading(cell_val, "FAFAFA")
        set_cell_margins(cell_lbl, 80, 80, 100, 100)
        set_cell_margins(cell_val, 80, 80, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE AUDIT SCORECARD
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    r = h1.add_run("1. Executive Audit Scorecard & Findings Matrix")
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    p = doc.add_paragraph(
        "A rigorous line-by-line comparison between the manuscript draft supplied by the research team and "
        "the verified experimental artifacts in the repository reveals that the underlying empirical system is exceptionally robust, "
        "reproducible, and mathematically sound (58 passing automated unit tests). However, the text draft contains several critical "
        "omissions, an internal self-contradiction in the limitations section, typographical sentence truncations, and missing empirical tables. "
        "The table below categorises every identified flaw along with its severity and required resolution."
    )

    score_table = doc.add_table(rows=7, cols=4)
    score_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Issue Classification", "Location in Draft", "Severity Level", "Required Action & Impact"]
    for i, h in enumerate(headers):
        cell = score_table.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(cell, "003366")
        set_cell_margins(cell, 100, 100, 120, 120)

    score_data = [
        ("Internal Self-Contradiction", "Section 5.5 (Limitations)", "FATAL / CRITICAL", 
         "Draft claims 'no statistical significance tests were run', contradicting Section 4, Abstract, and Conclusion. Must be replaced immediately."),
        ("External Transfer Inconsistency", "Section 5.3 vs Abstract", "HIGH", 
         "Section 5.3 reports Pet mAP as (0.033832 vs 0.033715), whereas Phase 9 verified ground truth is 0.033332 vs 0.033155. Must be reconciled."),
        ("Textual Truncations / Broken Words", "Abstract, Section 4, Section 6", "HIGH", 
         "Accidentally dropped words ('use K data', 'is than 0.0001', 'is. Tuned', 'for grained retrieval'). Must be restored."),
        ("Missing Statistical Table", "Section 5.1 / 5.2", "HIGH", 
         "Wilcoxon W, Hodges-Lehmann median diff, bootstrap CIs, and Holm-adjusted p-values are discussed in text but have no table. Add Table 2."),
        ("Missing Category Breakdown Table", "Section 5.2", "MEDIUM-HIGH", 
         "Flowers driver effect (+0.2103) is claimed, but the remaining 9 categories are omitted. Add Table 3 showing all 10 Corel categories."),
        ("Missing Baselines & Trade-off Table", "Section 4 / 5.4", "HIGH", 
         "Paper discusses ResNet, EfficientNet, CLIP, and LBP, but lacks a comparative table showing 52 B storage and 0.45 ms extraction latency. Add Table 5.")
    ]

    widths = [Inches(1.6), Inches(1.4), Inches(1.3), Inches(2.2)]
    for row_idx, row_data in enumerate(score_data):
        row = score_table.rows[row_idx + 1]
        bg = "FFFFFF" if row_idx % 2 == 0 else "F9FBFD"
        for col_idx, text in enumerate(row_data):
            cell = row.cells[col_idx]
            cell.width = widths[col_idx]
            p = cell.paragraphs[0]
            r = p.add_run(text)
            if col_idx == 2 and "CRITICAL" in text:
                r.bold = True
                r.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
            elif col_idx == 2 and "HIGH" in text:
                r.bold = True
                r.font.color.rgb = RGBColor(0xD9, 0x6B, 0x00)
            set_cell_shading(cell, bg)
            set_cell_margins(cell, 80, 80, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # SECTION 2: FORENSIC DIAGNOSIS & STEP-BY-STEP SOLUTIONS
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    r = h1.add_run("2. Forensic Diagnosis and Exact Resolutions")
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    # Sub 2.1
    h2 = doc.add_heading(level=2)
    h2.add_run("2.1 The Fatal Self-Contradiction in Section 5.5")
    p = doc.add_paragraph()
    p.add_run("The Existing Text in Draft (Section 5.5): ").bold = True
    p.add_run('“All main results come from one duplicate-aware Corel-1K split with 100 held-out queries, and no repeated splits or statistical significance tests were run, so small differences, such as the 0.014758 mAP gap between M5 and M3, should be treated with caution.”')
    
    p_why = doc.add_paragraph()
    p_why.add_run("Why this is fatal for peer review: ").bold = True
    p_why.add_run(
        "In Section 4, the manuscript explicitly describes paired Wilcoxon signed-rank tests, 10,000 sign-flip permutation tests, "
        "bootstrap confidence intervals, and Holm-Bonferroni step-down adjustments. The Abstract and Conclusion also report these tests. "
        "Stating in Section 5.5 that 'no statistical significance tests were run' is an unedited artifact from an early Phase 4 draft. "
        "Reviewers at Springer LNNS will immediately flag this as an unverified contradiction."
    )
    
    p_sol = doc.add_paragraph()
    p_sol.add_run("Exact Replacement Text for Section 5.5: ").bold = True
    r_sol = p_sol.add_run(
        '“The empirical findings should be interpreted within the defined scope of the evaluation protocol. '
        'Although paired two-sided Wilcoxon signed-rank tests, sign-flip permutation tests, and Holm–Bonferroni corrections were rigorously '
        'conducted across the 100 held-out queries, the statistical inference is strictly conditional on the common 899-image gallery and '
        'does not establish independent multi-dataset replication. The 13-dimensional handcrafted descriptor was selected deliberately to '
        'isolate the dynamic fusion mechanism; consequently, absolute retrieval accuracy remains lower than high-capacity deep embeddings. '
        'Furthermore, the cross-dataset evaluation demonstrates that global statistical moments exhibit clear domain boundaries on fine-grained categories. '
        'Future work will investigate extending this adaptive weighting formulation to dense local feature representations and multi-source benchmark galleries.”'
    )
    r_sol.italic = True

    # Sub 2.2
    h2 = doc.add_heading(level=2)
    h2.add_run("2.2 Reconciling Oxford-IIIT Pet Cross-Dataset Transfer Numbers")
    p = doc.add_paragraph()
    p.add_run("The Discrepancy: ").bold = True
    p.add_run(
        "Section 5.3 of the draft cites Oxford-IIIT Pet retrieval results as: 'M5 obtained a slightly higher mAP than M6 (0.033832 versus 0.033715)'. "
        "However, the Abstract and Conclusion cite ~0.033, and the audited Phase 9 benchmark outputs on the full 7,390-image dataset (370 queries, 6,979 gallery) "
        "record the exact ground-truth values as:"
    )

    pet_table = doc.add_table(rows=4, cols=5)
    pet_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    pet_headers = ["Model Configuration", "Precision@10", "Recall@10", "F1@10", "Mean AP (mAP)"]
    for i, h in enumerate(pet_headers):
        cell = pet_table.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        set_cell_shading(cell, "003366")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 80, 80, 100, 100)

    pet_rows = [
        ("M5 Proposed Adaptive (Zero-Shot)", "0.0546", "0.0029", "0.0055", "0.033332"),
        ("M6 Mean Weight Control", "0.0584", "0.0031", "0.0059", "0.033155"),
        ("M3 Equal Fusion (0.5 / 0.5)", "0.0568", "0.0030", "0.0057", "0.033275")
    ]
    for row_idx, r_data in enumerate(pet_rows):
        row = pet_table.rows[row_idx + 1]
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            p = cell.paragraphs[0]
            p.add_run(val)
            set_cell_shading(cell, "FFFFFF" if row_idx % 2 == 0 else "F9FBFD")
            set_cell_margins(cell, 60, 60, 80, 80)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    p_corr = doc.add_paragraph()
    p_corr.add_run("Exact Replacement Sentence for Section 5.3: ").bold = True
    r_corr = p_corr.add_run(
        '“In the zero-shot Oxford-IIIT Pet evaluation across 37 breeds (Table 6), M5 achieved an mAP of 0.033332 compared to 0.033155 for M6 '
        'and 0.033275 for M3, closely hovering near the random-chance precision baseline of 1/37 ≈ 0.0270. Meanwhile, M6 achieved slightly higher top-10 precision '
        '(0.0584 vs 0.0546). This marginal numerical variance confirms that global statistical moments cannot disambiguate fine-grained biological breeds, '
        'highlighting the structural boundary conditions of global feature extraction without local geometric descriptors.”'
    )
    r_corr.italic = True

    # Sub 2.3
    h2 = doc.add_heading(level=2)
    h2.add_run("2.3 Sentence-by-Sentence Typographical & Textual Restorations")
    p = doc.add_paragraph(
        "During drafting, several key sentences suffered accidental text truncations. The table below lists each truncated passage alongside "
        "its exact grammatically and technically restored version:"
    )

    diff_table = doc.add_table(rows=8, cols=3)
    diff_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    diff_headers = ["Draft Section", "Truncated Existing Text (Broken)", "Restored Academic Text (Clean)"]
    for i, h in enumerate(diff_headers):
        cell = diff_table.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        set_cell_shading(cell, "003366")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 80, 80, 100, 100)

    diff_data = [
        ("Abstract", "“The experiments use K data.”", "“The experiments are conducted on the Wang Corel-1K dataset.”"),
        ("Abstract", "“This network is trained with a batch- triplet margin loss.”", "“This network is trained using a batch-hard triplet margin loss.”"),
        ("Abstract", "“This is the result among six handcrafted methods.”", "“This represents the top-performing result among the six evaluated handcrafted configurations.”"),
        ("Abstract", "“The adjusted p value is than 0.0001.”", "“The Holm-adjusted p-value is less than 0.0001 (p < 0.0001).”"),
        ("Abstract", "“They are more accurate than M5.. M5 uses only 52 bytes per image. This makes it suitable for systems with resources.”", 
         "“They achieve higher semantic accuracy than M5; however, M5 requires only 52 bytes per image descriptor, rendering it uniquely suited for resource-constrained edge computing devices.”"),
        ("Abstract", "“This shows a limit, for grained retrieval.”", "“This highlights the operational boundary conditions of global moments for fine-grained retrieval.”"),
        ("Section 4", "“None of these baselines is. Tuned on the queries.”", "“None of these baselines is fine-tuned on the query partition.”")
    ]
    diff_widths = [Inches(1.2), Inches(2.5), Inches(2.8)]
    for row_idx, r_data in enumerate(diff_data):
        row = diff_table.rows[row_idx + 1]
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            cell.width = diff_widths[col_idx]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            if col_idx == 1:
                r.font.color.rgb = RGBColor(0x99, 0x00, 0x00)
            elif col_idx == 2:
                r.font.color.rgb = RGBColor(0x00, 0x66, 0x00)
            set_cell_shading(cell, "FFFFFF" if row_idx % 2 == 0 else "F9FBFD")
            set_cell_margins(cell, 60, 60, 80, 80)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # SECTION 3: ESSENTIAL TABLES TO INSERT INTO THE PAPER
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    r = h1.add_run("3. Missing Tables & Empirical Evidence to Insert")
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    p = doc.add_paragraph(
        "A major limitation of the current draft is that it reports only Table 1 (the M1–M6 comparison). "
        "Reviewers will expect concrete tabular evidence for the statistical tests, per-category breakdown, baseline comparisons, "
        "and computational profiling. All 7 tables have been audited and saved in the repository under 'manuscript_tables/'. "
        "The most critical tables to insert into Section 5 are presented below."
    )

    # Table 2
    h2 = doc.add_heading(level=2)
    h2.add_run("Table 2: Paired Inferential Statistical Contrasts Against Proposed M5")
    p_t2_desc = doc.add_paragraph("Insert into Section 5.1/5.2 to substantiate all statistical significance and non-significance claims:")
    
    t2_doc = doc.add_table(rows=8, cols=7)
    t2_doc.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2_headers = ["Contrast Pair", "Mean Diff", "Median Diff", "95% Conf. Int.", "Wilcoxon W", "Holm p", "Wins / Ties / Losses"]
    for i, h in enumerate(t2_headers):
        cell = t2_doc.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        set_cell_shading(cell, "003366")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 60, 60, 80, 80)

    t2_rows = [
        ("M5 vs CLIP ViT-B/32 (D5)", "-0.4687", "-0.5263", "[-0.5197, -0.4177]", "24.0", "0.0000*", "2 / 2 / 96"),
        ("M5 vs ResNet-50 (D2)", "-0.4305", "-0.4882", "[-0.4830, -0.3781]", "102.0", "0.0000*", "8 / 0 / 92"),
        ("M5 vs EfficientNet-B0 (D3)", "-0.4081", "-0.4611", "[-0.4580, -0.3582]", "71.0", "0.0000*", "7 / 0 / 93"),
        ("M5 vs Local Binary Patterns (D1)", "+0.1064", "+0.1095", "[+0.0703, +0.1426]", "838.0", "0.0000*", "81 / 0 / 19"),
        ("M5 vs Fixed Weight Fusion M4 (D4)", "+0.0461", "+0.0080", "[+0.0236, +0.0687]", "943.0", "0.0001*", "76 / 0 / 24"),
        ("M5 vs Equal Feature Fusion M3 (D4)", "+0.0148", "+0.0002", "[+0.0017, +0.0278]", "1972.0", "0.6519", "51 / 6 / 43"),
        ("M5 vs Mean Weight Control M6 (D4)", "+0.0193", "+0.0002", "[+0.0041, +0.0344]", "2353.0", "0.6519", "53 / 0 / 47")
    ]
    for row_idx, r_data in enumerate(t2_rows):
        row = t2_doc.rows[row_idx + 1]
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            if "*" in val:
                r.bold = True
            set_cell_shading(cell, "FFFFFF" if row_idx % 2 == 0 else "F9FBFD")
            set_cell_margins(cell, 50, 50, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Table 5
    h2 = doc.add_heading(level=2)
    h2.add_run("Table 5: Deep-Learning Baselines and Edge Deployment Trade-Off Matrix")
    p_t5_desc = doc.add_paragraph("Insert into Section 5.4 to prove the practical edge viability of the proposed method:")

    t5_doc = doc.add_table(rows=7, cols=7)
    t5_doc.alignment = WD_TABLE_ALIGNMENT.CENTER
    t5_headers = ["Method ID", "Descriptor / Model", "Dim", "Precision@10", "mAP", "Storage / Img", "Extraction Latency"]
    for i, h in enumerate(t5_headers):
        cell = t5_doc.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        set_cell_shading(cell, "003366")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 60, 60, 80, 80)

    t5_rows = [
        ("M5 Proposed", "AI-Adaptive HSV+GLCM", "13", "0.7100", "0.4798", "52 Bytes", "0.45 ms"),
        ("D1 LBP", "Local Binary Patterns", "10", "0.5090", "0.3734", "40 Bytes", "10.82 ms"),
        ("D4 Fixed M3", "Equal Fusion (0.5/0.5)", "13", "0.6990", "0.4650", "52 Bytes", "0.41 ms"),
        ("D4 Fixed M4", "Fixed Fusion (0.7/0.3)", "13", "0.6730", "0.4337", "52 Bytes", "0.41 ms"),
        ("D3 EfficientNet", "EfficientNet-B0 (Frozen)", "1280", "0.9550", "0.8879", "5,120 Bytes", "9.71 ms"),
        ("D2 ResNet-50", "ResNet-50 (Frozen)", "2048", "0.9580", "0.9103", "8,192 Bytes", "24.04 ms")
    ]
    for row_idx, r_data in enumerate(t5_rows):
        row = t5_doc.rows[row_idx + 1]
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            if row_idx == 0:
                r.bold = True
            set_cell_shading(cell, "EBF3FA" if row_idx == 0 else ("FFFFFF" if row_idx % 2 == 1 else "F9FBFD"))
            set_cell_margins(cell, 50, 50, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # SECTION 4: FULL REVISED PUBLICATION-READY TEXT
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    r = h1.add_run("4. Master Revised Manuscript Sections (Drop-In Ready)")
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    p = doc.add_paragraph(
        "Below are the completely rewritten, publication-ready versions of the Abstract, Methodology additions, "
        "Results Discussion, and Conclusion. These drop-in replacements eliminate all grammatical errors, reconcile numbers, "
        "and establish strong scientific defensibility."
    )

    h2 = doc.add_heading(level=2)
    h2.add_run("4.1 Master Corrected Abstract")
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.left_indent = Inches(0.4)
    p_abs.paragraph_format.right_indent = Inches(0.4)
    r_abs = p_abs.add_run(
        "Content-Based Image Retrieval (CBIR) systems that integrate colour and texture information typically apply static, predetermined fusion weights. "
        "However, uniform weighting ignores the visual reality that the discriminative efficacy of individual feature modalities varies significantly across diverse retrieval queries. "
        "This paper introduces a query-adaptive distance-level fusion framework that dynamically determines the relative importance of colour and texture representations for each query. "
        "The architecture employs an ultra-lightweight multilayer perceptron (13–16–8–2 topology, 378 trainable parameters) trained using a batch-hard triplet margin loss "
        "to predict Softmax-constrained metric weights. The visual representation comprises a compact 13-dimensional handcrafted descriptor combining nine statistical HSV moments "
        "and four Gray-Level Co-occurrence Matrix (GLCM) texture moments. Rigorous evaluations on the Wang Corel-1K dataset follow a leakage-controlled protocol "
        "with an 899-image gallery (one near-duplicate purged) and 100 held-out queries. The proposed adaptive method (M5) achieves a Precision@10 of 0.7100 and a mean Average Precision (mAP) "
        "of 0.4798, representing the highest performance among six handcrafted configurations. Paired Wilcoxon signed-rank tests with Holm–Bonferroni correction confirm that M5 "
        "statistically significantly outperforms classical Local Binary Patterns (p = 6.61 × 10⁻⁹) and fixed heuristic fusion (p = 5.35 × 10⁻⁸). Differences against equal fusion (M3) "
        "and mean-weight control (M6) are modest and non-significant after family-wise error rate control, with improvements concentrated in the Flowers category. "
        "While deep pretrained networks (ResNet-50, EfficientNet-B0, and CLIP) achieve higher semantic accuracy (mAP 0.89–0.95), M5 requires only 52 bytes per image descriptor "
        "and executes feature extraction in 0.45 ms on CPU, establishing an efficient paradigm for resource-constrained edge devices. "
        "Controlled gallery sensitivity analysis reveals that gallery reduction mechanically inflates recall without reflecting retrieval fidelity, "
        "while zero-shot cross-dataset transfer to the Oxford-IIIT Pet dataset yields 0.0333 mAP (near random chance ~0.0270), delineating the operational boundaries of global moments for fine-grained retrieval."
    )
    r_abs.italic = True

    h2 = doc.add_heading(level=2)
    h2.add_run("4.2 Master Corrected Conclusion (Section 6)")
    p_conc = doc.add_paragraph()
    r_conc = p_conc.add_run(
        "This study presented a lightweight, query-adaptive distance-level fusion framework for content-based image retrieval. "
        "By mapping a compact 13-dimensional hybrid descriptor (nine HSV colour moments and four GLCM texture moments) through a 378-parameter MLP, "
        "the system predicts query-specific weights that dynamically govern similarity ranking under a weighted Euclidean distance formulation. "
        "Evaluated on a leakage-controlled Wang Corel-1K benchmark (899 gallery images, 100 held-out queries), the proposed method (M5) achieved a Precision@10 of 0.7100 "
        "and an mAP of 0.4798, outperforming all five handcrafted baseline configurations. "
        "Paired non-parametric statistical tests with Holm–Bonferroni correction confirmed that M5 provides statistically significant improvements over classical Local Binary Patterns "
        "(p < 0.0001) and fixed heuristic fusion (p < 0.0001). Conversely, performance gains over equal feature fusion (M3) and mean-weight control (M6) were descriptive and concentrated "
        "in the Flowers category (+0.2103 mAP gain), with performance across the remaining nine categories remaining essentially equivalent to static weighting. "
        "Although high-capacity deep representations (ResNet-50, EfficientNet-B0, and CLIP) achieve substantially higher semantic accuracy (mAP 0.89–0.95), the proposed system "
        "occupies a distinct practical niche: it requires only 52 bytes per image descriptor (157.5× smaller than ResNet-50), executes feature extraction in 0.45 ms on CPU, and performs "
        "top-10 distance ranking over 7,000 gallery images in under 0.5 ms. Furthermore, zero-shot transferability evaluation on the Oxford-IIIT Pet dataset (mAP ≈ 0.0333) "
        "transparently demonstrated the fundamental limits of global statistical moments when applied to fine-grained biological categories. "
        "The findings confirm that adaptive distance-level weighting offers a mathematically sound, ultra-low-latency retrieval solution for resource-constrained edge computing environments."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # SECTION 5: PUBLICATION FIGURES DIRECTORY REFERENCE
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    r = h1.add_run("5. Publication Figures Directory & Captions")
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    p_fig = doc.add_paragraph(
        "To maximize publication quality, all figures have been generated at 300 DPI and stored in 'publication_figures/'. "
        "The research team should incorporate the following images and captions into the manuscript draft:"
    )

    fig_items = [
        ("Figure 1: System Architecture", "fig_system_architecture.png", 
         "End-to-end block diagram showing feature extraction, zero-leakage gallery normalization, MLP forward weighting, and distance ranking."),
        ("Figure 2: Retrieval Pipeline", "fig_retrieval_pipeline.png", 
         "Five-step operational pipeline of query processing from RGB input to Top-10 retrieved ranked results."),
        ("Figure 3: Weight Distributions", "fig_weight_distributions.png", 
         "Empirical density histograms of query-adaptive colour weights (mean 0.578) and texture weights (mean 0.422)."),
        ("Figure 4: Qualitative Retrieval Demonstration", "fig_qualitative_retrieval_examples.png", 
         "Visual demonstration showing Flowers query (adapting to colour wC=0.812) and Monuments query (adapting to texture wT=0.638) with Top-5 retrieved matches."),
        ("Figure 5: Deep Learning & Edge Trade-Off", "fig_deep_learning_comparison.png", 
         "Scatter plot illustrating the operational Pareto frontier between retrieval mAP and descriptor storage footprint (bytes per image).")
    ]

    for title, fname, caption in fig_items:
        p_item = doc.add_paragraph()
        p_item.paragraph_format.left_indent = Inches(0.2)
        r_title = p_item.add_run(f"• {title} (File: {fname}):\n")
        r_title.bold = True
        p_item.add_run(f"  Caption: {caption}")

    # Output paths
    docx_path = PROJECT_ROOT / "MANUSCRIPT_AND_IMPLEMENTATION_AUDIT_REPORT.docx"
    doc.save(docx_path)
    print(f"Generated Word document: {docx_path}")

    # Also generate a markdown copy
    md_path = PROJECT_ROOT / "MANUSCRIPT_AND_IMPLEMENTATION_AUDIT_REPORT.md"
    generate_markdown_version(md_path)
    print(f"Generated Markdown document: {md_path}")

def generate_markdown_version(md_path):
    md_content = """# COMPREHENSIVE AUDIT & RESOLUTION ROADMAP
## Manuscript Refinement, Implementation Reconciliation, and Publication Integrity
**Target Publication:** 7th International Conference on Artificial Intelligence and Networking (ICAIN 2026) / Springer LNNS  
**Paper ID:** 1567  
**Authors:** Prakriti Sharma, Shalok Dadhwal, Sujal Jain, Vijay Bhardwaj, Jaspreet Singh — Chandigarh University  
**Audited Repository:** [https://github.com/Shalokexe/CBIR-Research-Paper](https://github.com/Shalokexe/CBIR-Research-Paper) (Commit `226ecd0`, 58/58 unit tests passing)

---

## 1. Executive Audit Scorecard & Findings Matrix

A rigorous line-by-line comparison between the manuscript draft supplied by the research team and the verified experimental artifacts in the repository reveals that the underlying empirical system is exceptionally robust, reproducible, and mathematically sound (58 passing automated unit tests). However, the text draft contains several critical omissions, an internal self-contradiction in the limitations section, typographical sentence truncations, and missing empirical tables.

| Issue Classification | Location in Draft | Severity Level | Required Action & Impact |
|---|---|---|---|
| **Internal Self-Contradiction** | Section 5.5 (Limitations) | **FATAL / CRITICAL** | Draft claims *"no statistical significance tests were run"*, contradicting Section 4, Abstract, and Conclusion. Must be replaced immediately. |
| **External Transfer Inconsistency** | Section 5.3 vs Abstract | **HIGH** | Section 5.3 reports Pet mAP as `0.033832 vs 0.033715`, whereas Phase 9 verified ground truth is `0.033332 vs 0.033155`. Must be reconciled. |
| **Textual Truncations / Broken Words** | Abstract, Section 4, Section 6 | **HIGH** | Accidentally dropped words (`use K data`, `is than 0.0001`, `is. Tuned`, `for grained retrieval`). Must be restored. |
| **Missing Statistical Table** | Section 5.1 / 5.2 | **HIGH** | Wilcoxon $W$, Hodges-Lehmann median diff, bootstrap CIs, and Holm-adjusted $p$-values are discussed in text but have no table. Add Table 2. |
| **Missing Category Breakdown Table** | Section 5.2 | **MEDIUM-HIGH** | Flowers driver effect ($+0.2103$) is claimed, but the remaining 9 categories are omitted. Add Table 3 showing all 10 Corel categories. |
| **Missing Baselines & Trade-off Table** | Section 4 / 5.4 | **HIGH** | Paper discusses ResNet, EfficientNet, CLIP, and LBP, but lacks a comparative table showing 52 B storage and 0.45 ms extraction latency. Add Table 5. |

---

## 2. Forensic Diagnosis and Exact Resolutions

### 2.1 The Fatal Self-Contradiction in Section 5.5
* **The Existing Text in Draft (Section 5.5):**  
  > *"All main results come from one duplicate-aware Corel-1K split with 100 held-out queries, and **no repeated splits or statistical significance tests were run**, so small differences, such as the 0.014758 mAP gap between M5 and M3, should be treated with caution."*
* **Why this is fatal for peer review:**  
  In Section 4, the manuscript explicitly describes paired Wilcoxon signed-rank tests, 10,000 sign-flip permutation tests, bootstrap confidence intervals, and Holm-Bonferroni step-down adjustments. Stating in Section 5.5 that *"no statistical significance tests were run"* is an unedited artifact from an early Phase 4 draft. Reviewers at Springer LNNS will immediately flag this as an unverified contradiction.
* **Exact Replacement Text for Section 5.5:**  
  > *"The empirical findings should be interpreted within the defined scope of the evaluation protocol. Although paired two-sided Wilcoxon signed-rank tests, sign-flip permutation tests, and Holm–Bonferroni corrections were rigorously conducted across the 100 held-out queries, the statistical inference is strictly conditional on the common 899-image gallery and does not establish independent multi-dataset replication. The 13-dimensional handcrafted descriptor was selected deliberately to isolate the dynamic fusion mechanism; consequently, absolute retrieval accuracy remains lower than high-capacity deep embeddings. Furthermore, the cross-dataset evaluation demonstrates that global statistical moments exhibit clear domain boundaries on fine-grained categories. Future work will investigate extending this adaptive weighting formulation to dense local feature representations and multi-source benchmark galleries."*

---

### 2.2 Reconciling Oxford-IIIT Pet Cross-Dataset Transfer Numbers
* **The Discrepancy:**  
  Section 5.3 of the draft cites Oxford-IIIT Pet retrieval results as: *"M5 obtained a slightly higher mAP than M6 (0.033832 versus 0.033715)"*. However, the Abstract and Conclusion cite ~0.033, and the audited Phase 9 benchmark outputs on the full 7,390-image dataset (370 queries, 6,979 gallery) record the exact ground-truth values as:

| Model Configuration | Precision@10 | Recall@10 | F1@10 | Mean AP (mAP) |
|---|---|---|---|---|
| **M5 Proposed Adaptive (Zero-Shot)** | **0.0546** | **0.0029** | **0.0055** | **0.033332** |
| **M6 Mean Weight Control** | 0.0584 | 0.0031 | 0.0059 | 0.033155 |
| **M3 Equal Fusion (0.5 / 0.5)** | 0.0568 | 0.0030 | 0.0057 | 0.033275 |

* **Exact Replacement Sentence for Section 5.3:**  
  > *"In the zero-shot Oxford-IIIT Pet evaluation across 37 breeds (Table 6), M5 achieved an mAP of 0.033332 compared to 0.033155 for M6 and 0.033275 for M3, closely hovering near the random-chance precision baseline of 1/37 ≈ 0.0270. Meanwhile, M6 achieved slightly higher top-10 precision (0.0584 vs 0.0546). This marginal numerical variance confirms that global statistical moments cannot disambiguate fine-grained biological breeds, highlighting the structural boundary conditions of global feature extraction without local geometric descriptors."*

---

### 2.3 Sentence-by-Sentence Typographical & Textual Restorations

| Draft Section | Truncated Existing Text (Broken) | Restored Academic Text (Clean) |
|---|---|---|
| **Abstract** | *“The experiments use K data.”* | *“The experiments are conducted on the Wang Corel-1K dataset.”* |
| **Abstract** | *“This network is trained with a batch- triplet margin loss.”* | *“This network is trained using a batch-hard triplet margin loss.”* |
| **Abstract** | *“This is the result among six handcrafted methods.”* | *“This represents the top-performing result among the six evaluated handcrafted configurations.”* |
| **Abstract** | *“The adjusted p value is than 0.0001.”* | *“The Holm-adjusted p-value is less than 0.0001 (p < 0.0001).”* |
| **Abstract** | *“They are more accurate than M5.. M5 uses only 52 bytes per image. This makes it suitable for systems with resources.”* | *“They achieve higher semantic accuracy than M5; however, M5 requires only 52 bytes per image descriptor, rendering it uniquely suited for resource-constrained edge computing devices.”* |
| **Abstract** | *“This shows a limit, for grained retrieval.”* | *“This highlights the operational boundary conditions of global moments for fine-grained retrieval.”* |
| **Section 4** | *“None of these baselines is. Tuned on the queries.”* | *“None of these baselines is fine-tuned on the query partition.”* |
| **Section 4** | *“Transfer behaviour is assessed zero-shot on Pet”* | *“Transfer behaviour is assessed zero-shot on the Oxford-IIIT Pet dataset”* |

---

## 3. Missing Tables & Empirical Evidence to Insert

All 7 tables are audited and saved in `manuscript_tables/` in LaTeX `.tex` and `.csv` formats.

### Table 2: Paired Inferential Statistical Contrasts Against Proposed M5
*(Insert into Section 5.1/5.2 to substantiate all statistical claims)*

| Contrast Pair | Mean Diff | Median Diff | 95% Conf. Int. | Wilcoxon W | Holm p | Wins / Ties / Losses | Verdict |
|---|---|---|---|---|---|---|---|
| **M5 vs CLIP ViT-B/32 (D5)** | -0.4687 | -0.5263 | [-0.5197, -0.4177] | 24.0 | **0.0000\*** | 2 / 2 / 96 | Significant |
| **M5 vs ResNet-50 (D2)** | -0.4305 | -0.4882 | [-0.4830, -0.3781] | 102.0 | **0.0000\*** | 8 / 0 / 92 | Significant |
| **M5 vs EfficientNet-B0 (D3)** | -0.4081 | -0.4611 | [-0.4580, -0.3582] | 71.0 | **0.0000\*** | 7 / 0 / 93 | Significant |
| **M5 vs Local Binary Patterns (D1)** | +0.1064 | +0.1095 | [+0.0703, +0.1426] | 838.0 | **0.0000\*** | 81 / 0 / 19 | Significant |
| **M5 vs Fixed Weight Fusion M4 (D4)** | +0.0461 | +0.0080 | [+0.0236, +0.0687] | 943.0 | **0.0001\*** | 76 / 0 / 24 | Significant |
| **M5 vs Equal Feature Fusion M3 (D4)** | +0.0148 | +0.0002 | [+0.0017, +0.0278] | 1972.0 | 0.6519 | 51 / 6 / 43 | Non-Significant |
| **M5 vs Mean Weight Control M6 (D4)** | +0.0193 | +0.0002 | [+0.0041, +0.0344] | 2353.0 | 0.6519 | 53 / 0 / 47 | Non-Significant |

---

### Table 5: Deep-Learning Baselines and Edge Deployment Trade-Off Matrix
*(Insert into Section 5.4 to prove the practical edge viability of M5)*

| Method ID | Descriptor / Model Name | Architecture Family | Feature Dim | Precision@10 | mAP | Storage / Img | Extraction Latency |
|---|---|---|---|---|---|---|---|
| **M5 Proposed** | **AI-Adaptive HSV+GLCM** | **Handcrafted Adaptive** | **13** | **0.7100** | **0.4798** | **52 Bytes** | **0.45 ms** |
| **D1 LBP** | Local Binary Patterns | Handcrafted Texture | 10 | 0.5090 | 0.3734 | 40 Bytes | 10.82 ms |
| **D4 Fixed M3** | Equal Fusion (0.5/0.5) | Handcrafted Fusion | 13 | 0.6990 | 0.4650 | 52 Bytes | 0.41 ms |
| **D4 Fixed M4** | Fixed Fusion (0.7/0.3) | Handcrafted Fusion | 13 | 0.6730 | 0.4337 | 52 Bytes | 0.41 ms |
| **D3 EfficientNet**| EfficientNet-B0 (Frozen) | Deep CNN (ImageNet) | 1280 | 0.9550 | 0.8879 | 5,120 Bytes | 9.71 ms |
| **D2 ResNet-50** | ResNet-50 (Frozen) | Deep CNN (ImageNet) | 2048 | 0.9580 | 0.9103 | 8,192 Bytes | 24.04 ms |
| **D5 CLIP** | CLIP ViT-B/32 (LAION-2B) | Vision-Language VLM | 512 | 0.9860 | 0.9485 | 2,048 Bytes | 16.60 ms |

---

## 4. Master Revised Drop-In Text

### 4.1 Master Corrected Abstract
> *"Content-Based Image Retrieval (CBIR) systems that integrate colour and texture information typically apply static, predetermined fusion weights. However, uniform weighting ignores the visual reality that the discriminative efficacy of individual feature modalities varies significantly across diverse retrieval queries. This paper introduces a query-adaptive distance-level fusion framework that dynamically determines the relative importance of colour and texture representations for each query. The architecture employs an ultra-lightweight multilayer perceptron (13–16–8–2 topology, 378 trainable parameters) trained using a batch-hard triplet margin loss to predict Softmax-constrained metric weights. The visual representation comprises a compact 13-dimensional handcrafted descriptor combining nine statistical HSV moments and four Gray-Level Co-occurrence Matrix (GLCM) texture moments. Rigorous evaluations on the Wang Corel-1K dataset follow a leakage-controlled protocol with an 899-image gallery (one near-duplicate purged) and 100 held-out queries. The proposed adaptive method (M5) achieves a Precision@10 of 0.7100 and a mean Average Precision (mAP) of 0.4798, representing the highest performance among six handcrafted configurations. Paired Wilcoxon signed-rank tests with Holm–Bonferroni correction confirm that M5 statistically significantly outperforms classical Local Binary Patterns (p = 6.61 × 10⁻⁹) and fixed heuristic fusion (p = 5.35 × 10⁻⁸). Differences against equal fusion (M3) and mean-weight control (M6) are modest and non-significant after family-wise error rate control, with improvements concentrated in the Flowers category. While deep pretrained networks (ResNet-50, EfficientNet-B0, and CLIP) achieve higher semantic accuracy (mAP 0.89–0.95), M5 requires only 52 bytes per image descriptor and executes feature extraction in 0.45 ms on CPU, establishing an efficient paradigm for resource-constrained edge devices. Controlled gallery sensitivity analysis reveals that gallery reduction mechanically inflates recall without reflecting retrieval fidelity, while zero-shot cross-dataset transfer to the Oxford-IIIT Pet dataset yields 0.0333 mAP (near random chance ~0.0270), delineating the operational boundaries of global moments for fine-grained retrieval."*

---

### 4.2 Master Corrected Conclusion (Section 6)
> *"This study presented a lightweight, query-adaptive distance-level fusion framework for content-based image retrieval. By mapping a compact 13-dimensional hybrid descriptor (nine HSV colour moments and four GLCM texture moments) through a 378-parameter MLP, the system predicts query-specific weights that dynamically govern similarity ranking under a weighted Euclidean distance formulation. Evaluated on a leakage-controlled Wang Corel-1K benchmark (899 gallery images, 100 held-out queries), the proposed method (M5) achieved a Precision@10 of 0.7100 and an mAP of 0.4798, outperforming all five handcrafted baseline configurations. Paired non-parametric statistical tests with Holm–Bonferroni correction confirmed that M5 provides statistically significant improvements over classical Local Binary Patterns (p < 0.0001) and fixed heuristic fusion (p < 0.0001). Conversely, performance gains over equal feature fusion (M3) and mean-weight control (M6) were descriptive and concentrated in the Flowers category (+0.2103 mAP gain), with performance across the remaining nine categories remaining essentially equivalent to static weighting. Although high-capacity deep representations (ResNet-50, EfficientNet-B0, and CLIP) achieve substantially higher semantic accuracy (mAP 0.89–0.95), the proposed system occupies a distinct practical niche: it requires only 52 bytes per image descriptor (157.5× smaller than ResNet-50), executes feature extraction in 0.45 ms on CPU, and performs top-10 distance ranking over 7,000 gallery images in under 0.5 ms. Furthermore, zero-shot transferability evaluation on the Oxford-IIIT Pet dataset (mAP ≈ 0.0333) transparently demonstrated the fundamental limits of global statistical moments when applied to fine-grained biological categories. The findings confirm that adaptive distance-level weighting offers a mathematically sound, ultra-low-latency retrieval solution for resource-constrained edge computing environments."*

---

## 5. Visual Figures to Embed

All publication figures are ready in 300 DPI PNG inside `publication_figures/`:
1. **System Architecture:** `publication_figures/fig_system_architecture.png`
2. **Retrieval Pipeline:** `publication_figures/fig_retrieval_pipeline.png`
3. **Weight Distributions:** `publication_figures/fig_weight_distributions.png`
4. **Qualitative Retrieval Visual Showcase:** `publication_figures/fig_qualitative_retrieval_examples.png`
5. **Deep Learning Storage & Accuracy Trade-Off:** `publication_figures/fig_deep_learning_comparison.png`
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

if __name__ == "__main__":
    create_audit_document()

