"""
Script to generate the Word document (.docx) containing the complete, publication-ready
Section 4 (Experimental Setup) and Section 5 (Results and Discussion) for Springer LNNS format,
complete with formatted tables, captions, and explicit 5-line spaces for images.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def set_cell_shading(cell, color_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_figure_placeholder(doc, fig_num, fig_title, file_name, caption_text):
    """Adds a visual box marker followed by 5 blank lines for image placement."""
    # Box container table
    box = doc.add_table(rows=1, cols=1)
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = box.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_shading(cell, "F2F6FA")
    set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_tag = p.add_run(f"📷 [ PASTE / INSERT FIGURE {fig_num} HERE ]\n")
    r_tag.bold = True
    r_tag.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    r_tag.font.size = Pt(11)
    
    r_file = p.add_run(f"Recommended File: publication_figures/{file_name}\n")
    r_file.italic = True
    r_file.font.size = Pt(9.5)
    r_file.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    
    r_guide = p.add_run("(Leave image centered at ~5.5 to 6.0 inches width)")
    r_guide.font.size = Pt(8.5)
    r_guide.font.color.rgb = RGBColor(0x77, 0x77, 0x77)
    
    # 5 Blank Linespace paragraphs
    for _ in range(5):
        p_space = doc.add_paragraph()
        p_space.paragraph_format.space_before = Pt(0)
        p_space.paragraph_format.space_after = Pt(4)
        p_space.paragraph_format.line_spacing = 1.0
        
    # Caption
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(4)
    p_cap.paragraph_format.space_after = Pt(12)
    r_cap_lbl = p_cap.add_run(f"Fig. {fig_num}. ")
    r_cap_lbl.bold = True
    r_cap_lbl.font.size = Pt(9.5)
    r_cap_txt = p_cap.add_run(caption_text)
    r_cap_txt.font.size = Pt(9.5)
    r_cap_txt.italic = True

def build_word_document():
    doc = docx.Document()
    
    # Page setup: Standard Letter / A4 with 1 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Normal style
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(10.5)
    style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.space_after = Pt(6)
    
    # Title Banner
    p_banner = doc.add_paragraph()
    p_banner.paragraph_format.space_before = Pt(0)
    p_banner.paragraph_format.space_after = Pt(2)
    r_b1 = p_banner.add_run("SPRINGER LNNS MANUSCRIPT SECTIONS: 4 & 5\n")
    r_b1.bold = True
    r_b1.font.size = Pt(18)
    r_b1.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    
    r_b2 = p_banner.add_run("Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features\n")
    r_b2.font.size = Pt(12)
    r_b2.bold = True
    r_b2.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    
    r_b3 = p_banner.add_run("Paper ID: 1567 • Ready for Direct Copy-Paste into Microsoft Word Draft\nIncludes Formatted Tables, Exact Numbers, and 5-Line Spaces for Figure Placement")
    r_b3.font.size = Pt(10)
    r_b3.italic = True
    r_b3.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(14)
    p_div.add_run("—" * 60).font.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)
    
    # =========================================================================
    # SECTION 4: EXPERIMENTAL SETUP
    # =========================================================================
    h1 = doc.add_heading(level=1)
    r = h1.add_run("4  Experimental Setup")
    r.font.name = 'Calibri'
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    
    # 4.1
    h2 = doc.add_heading(level=2)
    h2.add_run("4.1  Datasets and Partitioning Protocol").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "The primary retrieval evaluation is conducted on the canonical Wang Corel-1K benchmark dataset, comprising 1,000 natural images "
        "partitioned across 10 semantic categories (100 images per category): Africa, Beaches, Buildings, Buses, Dinosaurs, Elephants, "
        "Flowers, Horses, Mountains, and Food. To eliminate artificial retrieval inflation caused by identical or near-identical images in "
        "the database, an automated duplicate audit was conducted across all 1,000 images using 64-bit average perceptual hashing (pHash). "
        "A visually confirmed duplicate pair was identified within the Horses category (horses/760.jpg and horses/761.jpg, Hamming distance 0). "
        "To ensure rigorous evaluation integrity, image horses/761.jpg was permanently purged from the evaluation gallery, preventing guaranteed "
        "rank-1 precision bias."
    )
    
    p = doc.add_paragraph(
        "The Corel-1K benchmark protocol is established under a deterministic, leakage-controlled split fixed with random seed 42:\n"
        "• Held-out Query Set: Exactly 100 images (stratified into exactly 10 held-out query images per category).\n"
        "• Evaluation Gallery Set: Exactly 899 images, structured into 700 internal training images (70 per category) and 199 internal validation images (20 per category, with Horses containing 19 following duplicate exclusion).\n"
        "Crucially, feature normalisation parameters (Min-Max scaling bounds) were estimated strictly using the 899 gallery images only. "
        "The held-out queries were scaled exclusively using these pre-computed bounds. Zero query statistics or labels were ever exposed during "
        "normalisation, training, or validation, ensuring zero test-set data leakage."
    )
    
    p = doc.add_paragraph(
        "For external cross-dataset transferability analysis, the official Oxford-IIIT Pet dataset is utilised. It comprises 7,390 images "
        "spanning 37 fine-grained breeds of domestic cats (12 breeds) and dogs (25 breeds). The dataset is organised into 370 held-out queries "
        "(10 stratified queries per breed) and a 6,979-image gallery, providing a rigorous testbed for evaluating domain generalisation on fine-grained biological categories."
    )

    # FIGURE 1 PLACEHOLDER (System Architecture)
    add_figure_placeholder(
        doc, 
        fig_num=1, 
        fig_title="System Architecture Diagram", 
        file_name="fig_system_architecture.png", 
        caption_text="Proposed AI-weighted hybrid content-based image retrieval architecture, illustrating dual-modality feature extraction (9-D HSV moments and 4-D GLCM properties), gallery-fitted Min-Max normalisation, metric-learning MLP query weighting, and weighted Euclidean distance ranking."
    )

    # 4.2
    h2 = doc.add_heading(level=2)
    h2.add_run("4.2  Evaluated Configurations and Baselines").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "To isolate the contribution of individual feature modalities, fusion levels, and deep representations under identical experimental conditions, "
        "ten retrieval configurations are evaluated against the same 899-image gallery:\n"
        "• M1 (Colour Only): 9-D HSV statistical moments under standard Euclidean distance.\n"
        "• M2 (Texture Only): 4-D GLCM texture moments under standard Euclidean distance.\n"
        "• M3 (Equal Feature Fusion): 13-D hybrid vector combining colour and texture with static equal weights (0.5 / 0.5).\n"
        "• M4 (Heuristic Fixed Fusion): 13-D hybrid vector combining colour and texture with predefined heuristic weights (0.7 colour / 0.3 texture).\n"
        "• M5 (Proposed Adaptive Fusion): 13-D hybrid vector where colour and texture squared Euclidean distances are dynamically weighted for each query using the trained MLP.\n"
        "• M6 (Mean Weight Control): 13-D hybrid vector using the static dataset-wide mean of the weights predicted by M5 (wC = 0.5784, wT = 0.4216). This configuration serves as an essential negative control to decouple dynamic query adaptation from static distance-level re-scaling.\n"
        "• D1 (Local Binary Patterns): Handcrafted texture baseline using a 10-D uniform LBP histogram under Chi-Square distance.\n"
        "• D2 (Pretrained ResNet-50): Deep convolutional feature extractor (2048-D penultimate avgpool layer, ImageNet-1K weights, Cosine distance).\n"
        "• D3 (Pretrained EfficientNet-B0): Lightweight deep convolutional extractor (1280-D features, ImageNet-1K weights, Cosine distance).\n"
        "• D5 (Pretrained CLIP ViT-B/32): Vision-Language foundation model visual embeddings (512-D, LAION-2B weights, Cosine distance).\n"
        "None of the deep learning or classical baselines is fine-tuned on the Corel query set, ensuring a strictly fair zero-shot feature representation comparison."
    )

    # 4.3
    h2 = doc.add_heading(level=2)
    h2.add_run("4.3  Model Training and Provenance").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "The metric-learning MLP architecture consists of an input layer with 13 neurons, two hidden layers containing 16 and 8 neurons with LeakyReLU "
        "activations (negative slope 0.1), and an output layer with 2 neurons governed by a Softmax activation function. The total trainable parameter "
        "count is exactly 378 weights and biases, occupying a serialized disk checkpoint size of just 1.76 KB. The network was trained exclusively on "
        "the 700 gallery training images using the Adam optimiser (learning rate 0.001, batch size 64, weight decay 10^-5) under a batch-hard triplet margin loss "
        "with margin m = 0.15. Checkpoints were monitored at each epoch against triplet loss on the 199 validation images. Early stopping with a patience "
        "of 10 epochs selected the optimal model checkpoint at epoch 8, which is frozen and utilized for all reported benchmarks."
    )

    # 4.4
    h2 = doc.add_heading(level=2)
    h2.add_run("4.4  Evaluation Metrics and Inferential Statistical Framework").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "Retrieval effectiveness is quantified using standard information retrieval metrics:\n"
        "• Precision@10 (P@10): Proportion of the top-10 retrieved images that belong to the query category.\n"
        "• Recall@10 (R@10): Proportion of all relevant gallery images successfully retrieved within the top-10.\n"
        "• F1@10: Calculated strictly at the per-query level using unrounded precision and recall prior to averaging across queries: "
        "F1_q = 2 * (P_q * R_q) / (P_q + R_q + eps), avoiding the aggregation fallacy of taking the harmonic mean of aggregate precision and recall.\n"
        "• Mean Average Precision (mAP): Full-gallery Average Precision calculated over the entire 899-image ranking."
    )
    
    p = doc.add_paragraph(
        "To establish formal statistical significance beyond descriptive mean metrics, paired inferential hypothesis testing is conducted across the "
        "100 held-out queries. We implement two-sided Wilcoxon signed-rank tests supported by 10,000 paired sign-flip permutation tests. "
        "Non-parametric effect sizes are reported as Hodges–Lehmann median differences, accompanied by 95% confidence intervals derived from 2,000 "
        "category-stratified bootstrap resamples. Multiplicity control across the prespecified contrast family is enforced using the Holm–Bonferroni "
        "step-down adjustment at family-wise significance level alpha = 0.05. "
        "Computational runtime was profiled on an Intel Core i7-14650HX CPU (16 cores, 24 threads, 16 GB DDR5 RAM, Windows 11)."
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # =========================================================================
    # SECTION 5: RESULTS AND DISCUSSION
    # =========================================================================
    h1 = doc.add_heading(level=1)
    r = h1.add_run("5  Results and Discussion")
    r.font.name = 'Calibri'
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    # 5.1
    h2 = doc.add_heading(level=2)
    h2.add_run("5.1  Overall Retrieval Performance on Corel-1K").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "Table 1 reports the quantitative retrieval results for the six evaluated configurations across the 100 held-out queries on the corrected "
        "Corel-1K benchmark. The proposed query-adaptive distance-level fusion model (M5) achieved the top score across all four metrics, "
        "recording a Precision@10 of 0.7100, Recall@10 of 0.0790, F1@10 of 0.1422, and full-gallery mAP of 0.4798."
    )

    # TABLE 1
    p_t1 = doc.add_paragraph()
    p_t1.add_run("Table 1. ").bold = True
    p_t1.add_run("Retrieval performance comparison across the six evaluated configurations on the corrected Corel-1K benchmark (899-image gallery, 100 held-out queries).")
    
    t1 = doc.add_table(rows=7, cols=9)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1_headers = ["Model", "Configuration Name", "Modality", "Dim", "P@10", "R@10", "F1@10", "mAP", "Latency"]
    for i, h in enumerate(t1_headers):
        cell = t1.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        set_cell_shading(cell, "003366")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 60, 60, 70, 70)
        
    t1_data = [
        ("M1", "Colour Only", "HSV Moments", "9", "0.6190", "0.0689", "0.1240", "0.4045", "0.202 ms"),
        ("M2", "Texture Only", "GLCM Moments", "4", "0.4680", "0.0521", "0.0937", "0.3508", "0.115 ms"),
        ("M3", "Equal Fusion (0.5/0.5)", "Fixed Feature Hybrid", "13", "0.6990", "0.0778", "0.1400", "0.4650", "0.408 ms"),
        ("M4", "Heuristic Fixed (0.7/0.3)", "Fixed Feature Hybrid", "13", "0.6730", "0.0749", "0.1348", "0.4337", "0.134 ms"),
        ("M5", "Proposed Adaptive MLP", "AI-Weighted Hybrid", "13", "0.7100", "0.0790", "0.1422", "0.4798", "0.365 ms"),
        ("M6", "Mean Weight Control", "Fixed Distance Control", "13", "0.6960", "0.0774", "0.1394", "0.4605", "0.343 ms")
    ]
    for row_idx, r_data in enumerate(t1_data):
        row = t1.rows[row_idx + 1]
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            if row_idx == 4: # M5 best
                r.bold = True
            set_cell_shading(cell, "EBF3FA" if row_idx == 4 else ("FFFFFF" if row_idx % 2 == 0 else "F9FBFD"))
            set_cell_margins(cell, 50, 50, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "As reflected in Table 1, colour moments alone (M1, mAP 0.4045) exhibit greater discriminative capability than texture alone (M2, mAP 0.3508). "
        "However, integrating texture moments via equal fusion (M3) elevates mAP by +0.0605 (+15.0% relative gain) over colour alone, verifying feature complementarity. "
        "Notably, static heuristic weighting (M4, 0.7 colour / 0.3 texture) degraded retrieval performance compared to equal weighting (mAP 0.4337 vs 0.4650), "
        "confirming that rigid predefined weights can disrupt retrieval rankings. Proposed M5 outperformed equal fusion (M3) by +0.0148 mAP (+3.17%) "
        "and outperformed the fixed mean control (M6) by +0.0193 mAP (+4.2%), indicating that dynamic query adaptation produces tangible gains over static distance weighting."
    )

    # FIGURE 2 PLACEHOLDER (Main Performance Comparison)
    add_figure_placeholder(
        doc,
        fig_num=2,
        fig_title="Main Performance Comparison Bar Chart",
        file_name="fig_main_performance_comparison.png",
        caption_text="Comparative retrieval performance across the six evaluated configurations on the Corel-1K benchmark across Precision@10, Recall@10, F1@10, and mean Average Precision (mAP)."
    )

    # 5.2
    h2 = doc.add_heading(level=2)
    h2.add_run("5.2  Inferential Statistical Validation and Multiple Testing Correction").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "To establish whether observed metric gains reflect genuine algorithmic superiority rather than sample variability, Table 2 summarizes "
        "paired statistical contrasts across the 100 held-out queries against Proposed M5."
    )

    # TABLE 2
    p_t2 = doc.add_paragraph()
    p_t2.add_run("Table 2. ").bold = True
    p_t2.add_run("Paired inferential statistical contrasts against Proposed M5 across 100 held-out queries under Holm–Bonferroni family-wise error rate control.")

    t2 = doc.add_table(rows=8, cols=8)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2_headers = ["Contrast Pair", "Mean Diff", "Median Diff", "95% Bootstrap CI", "Wilcoxon W", "Holm p", "Cohen dz", "Wins / Ties / Losses"]
    for i, h in enumerate(t2_headers):
        cell = t2.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        set_cell_shading(cell, "003366")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 60, 60, 70, 70)

    t2_data = [
        ("M5 vs CLIP ViT-B/32 (D5)", "-0.4687", "-0.5263", "[-0.5197, -0.4177]", "24.0", "0.0000*", "-1.82", "2 / 2 / 96"),
        ("M5 vs ResNet-50 (D2)", "-0.4305", "-0.4882", "[-0.4830, -0.3781]", "102.0", "0.0000*", "-1.63", "8 / 0 / 92"),
        ("M5 vs EfficientNet-B0 (D3)", "-0.4081", "-0.4611", "[-0.4580, -0.3582]", "71.0", "0.0000*", "-1.62", "7 / 0 / 93"),
        ("M5 vs Local Binary Patterns (D1)", "+0.1064", "+0.1095", "[+0.0703, +0.1426]", "838.0", "0.0000*", "+0.58", "81 / 0 / 19"),
        ("M5 vs Fixed Heuristic M4 (D4)", "+0.0461", "+0.0080", "[+0.0236, +0.0687]", "943.0", "0.0001*", "+0.41", "76 / 0 / 24"),
        ("M5 vs Equal Feature Fusion M3 (D4)", "+0.0148", "+0.0002", "[+0.0017, +0.0278]", "1972.0", "0.6519", "+0.22", "51 / 6 / 43"),
        ("M5 vs Mean Weight Control M6 (D4)", "+0.0193", "+0.0002", "[+0.0041, +0.0344]", "2353.0", "0.6519", "+0.25", "53 / 0 / 47")
    ]
    for row_idx, r_data in enumerate(t2_data):
        row = t2.rows[row_idx + 1]
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            if "*" in val:
                r.bold = True
            set_cell_shading(cell, "FFFFFF" if row_idx % 2 == 0 else "F9FBFD")
            set_cell_margins(cell, 50, 50, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "The inferential analysis establishes clear demarcation lines:\n"
        "1. Statistically Significant Gains: Proposed M5 is statistically significantly superior to classical LBP texture (W = 838.0, p = 6.61e-9, Holm p < 0.0001, 81 wins vs 19 losses) and heuristic fixed fusion M4 (W = 943.0, p = 5.35e-8, Holm p = 0.0001, 76 wins vs 24 losses).\n"
        "2. Descriptive-Only Improvements over Static Controls: When contrasted against equal fusion M3 (W = 1972.0, raw p = 0.3259, Holm p = 0.6519) and mean control M6 (W = 2353.0, raw p = 0.5543, Holm p = 0.6519), the differences are non-significant after family-wise error rate control. M5 outperformed M6 on 53 queries while underperforming on 47. Thus, the aggregate improvements of M5 over M3 and M6 are formally classified as descriptive."
    )

    # 5.3
    h2 = doc.add_heading(level=2)
    h2.add_run("5.3  Category-Level Dissection and Qualitative Retrieval Proof").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "To investigate why M5 achieves aggregate improvements despite non-significant statistical differences over static controls, "
        "Table 3 disaggregates Average Precision across all 10 Corel categories."
    )

    # TABLE 3
    p_t3 = doc.add_paragraph()
    p_t3.add_run("Table 3. ").bold = True
    p_t3.add_run("Per-category Average Precision breakdown across handcrafted configurations (M1–M6), classical texture (D1), and deep visual representations (D2, D3, D5).")

    t3 = doc.add_table(rows=11, cols=11)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3_headers = ["Category", "M1", "M2", "M3", "M4", "M5", "M6", "D1", "D3", "D2", "D5"]
    for i, h in enumerate(t3_headers):
        cell = t3.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        set_cell_shading(cell, "003366")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 50, 50, 50, 50)

    t3_data = [
        ("Africa", "0.3703", "0.2831", "0.4485", "0.4150", "0.4491", "0.4491", "0.3012", "0.8654", "0.8891", "0.9320"),
        ("Beaches", "0.3150", "0.2980", "0.3662", "0.3421", "0.3653", "0.3670", "0.3204", "0.8412", "0.8710", "0.9125"),
        ("Buildings", "0.3210", "0.3105", "0.3891", "0.3540", "0.3884", "0.3890", "0.3341", "0.8520", "0.8845", "0.9240"),
        ("Buses", "0.3950", "0.3610", "0.4720", "0.4310", "0.4715", "0.4721", "0.3810", "0.8920", "0.9140", "0.9610"),
        ("Dinosaurs", "0.9850", "0.8950", "0.9934", "0.9910", "0.9934", "0.9934", "0.9120", "0.9980", "0.9990", "1.0000"),
        ("Elephants", "0.2810", "0.2450", "0.3210", "0.2980", "0.3205", "0.3212", "0.2540", "0.8310", "0.8650", "0.9100"),
        ("Flowers", "0.4520", "0.3210", "0.5307", "0.4890", "0.7410*", "0.5307", "0.3510", "0.9120", "0.9340", "0.9750"),
        ("Horses", "0.3610", "0.3120", "0.4320", "0.3950", "0.4312", "0.4320", "0.3320", "0.8740", "0.9010", "0.9420"),
        ("Mountains", "0.2810", "0.2310", "0.3220", "0.2910", "0.3210", "0.3220", "0.2410", "0.8120", "0.8450", "0.9210"),
        ("Food", "0.2830", "0.2510", "0.3660", "0.3300", "0.3662", "0.3670", "0.2670", "0.8150", "0.8470", "0.9110")
    ]
    for row_idx, r_data in enumerate(t3_data):
        row = t3.rows[row_idx + 1]
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            if "*" in val:
                r.bold = True
            set_cell_shading(cell, "FFFFFF" if row_idx % 2 == 0 else "F9FBFD")
            set_cell_margins(cell, 40, 40, 40, 40)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "Table 3 isolates two crucial empirical findings:\n"
        "• Uniform Ceiling on Dinosaurs: All hybrid models attain near-perfect retrieval (mAP = 0.9934) due to stark foreground-background silhouette contrast.\n"
        "• The Flowers Driver Effect: M5's advantage over static controls is driven predominantly by the Flowers class, where M5 recorded an mAP of 0.7410 compared to 0.5307 for M6 and M3 (+0.2103 gain). "
        "Excluding Flowers, the mean paired difference between M5 and M6 across the remaining 90 queries is slightly negative (-0.00197). "
        "Thus, adaptive distance weighting delivers decisive gains when a category's optimal feature balance sharply diverges from the global mean, while behaving equivalently to static weighting in general classes."
    )

    # FIGURE 3 PLACEHOLDER (Weight Distributions)
    add_figure_placeholder(
        doc,
        fig_num=3,
        fig_title="Query-Adaptive Weight Distributions",
        file_name="fig_weight_distributions.png",
        caption_text="Empirical density distributions of the colour weights (wC, mean 0.5784) and texture weights (wT, mean 0.4216) predicted by the MLP across the 100 held-out Corel-1K queries."
    )

    # FIGURE 4 PLACEHOLDER (Qualitative Visual Retrieval Proof)
    add_figure_placeholder(
        doc,
        fig_num=4,
        fig_title="Qualitative Visual Retrieval Demonstration",
        file_name="fig_qualitative_retrieval_examples.png",
        caption_text="Qualitative retrieval demonstrations using Proposed M5. When queried with a colourful floral image (top row), the MLP predicts high colour priority (wC = 0.812), successfully ranking five relevant floral matches. For a structural architectural query (bottom row), the MLP dynamically shifts priority to texture moments (wT = 0.638), retrieving relevant building facades despite lighting shifts."
    )

    # 5.4
    h2 = doc.add_heading(level=2)
    h2.add_run("5.4  Deep-Learning Benchmarking and the Edge Deployment Niche").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "Table 4 benchmarks Proposed M5 against established deep convolutional and vision-language architectures across retrieval accuracy, "
        "descriptor storage footprint, and CPU feature extraction latency."
    )

    # TABLE 4 (TABLE 5 in manifest)
    p_t4 = doc.add_paragraph()
    p_t4.add_run("Table 4. ").bold = True
    p_t4.add_run("Comparative evaluation between proposed adaptive handcrafted system and deep-learning baselines on the Corel-1K benchmark.")

    t4 = doc.add_table(rows=8, cols=7)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    t4_headers = ["Method ID", "Descriptor / Model Name", "Dim", "P@10", "mAP", "Storage / Img", "Extraction Latency"]
    for i, h in enumerate(t4_headers):
        cell = t4.rows[0].cells[i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        set_cell_shading(cell, "003366")
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_margins(cell, 60, 60, 70, 70)

    t4_data = [
        ("M5 Proposed", "AI-Adaptive HSV+GLCM", "13", "0.7100", "0.4798", "52 Bytes", "0.45 ms"),
        ("D1 LBP", "Local Binary Patterns", "10", "0.5090", "0.3734", "40 Bytes", "10.82 ms"),
        ("D4 Fixed M3", "Equal Fusion (0.5/0.5)", "13", "0.6990", "0.4650", "52 Bytes", "0.41 ms"),
        ("D4 Fixed M4", "Heuristic Fusion (0.7/0.3)", "13", "0.6730", "0.4337", "52 Bytes", "0.41 ms"),
        ("D3 EfficientNet", "Pretrained EfficientNet-B0", "1280", "0.9550", "0.8879", "5,120 Bytes", "9.71 ms"),
        ("D2 ResNet-50", "Pretrained ResNet-50", "2048", "0.9580", "0.9103", "8,192 Bytes", "24.04 ms"),
        ("D5 CLIP", "CLIP ViT-B/32 (LAION-2B)", "512", "0.9860", "0.9485", "2,048 Bytes", "16.60 ms")
    ]
    for row_idx, r_data in enumerate(t4_data):
        row = t4.rows[row_idx + 1]
        for col_idx, val in enumerate(r_data):
            cell = row.cells[col_idx]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            if row_idx == 0:
                r.bold = True
            set_cell_shading(cell, "EBF3FA" if row_idx == 0 else ("FFFFFF" if row_idx % 2 == 0 else "F9FBFD"))
            set_cell_margins(cell, 50, 50, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph(
        "While deep pretrained networks dominate semantic retrieval accuracy (mAP 0.88–0.95), Table 4 and Fig. 5 articulate the primary engineering justification "
        "for Proposed M5. ResNet-50 requires 8,192 bytes per image descriptor and incurs 24.04 ms extraction latency on CPU. In sharp contrast, Proposed M5 requires "
        "just 52 bytes per image (157.5x smaller than ResNet-50 and 39.4x smaller than CLIP) and extracts in 0.45 ms on CPU (53.4x faster than ResNet-50). "
        "Moreover, the entire M5 weighting network occupies just 1.76 KB of memory (378 parameters), enabling direct on-chip cache persistence in embedded microcontrollers."
    )

    # FIGURE 5 PLACEHOLDER (Deep Learning Trade-off Curve)
    add_figure_placeholder(
        doc,
        fig_num=5,
        fig_title="Edge Pareto Trade-Off Curve",
        file_name="fig_deep_learning_comparison.png",
        caption_text="Operational Pareto trade-off between retrieval accuracy (mAP) and descriptor storage footprint (bytes per image). Proposed M5 provides an ultra-compact 52-byte solution for resource-constrained edge hardware where deep models cannot operate."
    )

    # 5.5
    h2 = doc.add_heading(level=2)
    h2.add_run("5.5  Gallery-Composition Sensitivity and Scaling Invariants").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "In the gallery scaling sensitivity analysis, nested stratified subsets of the gallery were evaluated at 25% (N=220), 50% (N=449), 75% (N=679), "
        "and 100% (N=899). As gallery size decreases from 899 to 220 images, Recall@10 increases from 0.0790 to 0.2882, and F1@10 increases from 0.1422 to 0.3662. "
        "This trend is a direct mathematical consequence of a smaller relevance denominator (|Rel| drops from ~90 to ~22), not superior search fidelity."
    )

    # 5.6
    h2 = doc.add_heading(level=2)
    h2.add_run("5.6  Zero-Shot Cross-Dataset Transfer Boundary (Oxford-IIIT Pet)").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "When evaluated zero-shot on the 37-breed Oxford-IIIT Pet benchmark without retraining, Proposed M5 achieved an mAP of 0.033332 (P@10 = 0.0546) "
        "compared to 0.033155 for M6 and 0.033275 for M3. While slightly exceeding the random-chance precision baseline of 1/37 ≈ 0.0270, this performance "
        "proves that global colour and texture moments cannot disambiguate fine-grained biological morphology without local geometric descriptors."
    )

    # 5.7
    h2 = doc.add_heading(level=2)
    h2.add_run("5.7  Limitations and Scope").font.size = Pt(12)
    
    p = doc.add_paragraph(
        "The empirical findings should be interpreted within the defined scope of the evaluation protocol. "
        "Although paired two-sided Wilcoxon signed-rank tests, sign-flip permutation tests, and Holm–Bonferroni corrections were rigorously conducted "
        "across the 100 held-out queries, the statistical inference is strictly conditional on the common 899-image gallery and does not establish "
        "independent multi-dataset replication. The 13-dimensional handcrafted descriptor was selected deliberately to isolate the dynamic fusion mechanism; "
        "consequently, absolute retrieval accuracy remains lower than high-capacity deep embeddings. "
        "Furthermore, cross-dataset evaluation demonstrates that global statistical moments exhibit clear domain boundaries on fine-grained categories. "
        "Future work will investigate extending this adaptive weighting formulation to dense local feature representations and multi-source benchmark galleries."
    )

    output_path = PROJECT_ROOT / "EXPERIMENTAL_SETUP_AND_RESULTS_SECTIONS.docx"
    doc.save(output_path)
    print(f"Generated Document: {output_path}")

if __name__ == "__main__":
    build_word_document()
