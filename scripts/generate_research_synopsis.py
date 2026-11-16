import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_research_synopsis(output_path):
    doc = docx.Document()
    
    # Page setup
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
        sec.page_width = Inches(8.27)
        sec.page_height = Inches(11.69)
        
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    
    # -------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(36)
    r_title = p_title.add_run("ENHANCED CONTENT-BASED IMAGE RETRIEVAL USING AI-WEIGHTED HYBRID COLOUR AND TEXTURE FEATURES")
    r_title.bold = True
    r_title.font.size = Pt(16)
    r_title.font.color.rgb = RGBColor(0, 32, 96)
    p_title.paragraph_format.space_after = Pt(24)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("A Project Work Synopsis\nSubmitted in the partial fulfillment for the award of the degree of")
    r_sub.font.size = Pt(12)
    p_sub.paragraph_format.space_after = Pt(12)
    
    p_deg = doc.add_paragraph()
    p_deg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_deg1 = p_deg.add_run("BACHELOR OF ENGINEERING\nIN\nCOMPUTER SCIENCE WITH SPECIALIZATION IN\n")
    r_deg1.bold = True
    r_deg1.font.size = Pt(13)
    r_deg2 = p_deg.add_run("ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING")
    r_deg2.bold = True
    r_deg2.font.size = Pt(14)
    r_deg2.font.color.rgb = RGBColor(180, 0, 0)
    p_deg.paragraph_format.space_after = Pt(28)
    
    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_by_lbl = p_by.add_run("Submitted by:\n")
    r_by_lbl.bold = True
    r_by_lbl.font.size = Pt(12)
    
    students_text = (
        "Shalok Dadhwal (UID: 23BCS10042)\n"
        "Prakriti Sharma (UID: 23BCS10038)\n"
        "Sujal Jain (UID: 23BCS10055)\n"
        "Jaspreet Singh (UID: 23BCS10080)\n"
    )
    r_by_names = p_by.add_run(students_text)
    r_by_names.font.size = Pt(12)
    p_by.paragraph_format.space_after = Pt(20)
    
    p_sup = doc.add_paragraph()
    p_sup.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sup_lbl = p_sup.add_run("Under the Supervision of:\n")
    r_sup_lbl.bold = True
    r_sup_lbl.font.size = Pt(12)
    r_sup_name = p_sup.add_run("Prof. Dr. Vijay Bhardwaj\n")
    r_sup_name.bold = True
    r_sup_name.font.size = Pt(13)
    r_sup_desig = p_sup.add_run("Department of Computer Science and Engineering / Apex Institute of Technology")
    r_sup_desig.font.size = Pt(11)
    p_sup.paragraph_format.space_after = Pt(36)
    
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cu = p_inst.add_run("CHANDIGARH UNIVERSITY, GHARUAN, MOHALI - 140413,\nPUNJAB, INDIA\n")
    r_cu.bold = True
    r_cu.font.size = Pt(12)
    r_date = p_inst.add_run("November, 2026")
    r_date.bold = True
    r_date.font.size = Pt(12)
    
    doc.add_page_break()
    
    # -------------------------------------------------------------
    # ABSTRACT & KEYWORDS
    # -------------------------------------------------------------
    p_abs_h = doc.add_paragraph()
    r_abs_h = p_abs_h.add_run("Abstract")
    r_abs_h.bold = True
    r_abs_h.font.size = Pt(15)
    r_abs_h.font.color.rgb = RGBColor(0, 32, 96)
    p_abs_h.paragraph_format.space_after = Pt(10)
    
    abs_text = (
        "Content-Based Image Retrieval (CBIR) systems integrating colour and texture visual modalities conventionally employ rigid, "
        "predetermined fusion coefficients. However, static weighting overlooks the critical empirical reality that the discriminative "
        "efficacy of individual visual modalities fluctuates dramatically across diverse image queries. This synopsis presents a query-adaptive "
        "distance-level fusion framework that dynamically determines the relative importance of colour and texture representations for each query. "
        "The architecture utilizes an ultra-lightweight Multi-Layer Perceptron (13–16–8–2 topology, 378 trainable parameters) trained via a batch-hard "
        "triplet margin loss (margin m = 0.15) to predict Softmax-constrained metric weights subject to the unit simplex constraint (wC + wT = 1.0). "
        "The visual descriptor comprises an ultra-compact 13-dimensional handcrafted vector combining nine statistical HSV moments and four "
        "Gray-Level Co-occurrence Matrix (GLCM) Haralick texture moments. Rigorous evaluations on the Wang Corel-1K dataset follow an audited, "
        "leakage-controlled protocol across an 899-image gallery (with one exact near-duplicate purged via 64-bit perceptual hashing) and 100 held-out queries. "
        "The proposed adaptive configuration (M5) attains a Precision@10 of 0.7100 and a full-gallery mean Average Precision (mAP) of 0.4798, "
        "outperforming all five handcrafted baseline configurations. Paired two-sided Wilcoxon signed-rank tests with Holm–Bonferroni step-down correction "
        "establish that M5 is statistically significantly superior to classical Local Binary Patterns (p < 0.0001) and heuristic fixed fusion (p < 0.0001). "
        "Crucially, M5 requires just 52 bytes per image descriptor (157.5x smaller than ResNet-50) and executes feature extraction in 0.45 ms on CPU "
        "(53.4x faster than ResNet-50), providing an ultra-low-latency CBIR paradigm for resource-constrained edge computing environments. "
        "Zero-shot transferability evaluation on the 37-breed Oxford-IIIT Pet benchmark yields 0.0333 mAP, formally delineating the operational boundary "
        "of global statistical moments for fine-grained biological retrieval."
    )
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_abs = p_abs.add_run(abs_text)
    p_abs.paragraph_format.line_spacing = 1.3
    p_abs.paragraph_format.space_after = Pt(12)
    
    p_kw = doc.add_paragraph()
    r_kw_lbl = p_kw.add_run("Keywords: ")
    r_kw_lbl.bold = True
    r_kw = p_kw.add_run("Content-Based Image Retrieval, Adaptive Feature Fusion, Colour and Texture Moments, Multi-Layer Perceptron, Computational Efficiency, Inferential Statistical Validation, Edge Computing.")
    p_kw.paragraph_format.space_after = Pt(20)
    
    # -------------------------------------------------------------
    # TABLE OF CONTENTS
    # -------------------------------------------------------------
    p_toc_h = doc.add_paragraph()
    r_toc_h = p_toc_h.add_run("Table of Contents")
    r_toc_h.bold = True
    r_toc_h.font.size = Pt(15)
    r_toc_h.font.color.rgb = RGBColor(0, 32, 96)
    p_toc_h.paragraph_format.space_after = Pt(10)
    
    toc_data = [
        ("Title Page", "i"),
        ("Abstract", "ii"),
        ("1.    Introduction", "1"),
        ("1.1      Problem Definition", "1"),
        ("1.2      Project Overview", "1"),
        ("1.3      Hardware Specification", "2"),
        ("1.4      Software Specification", "2"),
        ("2.    Literature Survey", "3"),
        ("2.1      Existing System", "3"),
        ("2.2      Proposed System", "3"),
        ("2.3      Literature Review Summary", "4"),
        ("3.    Problem Formulation", "5"),
        ("4.    Research Objective", "5"),
        ("5.    Methodologies", "6"),
        ("6.    Experimental Setup", "7"),
        ("7.    Conclusion", "8"),
        ("8.    Tentative Chapter Plan for the Proposed Work", "9"),
        ("9.    References", "10")
    ]
    
    t_toc = doc.add_table(rows=len(toc_data), cols=2)
    t_toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_toc.autofit = False
    
    for row in t_toc.rows:
        row.cells[0].width = Inches(5.2)
        row.cells[1].width = Inches(1.3)
        
    for idx, (section, page) in enumerate(toc_data):
        row = t_toc.rows[idx]
        row.cells[0].text = section
        row.cells[1].text = page
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for c in row.cells:
            set_cell_margins(c, top=40, bottom=40, left=60, right=60)
            
    doc.add_page_break()
    
    # -------------------------------------------------------------
    # 1. INTRODUCTION
    # -------------------------------------------------------------
    p_s1 = doc.add_paragraph()
    r_s1 = p_s1.add_run("1. INTRODUCTION")
    r_s1.bold = True
    r_s1.font.size = Pt(14)
    r_s1.font.color.rgb = RGBColor(0, 32, 96)
    p_s1.paragraph_format.space_before = Pt(12)
    p_s1.paragraph_format.space_after = Pt(6)
    
    # 1.1 Problem Definition
    p_s11 = doc.add_paragraph()
    r_s11 = p_s11.add_run("1.1 Problem Definition")
    r_s11.bold = True
    r_s11.font.size = Pt(13)
    p_s11.paragraph_format.space_after = Pt(4)
    
    p_s11_body = doc.add_paragraph()
    p_s11_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_s11_body.paragraph_format.line_spacing = 1.3
    p_s11_body.paragraph_format.space_after = Pt(8)
    p_s11_body.add_run(
        "Content-Based Image Retrieval (CBIR) searches large visual repositories based on intrinsic image properties rather than "
        "manual keywords or metadata. In hybrid systems that integrate chromatic and textural cues, determining the relative contribution "
        "of each feature modality is critical. Conventional approaches either concatenate visual vectors directly into high-dimensional composite spaces "
        "or apply fixed, uniform weighting coefficients across all queries. Concatenation induces dimensional dominance, where components with higher "
        "variance overshadow less expansive but semantically critical modalities. Furthermore, static weighting enforces rigid compromises across "
        "heterogeneous image distributions: colour dominates floral or outdoor scenes, whereas structural texture characterizes architectural surfaces. "
        "The problem addressed in this work is the lack of an ultra-lightweight, query-adaptive distance-level fusion mechanism that can dynamically assign "
        "optimal modality weights on resource-constrained edge hardware without incurring the severe memory and computational latency overhead of deep CNNs."
    )
    
    # 1.2 Problem Overview
    p_s12 = doc.add_paragraph()
    r_s12 = p_s12.add_run("1.2 Problem Overview")
    r_s12.bold = True
    r_s12.font.size = Pt(13)
    p_s12.paragraph_format.space_after = Pt(4)
    
    p_s12_body = doc.add_paragraph()
    p_s12_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_s12_body.paragraph_format.line_spacing = 1.3
    p_s12_body.paragraph_format.space_after = Pt(8)
    p_s12_body.add_run(
        "Modern deep learning representations (such as ResNet-50, EfficientNet, and CLIP foundation models) deliver exceptional semantic retrieval accuracy "
        "(mAP 0.88 to 0.95), but require 2,048 to 8,192 bytes per image descriptor and take 12 to 25 ms per image on CPU. These figures make them "
        "impractical for embedded micro-servers and battery-powered Internet of Things (IoT) edge devices. This project proposes an AI-weighted distance-level "
        "fusion architecture built around an ultra-compact 13-dimensional handcrafted visual descriptor: nine statistical HSV colour moments and four "
        "Gray-Level Co-occurrence Matrix (GLCM) Haralick texture moments. A tiny 378-parameter Multilayer Perceptron (MLP) trained via metric learning "
        "dynamically determines query-specific metric weights under a strict Softmax unit simplex constraint. By computing independent Euclidean distances "
        "in colour and texture spaces and linearly combining them via predicted query weights, the framework preserves modality metric structures, prevents "
        "dimensional dominance, and achieves microsecond-level retrieval latencies."
    )
    
    # 1.3 Hardware Specification
    p_s13 = doc.add_paragraph()
    r_s13 = p_s13.add_run("1.3 Hardware Specification")
    r_s13.bold = True
    r_s13.font.size = Pt(13)
    p_s13.paragraph_format.space_after = Pt(4)
    
    hw_specs = [
        ("Development Host CPU", "Intel Core i7-14650HX (16 Cores, 24 Threads, up to 5.2 GHz)"),
        ("Host System Memory", "16 GB DDR5 RAM (5600 MHz)"),
        ("Storage Subsystem", "512 GB PCIe 4.0 NVMe SSD (minimum 5 GB free for image database & caches)"),
        ("Target Edge Hardware", "Raspberry Pi 4/5, NVIDIA Jetson Nano, Intel NUC (<= 512 MB RAM footprint)")
    ]
    t_hw = doc.add_table(rows=len(hw_specs)+1, cols=2)
    t_hw.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_hw.rows[0].cells[0].text = "Component"
    t_hw.rows[0].cells[1].text = "Technical Specification"
    set_cell_background(t_hw.rows[0].cells[0], "F2F2F2")
    set_cell_background(t_hw.rows[0].cells[1], "F2F2F2")
    t_hw.rows[0].cells[0].paragraphs[0].runs[0].font.bold = True
    t_hw.rows[0].cells[1].paragraphs[0].runs[0].font.bold = True
    
    for idx, (comp, spec) in enumerate(hw_specs, start=1):
        t_hw.rows[idx].cells[0].text = comp
        t_hw.rows[idx].cells[1].text = spec
        for c in t_hw.rows[idx].cells:
            set_cell_margins(c, top=60, bottom=60, left=100, right=100)
    p_s13.paragraph_format.space_after = Pt(8)
    
    # 1.4 Software Specification
    p_s14 = doc.add_paragraph()
    r_s14 = p_s14.add_run("1.4 Software Specification")
    r_s14.bold = True
    r_s14.font.size = Pt(13)
    p_s14.paragraph_format.space_before = Pt(6)
    p_s14.paragraph_format.space_after = Pt(4)
    
    sw_specs = [
        ("Operating System", "Microsoft Windows 11 64-bit / Linux (Ubuntu 22.04 LTS)"),
        ("Programming Language", "Python 3.10+ (Executed and verified on Python 3.11.9)"),
        ("Core Deep Learning Framework", "PyTorch 2.15+ (CPU/CUDA compatible)"),
        ("Scientific Computing Libraries", "NumPy 2.4+, SciPy 1.17+, scikit-image 0.26+, OpenCV"),
        ("Quality Assurance & Testing", "Pytest 9.1+ (58/58 unit test suite covering 10 execution phases)"),
        ("Typesetting & Publishing", "Springer Nature LaTeX Template (sn-jnl / Overleaf), python-docx, python-pptx")
    ]
    t_sw = doc.add_table(rows=len(sw_specs)+1, cols=2)
    t_sw.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_sw.rows[0].cells[0].text = "Software Tool"
    t_sw.rows[0].cells[1].text = "Version & Functional Purpose"
    set_cell_background(t_sw.rows[0].cells[0], "F2F2F2")
    set_cell_background(t_sw.rows[0].cells[1], "F2F2F2")
    t_sw.rows[0].cells[0].paragraphs[0].runs[0].font.bold = True
    t_sw.rows[0].cells[1].paragraphs[0].runs[0].font.bold = True
    
    for idx, (soft, purpose) in enumerate(sw_specs, start=1):
        t_sw.rows[idx].cells[0].text = soft
        t_sw.rows[idx].cells[1].text = purpose
        for c in t_sw.rows[idx].cells:
            set_cell_margins(c, top=60, bottom=60, left=100, right=100)
    p_s14.paragraph_format.space_after = Pt(12)
    
    doc.add_page_break()
    
    # -------------------------------------------------------------
    # 2. LITERATURE SURVEY
    # -------------------------------------------------------------
    p_s2 = doc.add_paragraph()
    r_s2 = p_s2.add_run("2. LITERATURE SURVEY")
    r_s2.bold = True
    r_s2.font.size = Pt(14)
    r_s2.font.color.rgb = RGBColor(0, 32, 96)
    p_s2.paragraph_format.space_before = Pt(8)
    p_s2.paragraph_format.space_after = Pt(6)
    
    p_s21 = doc.add_paragraph()
    r_s21 = p_s21.add_run("2.1 Existing System")
    r_s21.bold = True
    r_s21.font.size = Pt(13)
    p_s21.paragraph_format.space_after = Pt(4)
    
    p_s21_body = doc.add_paragraph()
    p_s21_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_s21_body.paragraph_format.line_spacing = 1.3
    p_s21_body.paragraph_format.space_after = Pt(8)
    p_s21_body.add_run(
        "Existing CBIR systems fall into two major categories: classical handcrafted multi-feature frameworks and deep learning-based representations. "
        "Classical frameworks typically extract colour histograms, colour moments, Local Binary Patterns (LBP), or GLCM matrices, combining them "
        "via feature concatenation or static weighting formulas (e.g., 0.5/0.5 or 0.7/0.3). Feature-level concatenation suffers from dimensional imbalance, "
        "while static weighting fails to adapt when a query's discriminative information is skewed toward a single modality. "
        "Conversely, contemporary deep CNNs (such as ResNet-50 and EfficientNet) generate dense semantic representations that achieve high precision "
        "but require extensive floating-point compute and multi-kilobyte storage footprints per image. Furthermore, widely cited literature in the CBIR domain "
        "frequently suffers from data leakage (computing normalisation bounds across the combined test-train set) and gallery redundancy (near-duplicate images "
        "inflating top-rank retrieval accuracy)."
    )
    
    p_s22 = doc.add_paragraph()
    r_s22 = p_s22.add_run("2.2 Proposed System")
    r_s22.bold = True
    r_s22.font.size = Pt(13)
    p_s22.paragraph_format.space_after = Pt(4)
    
    p_s22_body = doc.add_paragraph()
    p_s22_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_s22_body.paragraph_format.line_spacing = 1.3
    p_s22_body.paragraph_format.space_after = Pt(8)
    p_s22_body.add_run(
        "The proposed system introduces an AI-weighted, distance-level fusion paradigm designed for ultra-compact edge deployment. "
        "Images are parameterized using a 13-D hybrid vector comprising nine statistical HSV colour moments and four Haralick GLCM texture moments. "
        "Normalisation bounds are strictly fitted exclusively to the gallery training split, preventing test data leakage. An ultra-tiny 378-parameter MLP "
        "(1.76 KB checkpoint) maps query features to dynamic weights [wC(q), wT(q)] constrained to the unit simplex via Softmax. "
        "Distances in colour and texture spaces are computed independently and linearly combined using the query's predicted weights. "
        "The system achieves 0.7100 Precision@10 and 0.4798 mAP on Corel-1K, statistically significantly outperforming classical LBP and static heuristic fusion, "
        "while executing feature extraction in 0.45 ms on CPU with a storage footprint of only 52 bytes per image."
    )
    
    p_s23 = doc.add_paragraph()
    r_s23 = p_s23.add_run("2.3 Literature Review Summary")
    r_s23.bold = True
    r_s23.font.size = Pt(13)
    p_s23.paragraph_format.space_after = Pt(4)
    
    lit_articles = [
        ("2022 [1]", "Dubey, S. R.", "PyTorch / TensorFlow", "Deep learning CNN survey in CBIR", "IEEE TCSVT", "mAP, Precision@K across Corel & Oxford"),
        ("2008 [2]", "Chun, Y. D. et al.", "C++ / MATLAB", "Multiresolution colour and BDIP/BVQP texture", "IEEE Trans. Multimedia", "Average Precision, Recall across natural images"),
        ("1995 [3]", "Stricker & Orengo", "C / UNIX", "3-order statistical colour moments (HSV)", "Proc. SPIE", "Retrieval precision, Illumination invariance"),
        ("2018 [4]", "Singh, A. et al.", "MATLAB", "Colour texture description with novel LBP", "Pattern Recognition", "Average Retrieval Precision (ARP), ARR"),
        ("2011 [5]", "Yue, J. et al.", "MATLAB", "Colour and texture fused features (HSV + GLCM)", "Math. Comput. Model.", "Precision, Recall curves"),
        ("2014 [6]", "Wang, X.-Y. et al.", "MATLAB / C", "Heuristic fixed integration (0.7 colour / 0.3 texture)", "Multimed. Tools Appl.", "Precision@10, Average Recall"),
        ("2023 [7]", "Akmal, A. et al.", "Python / Keras", "Automatic weight fusion using ANN (feature level)", "JOIV", "Precision@10, Accuracy percentage"),
        ("2025 [8]", "Naveen & Parvathi", "Python / PyTorch", "AI-powered multi-feature fusion framework", "Scientific Reports", "mAP, F1-Score, Retrieval Time"),
        ("2026 [9]", "Tyagi, S. et al.", "PyTorch / CUDA", "Hybrid colour, texture, and deep learning fusion", "Scientific Reports", "Top-K Precision, Computational Complexity")
    ]
    
    t_lit = doc.add_table(rows=len(lit_articles)+1, cols=6)
    t_lit.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_lit.autofit = False
    
    headers_lit = ["Year & Citation", "Article / Author", "Tools / Software", "Technique", "Source", "Evaluation Parameter"]
    widths_lit = [Inches(1.0), Inches(1.3), Inches(1.1), Inches(1.6), Inches(1.1), Inches(1.4)]
    
    for idx, h in enumerate(headers_lit):
        cell = t_lit.cell(0, idx)
        cell.text = h
        cell.width = widths_lit[idx]
        set_cell_background(cell, "F2F2F2")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.bold = True
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        
    for r_idx, row_data in enumerate(lit_articles, start=1):
        for c_idx, val in enumerate(row_data):
            cell = t_lit.cell(r_idx, c_idx)
            cell.text = val
            cell.width = widths_lit[c_idx]
            cell.paragraphs[0].runs[0].font.size = Pt(9.5)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            
    doc.add_page_break()
    
    # -------------------------------------------------------------
    # 3. PROBLEM FORMULATION
    # -------------------------------------------------------------
    p_s3 = doc.add_paragraph()
    r_s3 = p_s3.add_run("3. PROBLEM FORMULATION")
    r_s3.bold = True
    r_s3.font.size = Pt(14)
    r_s3.font.color.rgb = RGBColor(0, 32, 96)
    p_s3.paragraph_format.space_before = Pt(8)
    p_s3.paragraph_format.space_after = Pt(6)
    
    p_s3_body = doc.add_paragraph()
    p_s3_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_s3_body.paragraph_format.line_spacing = 1.3
    p_s3_body.paragraph_format.space_after = Pt(8)
    p_s3_body.add_run(
        "Let a database gallery of images be denoted by G = {d1, d2, ..., dN}, where each image di is represented by a normalized colour descriptor "
        "f_C(di) in R^9 and a normalized texture descriptor f_T(di) in R^4. Given a query image q represented by [f_C(q), f_T(q)], the retrieval objective "
        "is to output a ranked sequence of gallery images ordered by increasing overall distance D(q, di). Under distance-level fusion, the distance is formulated as:\n\n"
        "        D(q, di) = w_C(q) * D_C(q, di) + w_T(q) * D_T(q, di)\n\n"
        "where D_C(q, di) = ||f_C(q) - f_C(di)||_2 is the Euclidean colour distance, D_T(q, di) = ||f_T(q) - f_T(di)||_2 is the Euclidean texture distance, "
        "and [w_C(q), w_T(q)] are query-dependent metric weights. To preserve geometric validity and interpretability as a convex metric combination, "
        "the weights must satisfy the unit simplex constraints: w_C(q) >= 0, w_T(q) >= 0, and w_C(q) + w_T(q) = 1.0.\n\n"
        "The optimization problem consists of learning a mapping function g_theta: R^13 -> Delta^1 parameterized by neural weights theta that minimizes "
        "retrieval ranking violations. Using a triplet margin objective, given anchor a, positive p (same category), and negative n (different category):\n\n"
        "        L(theta) = max(0, D_theta(a, p) - D_theta(a, n) + m)\n\n"
        "where margin m = 0.15 enforces separation in the adaptive metric space. Solving this problem yields query-adaptive metric weights that "
        "maximize intra-class compact clustering while remaining ultra-lightweight and computationally deployable on edge devices."
    )
    
    # -------------------------------------------------------------
    # 4. OBJECTIVES
    # -------------------------------------------------------------
    p_s4 = doc.add_paragraph()
    r_s4 = p_s4.add_run("4. OBJECTIVES")
    r_s4.bold = True
    r_s4.font.size = Pt(14)
    r_s4.font.color.rgb = RGBColor(0, 32, 96)
    p_s4.paragraph_format.space_before = Pt(8)
    p_s4.paragraph_format.space_after = Pt(6)
    
    objs = [
        ("Objective 1: Compact Multi-Modal Feature Engineering",
         "Extract an ultra-compact 13-D hybrid visual representation combining nine low-order HSV statistical moments (mean, standard deviation, and cubic root of skewness across H, S, V channels) and four Gray-Level Co-occurrence Matrix (GLCM) Haralick texture moments (contrast, correlation, energy, homogeneity at d=1 averaged across 0°, 45°, 90°, 135°)."),
        ("Objective 2: Metric-Learning MLP for Query-Adaptive Weighting",
         "Design and train an ultra-lightweight 13–16–8–2 Multilayer Perceptron (378 trainable parameters, 1.76 KB checkpoint) under batch-hard triplet margin loss (m=0.15) with a Softmax output layer, dynamically predicting non-negative query weights on the unit simplex (wC + wT = 1.0)."),
        ("Objective 3: Rigorous Leakage-Controlled Benchmark Evaluation",
         "Evaluate the framework on the Wang Corel-1K dataset using a deterministic seed-42 protocol (100 queries, 899 gallery images) following 64-bit perceptual hashing duplicate removal (purging horses/761.jpg), with Min-Max normalisation fitted exclusively over gallery images to guarantee zero test leakage."),
        ("Objective 4: Inferential Statistical Testing and Edge Viability Profiling",
         "Conduct paired two-sided Wilcoxon signed-rank tests, 10,000 sign-flip permutations, and Holm-Bonferroni family-wise error rate corrections to formally separate statistically significant gains from descriptive improvements, benchmark against deep representations (ResNet-50, EfficientNet-B0, CLIP), and establish the operational transfer boundary on the 37-breed Oxford-IIIT Pet benchmark.")
    ]
    
    for title, desc in objs:
        p_o = doc.add_paragraph()
        p_o.paragraph_format.left_indent = Inches(0.2)
        p_o.paragraph_format.space_after = Pt(4)
        r_ot = p_o.add_run(f"• {title}: ")
        r_ot.bold = True
        p_o.add_run(desc)
        
    doc.add_page_break()
    
    # -------------------------------------------------------------
    # 5. METHODOLOGY
    # -------------------------------------------------------------
    p_s5 = doc.add_paragraph()
    r_s5 = p_s5.add_run("5. METHODOLOGY")
    r_s5.bold = True
    r_s5.font.size = Pt(14)
    r_s5.font.color.rgb = RGBColor(0, 32, 96)
    p_s5.paragraph_format.space_before = Pt(8)
    p_s5.paragraph_format.space_after = Pt(6)
    
    p_s5_body = doc.add_paragraph()
    p_s5_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_s5_body.paragraph_format.line_spacing = 1.3
    p_s5_body.paragraph_format.space_after = Pt(8)
    p_s5_body.add_run(
        "The proposed methodology comprises four sequential operational modules:\n\n"
        "1. Visual Feature Extraction: Given an RGB image, it is converted to HSV colour space to extract nine statistical moments (mean, std, "
        "and skewness per channel). Simultaneously, the image is converted to grayscale, quantized to 16 levels, and four GLCM Haralick moments "
        "(contrast, correlation, energy, homogeneity) are computed across four directions at offset d=1 and averaged. The resulting vectors are "
        "f_C in R^9 and f_T in R^4, forming composite vector x in R^13.\n\n"
        "2. Gallery-Only Normalisation: To eliminate test-set leakage, Min-Max feature bounds are computed strictly over the 700 gallery training images. "
        "Query features are scaled using these fixed gallery bounds: x_norm = (x - min_train) / (max_train - min_train + eps).\n\n"
        "3. Metric-Learning MLP Weight Inference: The normalized 13-D vector is passed through the 13–16–8–2 MLP. Layers 1 and 2 utilize LeakyReLU "
        "activations (negative slope 0.1), while the output layer applies Softmax to yield [wC(q), wT(q)] in Delta^1. The network is trained using Adam "
        "(lr=1e-3, weight decay=1e-5) under batch-hard triplet loss (batch size 64, margin 0.15) with early stopping on 199 validation images.\n\n"
        "4. Distance-Level Ranking Engine: Given query q and database images d in G, Euclidean distances D_C(q, d) and D_T(q, d) are computed independently. "
        "The overall distance is evaluated as D(q, d) = w_C(q)*D_C(q, d) + w_T(q)*D_T(q, d). Database images are sorted in ascending order of D(q, d), "
        "and top-K matches are returned."
    )
    
    # -------------------------------------------------------------
    # 6. EXPERIMENTAL SETUP
    # -------------------------------------------------------------
    p_s6 = doc.add_paragraph()
    r_s6 = p_s6.add_run("6. EXPERIMENTAL SETUP")
    r_s6.bold = True
    r_s6.font.size = Pt(14)
    r_s6.font.color.rgb = RGBColor(0, 32, 96)
    p_s6.paragraph_format.space_before = Pt(8)
    p_s6.paragraph_format.space_after = Pt(6)
    
    p_s6_body = doc.add_paragraph()
    p_s6_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_s6_body.paragraph_format.line_spacing = 1.3
    p_s6_body.paragraph_format.space_after = Pt(8)
    p_s6_body.add_run(
        "• Primary Benchmark: Wang Corel-1K dataset (1,000 images, 10 categories of 100 images each: Africa, Beaches, Buildings, Buses, Dinosaurs, "
        "Elephants, Flowers, Horses, Mountains, Food). Average perceptual hashing (64-bit) identified an exact duplicate pair (horses/760.jpg and horses/761.jpg, "
        "Hamming distance 0); horses/761.jpg was purged from the gallery.\n"
        "• Data Split Protocol: Deterministic random seed 42 partitions the dataset into 100 held-out queries (10 per category) and 899 gallery images "
        "(700 train, 199 validation). Zero query statistics are exposed during normalisation.\n"
        "• Cross-Dataset Transfer Benchmark: Oxford-IIIT Pet dataset (37 breeds, 10 queries per breed = 370 queries, 6,979 gallery images), evaluated zero-shot without retraining.\n"
        "• Evaluated Configurations: M1 (Colour Only, wC=1.0), M2 (Texture Only, wT=1.0), M3 (Equal Fusion 0.5/0.5), M4 (Heuristic Fixed 0.7/0.3), "
        "M5 (Proposed AI-Adaptive MLP), M6 (Fixed Mean Control, wC=0.578, wT=0.422), D1 (Uniform LBP 10-D), D2 (ResNet-50 2048-D), D3 (EfficientNet-B0 1280-D), D5 (CLIP ViT-B/32 512-D).\n"
        "• Evaluation Metrics: Precision@10, Recall@10, F1@10, and mean Average Precision (mAP) computed over the complete 899-image gallery ranking.\n"
        "• Statistical Validation: Paired two-sided Wilcoxon signed-rank tests across the 100 queries, 10,000 sign-flip permutation tests, Hodges–Lehmann median differences, "
        "Cohen's dz, 95% bootstrap confidence intervals (2,000 resamples), and Holm–Bonferroni step-down family-wise error rate control."
    )
    
    # -------------------------------------------------------------
    # 7. CONCLUSION
    # -------------------------------------------------------------
    p_s7 = doc.add_paragraph()
    r_s7 = p_s7.add_run("7. CONCLUSION")
    r_s7.bold = True
    r_s7.font.size = Pt(14)
    r_s7.font.color.rgb = RGBColor(0, 32, 96)
    p_s7.paragraph_format.space_before = Pt(8)
    p_s7.paragraph_format.space_after = Pt(6)
    
    p_s7_body = doc.add_paragraph()
    p_s7_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_s7_body.paragraph_format.line_spacing = 1.3
    p_s7_body.paragraph_format.space_after = Pt(8)
    p_s7_body.add_run(
        "This research develops and validates an ultra-lightweight, query-adaptive distance-level fusion framework for CBIR on resource-constrained edge hardware. "
        "By utilizing a 13-D hybrid descriptor and an ultra-tiny 378-parameter MLP (1.76 KB checkpoint), Proposed M5 attains Precision@10 of 0.7100 and mAP of 0.4798 "
        "on the Corel-1K benchmark, outperforming all handcrafted baselines. Paired non-parametric tests with Holm-Bonferroni correction verify that M5 is statistically "
        "significantly superior to classical LBP (p < 0.0001) and heuristic fixed fusion (p < 0.0001). Gains over equal fusion (M3) and mean control (M6) are descriptive "
        "and concentrated in the Flowers category (+0.2103 gain). Most importantly, M5 requires only 52 bytes per image descriptor (157.5x smaller than ResNet-50) and "
        "executes feature extraction in 0.45 ms on CPU (53.4x faster than ResNet-50), confirming its outstanding viability for battery-powered, real-time edge devices."
    )
    
    doc.add_page_break()
    
    # -------------------------------------------------------------
    # 8. TENTATIVE CHAPTER PLAN
    # -------------------------------------------------------------
    p_s8 = doc.add_paragraph()
    r_s8 = p_s8.add_run("8. TENTATIVE CHAPTER PLAN FOR THE PROPOSED WORK")
    r_s8.bold = True
    r_s8.font.size = Pt(14)
    r_s8.font.color.rgb = RGBColor(0, 32, 96)
    p_s8.paragraph_format.space_before = Pt(8)
    p_s8.paragraph_format.space_after = Pt(6)
    
    chap_plans = [
        ("CHAPTER 1: INTRODUCTION",
         "Background and motivation of CBIR; visual feature modalities (colour and texture); challenges in feature fusion and edge computing; identification of client and industrial need; contemporary issues; problem identification; task identification; project Gantt timeline; report structure overview."),
        ("CHAPTER 2: LITERATURE REVIEW",
         "Chronological evolution of CBIR (1990s to 2020s); bibliometric analysis of CBIR research; review of colour descriptors, texture patterns (LBP, GLCM), and hybrid fusion models; critical examination of deep CNNs and vision-language models; identification of research gaps (data leakage, static weighting, edge latency); problem definition, goals and objectives."),
        ("CHAPTER 3: OBJECTIVE",
         "Formal statement of project goals; specific measurable objectives covering descriptor engineering, metric-learning MLP architecture, leakage-controlled protocol, and inferential statistical testing."),
        ("CHAPTER 4: METHODOLOGIES",
         "Mathematical formulation of 9 HSV moments and 4 GLCM moments; gallery-only Min-Max normalisation; 13–16–8–2 MLP topology and LeakyReLU activations; Softmax unit-simplex constraint; batch-hard triplet loss formulation; distance-level Euclidean fusion equation; end-to-end algorithmic flowchart and pseudocode."),
        ("CHAPTER 5: EXPERIMENTAL SETUP AND RESULTS ANALYSIS",
         "Dataset description (Wang Corel-1K duplicate auditing, Oxford-IIIT Pet transfer benchmark); evaluation metrics (P@10, R@10, F1@10, mAP); benchmark results comparing M1 to M6; inferential statistical analysis (Wilcoxon tests, sign-flip permutations, Holm-Bonferroni correction); per-category decomposition and qualitative visual retrieval plates; deep baseline comparisons (ResNet-50, EfficientNet-B0, CLIP); gallery scaling sensitivity; cross-dataset transfer boundaries."),
        ("CHAPTER 6: CONCLUSION AND FUTURE SCOPE",
         "Comprehensive summary of project deliverables and achievements; critical deviation analysis from initial expectations; edge deployment trade-offs; practical applications in embedded vision; future research directions (dense local features, INT8 microcontroller deployment, cross-modal retrieval); complete references and Appendix User Manual.")
    ]
    
    for c_title, c_desc in chap_plans:
        p_cp = doc.add_paragraph()
        p_cp.paragraph_format.space_after = Pt(4)
        r_cpt = p_cp.add_run(f"{c_title}\n")
        r_cpt.bold = True
        r_cpt.font.size = Pt(12)
        r_cpt.font.color.rgb = RGBColor(0, 32, 96)
        r_cpd = p_cp.add_run(c_desc)
        r_cpd.font.size = Pt(11)
        p_cp.paragraph_format.line_spacing = 1.25
        
    doc.add_page_break()
    
    # -------------------------------------------------------------
    # 9. REFERENCES
    # -------------------------------------------------------------
    p_s9 = doc.add_paragraph()
    r_s9 = p_s9.add_run("9. REFERENCES")
    r_s9.bold = True
    r_s9.font.size = Pt(14)
    r_s9.font.color.rgb = RGBColor(0, 32, 96)
    p_s9.paragraph_format.space_before = Pt(8)
    p_s9.paragraph_format.space_after = Pt(8)
    
    refs = [
        "[1] S. R. Dubey, 'A decade survey of content based image retrieval using deep learning,' IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 5, pp. 2687–2704, 2022.",
        "[2] Y. D. Chun, N. C. Kim, and I. H. Jang, 'Content-based image retrieval using multiresolution color and texture features,' IEEE Transactions on Multimedia, vol. 10, no. 6, pp. 1073–1084, 2008.",
        "[3] M. A. Stricker and M. Orengo, 'Similarity of color images,' in Storage and Retrieval for Image and Video Databases III, Proc. SPIE, vol. 2420, pp. 381–392, 1995.",
        "[4] A. Singh, E. Walia, and K. P. Kaur, 'Color texture description with novel local binary patterns for effective image retrieval,' Pattern Recognition, vol. 76, pp. 50–66, 2018.",
        "[5] J. Yue, Z. Li, L. Liu, and Z. Fu, 'Content-based image retrieval using color and texture fused features,' Mathematical and Computer Modelling, vol. 54, no. 3–4, pp. 1121–1127, 2011.",
        "[6] S. Tyagi, P. Shukla, P. Singh, and H. Tekchandani, 'Enhanced content-based image retrieval via hybrid color, texture, and deep learning features,' Scientific Reports, vol. 16, p. 14888, 2026.",
        "[7] K. Naveen and R. M. S. Parvathi, 'AI powered multi-feature fusion framework for retrieving images using color, texture and shape descriptors,' Scientific Reports, vol. 15, p. 45601, 2025.",
        "[8] X.-Y. Wang, B.-B. Zhang, and H.-Y. Yang, 'Content-based image retrieval by integrating color and texture features,' Multimedia Tools and Applications, vol. 68, no. 3, pp. 545–569, 2014.",
        "[9] A. Akmal, R. Munir, and J. Santoso, 'Automatic weight of color, texture, and shape features in content-based image retrieval using artificial neural network,' JOIV: International Journal on Informatics Visualization, vol. 7, no. 3, pp. 665–672, 2023.",
        "[10] H. Bu, N. C. Kim, and S. H. Kim, 'Content-based image retrieval using a fusion of global and local features,' ETRI Journal, vol. 45, no. 3, pp. 505–518, 2023.",
        "[11] T. Ojala, M. Pietikäinen, and T. Mäenpää, 'Multiresolution gray-scale and rotation invariant texture classification with local binary patterns,' IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 7, pp. 971–987, 2002.",
        "[12] P. Liu, J. Guo, C. Wu, and D. Cai, 'Fusion of deep learning and compressed domain features for content-based image retrieval,' IEEE Transactions on Image Processing, vol. 26, no. 12, pp. 5706–5717, 2017.",
        "[13] S. Devulapalli, A. Potti, R. Krishnan, and M. S. Khan, 'Experimental evaluation of unsupervised image retrieval application using hybrid feature extraction by integrating deep learning and handcrafted techniques,' Materials Today: Proceedings, vol. 81, pp. 983–988, 2023.",
        "[14] M. K. Alsmadi, 'Content-based image retrieval using color, shape and texture descriptors and features,' Arabian Journal for Science and Engineering, vol. 45, pp. 7361–7379, 2020.",
        "[15] M. Subramanian, V. Lingamuthu, C. Venkatesan, and S. Perumal, 'Content-based image retrieval using colour, gray, advanced texture, shape features, and random forest classifier with optimized particle swarm optimization,' Computational Intelligence and Neuroscience, vol. 2022, Art. no. 7439465, 2022."
    ]
    
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.3)
        p_ref.paragraph_format.first_line_indent = Inches(-0.3)
        p_ref.paragraph_format.space_after = Pt(4)
        r_run = p_ref.add_run(r)
        r_run.font.size = Pt(10.5)
        
    doc.save(output_path)
    print(f"Research Synopsis successfully saved to: {output_path}")

if __name__ == "__main__":
    create_research_synopsis("Research_Synopsis_Final.docx")
