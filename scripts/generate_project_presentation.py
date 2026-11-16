import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_path):
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333) # 16:9 widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # blank layout
    
    # Colors
    C_NAVY = RGBColor(0, 32, 96)       # Primary Deep Navy
    C_CRIMSON = RGBColor(180, 0, 0)    # CU Crimson Accent
    C_DARK = RGBColor(40, 40, 40)      # Body Dark Gray
    C_LIGHT_BG = RGBColor(248, 249, 250) # Light Gray Box
    C_WHITE = RGBColor(255, 255, 255)
    C_BORDER = RGBColor(210, 215, 220)
    
    def add_header(slide, title_text, category_text="CHANDIGARH UNIVERSITY | DEPARTMENT OF AIT-CSE"):
        # Top banner line
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = C_CRIMSON
        top_bar.line.color.rgb = C_CRIMSON
        
        # Category subtitle
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11.7), Inches(0.4))
        p_cat = tb_cat.text_frame.paragraphs[0]
        r_cat = p_cat.add_run()
        r_cat.text = category_text.upper()
        r_cat.font.name = "Times New Roman"
        r_cat.font.size = Pt(11)
        r_cat.font.bold = True
        r_cat.font.color.rgb = C_CRIMSON
        
        # Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.7))
        p_title = tb_title.text_frame.paragraphs[0]
        r_title = p_title.add_run()
        r_title.text = title_text
        r_title.font.name = "Times New Roman"
        r_title.font.size = Pt(24)
        r_title.font.bold = True
        r_title.font.color.rgb = C_NAVY
        
        # Bottom dividing line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = C_BORDER
        line.line.color.rgb = C_BORDER

    def add_card(slide, left, top, width, height, fill_color=C_LIGHT_BG, border_color=C_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = fill_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    # =============================================================
    # SLIDE 1: TITLE SLIDE
    # =============================================================
    s1 = prs.slides.add_slide(blank_layout)
    
    # Background accent
    top_bar1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.25))
    top_bar1.fill.solid()
    top_bar1.fill.fore_color.rgb = C_CRIMSON
    top_bar1.line.color.rgb = C_CRIMSON
    
    # Institution & Department
    tb_inst = s1.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.0))
    p1_inst = tb_inst.text_frame.paragraphs[0]
    p1_inst.alignment = PP_ALIGN.CENTER
    r1 = p1_inst.add_run()
    r1.text = "CHANDIGARH UNIVERSITY\n"
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(20)
    r1.font.bold = True
    r1.font.color.rgb = C_NAVY
    
    r2 = p1_inst.add_run()
    r2.text = "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING / APEX INSTITUTE OF TECHNOLOGY\n"
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(13)
    r2.font.bold = True
    r2.font.color.rgb = C_CRIMSON
    
    r3 = p1_inst.add_run()
    r3.text = "DISCOVER . LEARN . EMPOWER"
    r3.font.name = "Times New Roman"
    r3.font.size = Pt(10)
    r3.font.italic = True
    r3.font.color.rgb = C_DARK
    
    # Title Box
    add_card(s1, Inches(1.0), Inches(2.2), Inches(11.333), Inches(2.0), fill_color=RGBColor(240, 244, 250), border_color=C_NAVY)
    tb_t = s1.shapes.add_textbox(Inches(1.2), Inches(2.35), Inches(10.933), Inches(1.7))
    tb_t.text_frame.word_wrap = True
    p_t = tb_t.text_frame.paragraphs[0]
    p_t.alignment = PP_ALIGN.CENTER
    rt = p_t.add_run()
    rt.text = "Enhanced Content-Based Image Retrieval Using\nAI-Weighted Hybrid Colour and Texture Features"
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(26)
    rt.font.bold = True
    rt.font.color.rgb = C_NAVY
    
    p_sub = tb_t.text_frame.add_paragraph()
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.space_before = Pt(8)
    rsub = p_sub.add_run()
    rsub.text = "Bachelor of Engineering in Computer Science and Engineering (AI & ML) — Final Project Evaluation"
    rsub.font.name = "Times New Roman"
    rsub.font.size = Pt(13)
    rsub.font.bold = True
    rsub.font.color.rgb = C_CRIMSON
    
    # Details Cards (Students & Supervisor)
    add_card(s1, Inches(1.0), Inches(4.5), Inches(5.5), Inches(2.3))
    tb_std = s1.shapes.add_textbox(Inches(1.2), Inches(4.6), Inches(5.1), Inches(2.1))
    p_std_h = tb_std.text_frame.paragraphs[0]
    r_std_h = p_std_h.add_run()
    r_std_h.text = "Submitted by:"
    r_std_h.font.name = "Times New Roman"
    r_std_h.font.size = Pt(14)
    r_std_h.font.bold = True
    r_std_h.font.color.rgb = C_NAVY
    
    students_info = [
        "Shalok Dadhwal (UID: 23BCS10042)",
        "Prakriti Sharma (UID: 23BCS10038)",
        "Sujal Jain (UID: 23BCS10055)",
        "Jaspreet Singh (UID: 23BCS10080)"
    ]
    for s_info in students_info:
        p_si = tb_std.text_frame.add_paragraph()
        p_si.space_before = Pt(3)
        r_si = p_si.add_run()
        r_si.text = f"•  {s_info}"
        r_si.font.name = "Times New Roman"
        r_si.font.size = Pt(12)
        r_si.font.color.rgb = C_DARK
        
    add_card(s1, Inches(6.833), Inches(4.5), Inches(5.5), Inches(2.3))
    tb_sup = s1.shapes.add_textbox(Inches(7.033), Inches(4.6), Inches(5.1), Inches(2.1))
    p_sup_h = tb_sup.text_frame.paragraphs[0]
    r_sup_h = p_sup_h.add_run()
    r_sup_h.text = "Under the Supervision of:"
    r_sup_h.font.name = "Times New Roman"
    r_sup_h.font.size = Pt(14)
    r_sup_h.font.bold = True
    r_sup_h.font.color.rgb = C_NAVY
    
    p_sup_n = tb_sup.text_frame.add_paragraph()
    p_sup_n.space_before = Pt(8)
    r_sup_n = p_sup_n.add_run()
    r_sup_n.text = "Prof. Dr. Vijay Bhardwaj"
    r_sup_n.font.name = "Times New Roman"
    r_sup_n.font.size = Pt(15)
    r_sup_n.font.bold = True
    r_sup_n.font.color.rgb = C_CRIMSON
    
    p_sup_d = tb_sup.text_frame.add_paragraph()
    p_sup_d.space_before = Pt(4)
    r_sup_d = p_sup_d.add_run()
    r_sup_d.text = "Professor, Department of Computer Science & Engineering\nApex Institute of Technology, Chandigarh University\n\nDate: 21st November 2026"
    r_sup_d.font.name = "Times New Roman"
    r_sup_d.font.size = Pt(11.5)
    r_sup_d.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 2: OUTLINE
    # =============================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Presentation Outline")
    
    outline_items = [
        ("01", "Introduction to Project", "Context, CBIR significance, and multi-modal visual cues"),
        ("02", "Problem Formulation", "Shortcomings of static weighting and dimensional dominance"),
        ("03", "Objectives of the Work", "Four core engineering, architectural, and validation goals"),
        ("04", "Methodology & Architecture", "13-D moments, gallery normalisation, MLP, and distance fusion"),
        ("05", "Results and Outputs", "Corel-1K benchmark, statistical testing, and category analysis"),
        ("06", "Edge Viability & Transfer", "Microsecond CPU latency profiling and Oxford Pet domain boundary"),
        ("07", "Conclusion & Future Scope", "Key scientific takeaways, achievements, and roadmap ahead"),
        ("08", "References", "Formal academic citations in 16pt Times New Roman")
    ]
    
    for idx, (num, title, desc) in enumerate(outline_items):
        col = idx % 2
        row = idx // 2
        left = Inches(1.0 + col * 5.8)
        top = Inches(1.65 + row * 1.3)
        add_card(s2, left, top, Inches(5.5), Inches(1.15))
        
        tb = s2.shapes.add_textbox(left + Inches(0.15), top + Inches(0.1), Inches(5.2), Inches(0.95))
        p = tb.text_frame.paragraphs[0]
        r_n = p.add_run()
        r_n.text = f"[{num}]  {title}\n"
        r_n.font.name = "Times New Roman"
        r_n.font.size = Pt(14)
        r_n.font.bold = True
        r_n.font.color.rgb = C_NAVY
        
        r_d = p.add_run()
        r_d.text = desc
        r_d.font.name = "Times New Roman"
        r_d.font.size = Pt(11)
        r_d.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 3: INTRODUCTION TO PROJECT (Part 1)
    # =============================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Introduction: Content-Based Image Retrieval")
    
    # 3 Column Cards
    intro_cols = [
        ("The CBIR Paradigm", "Intrinsic Visual Search", 
         "• Searches massive image repositories using visual features (colour, texture, shape) rather than text keywords.\n\n"
         "• Vital for applications lacking annotations: medical diagnostics, autonomous surveillance, satellite remote sensing, and e-commerce.", C_NAVY),
        ("Complementary Modalities", "Colour & Texture Synergy",
         "• Colour Moments: Capture chromatic distributions invariant to scale, translation, and rotation (HSV space).\n\n"
         "• Texture Features: Capture structural spatial dependencies, granularity, and coarseness via GLCM Haralick moments.\n\n"
         "• Fusing modalities provides robust semantic discrimination.", C_CRIMSON),
        ("The Weighting Dilemma", "The Core Engineering Gap",
         "• How should colour and texture distances be combined?\n\n"
         "• Traditional systems fix weights statically (e.g. 50/50 or 70/30) across every query.\n\n"
         "• In reality, feature importance shifts dynamically from query to query.", C_NAVY)
    ]
    for idx, (title, sub, body, color) in enumerate(intro_cols):
        left = Inches(0.8 + idx * 3.95)
        add_card(s3, left, Inches(1.6), Inches(3.75), Inches(5.3))
        tb = s3.shapes.add_textbox(left + Inches(0.2), Inches(1.8), Inches(3.35), Inches(4.9))
        tb.text_frame.word_wrap = True
        
        p_h = tb.text_frame.paragraphs[0]
        r_h = p_h.add_run()
        r_h.text = title + "\n"
        r_h.font.name = "Times New Roman"
        r_h.font.size = Pt(16)
        r_h.font.bold = True
        r_h.font.color.rgb = color
        
        p_sub = tb.text_frame.add_paragraph()
        p_sub.space_before = Pt(2)
        r_sub = p_sub.add_run()
        r_sub.text = sub + "\n\n"
        r_sub.font.name = "Times New Roman"
        r_sub.font.size = Pt(11)
        r_sub.font.bold = True
        r_sub.font.color.rgb = C_DARK
        
        p_b = tb.text_frame.add_paragraph()
        r_b = p_b.add_run()
        r_b.text = body
        r_b.font.name = "Times New Roman"
        r_b.font.size = Pt(12)
        r_b.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 4: INTRODUCTION TO PROJECT (Part 2 - Edge Challenges)
    # =============================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Contemporary Edge Challenges: Deep Learning vs Efficiency")
    
    add_card(s4, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_dl = s4.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tb_dl.text_frame.word_wrap = True
    p_dl_h = tb_dl.text_frame.paragraphs[0]
    r_dl_h = p_dl_h.add_run()
    r_dl_h.text = "Deep Learning Bottlenecks on Edge Devices\n"
    r_dl_h.font.name = "Times New Roman"
    r_dl_h.font.size = Pt(16)
    r_dl_h.font.bold = True
    r_dl_h.font.color.rgb = C_CRIMSON
    
    dl_points = (
        "• Prohibitive Storage Footprint:\n"
        "  - ResNet-50 requires 8,192 bytes per image (2048 float32 dimensions).\n"
        "  - Storing 100,000 images requires > 800 MB RAM solely for feature vectors.\n\n"
        "• Extreme Extraction Latency on CPU:\n"
        "  - ResNet-50 incurs 24.04 ms per query on CPU.\n"
        "  - CLIP ViT-B/32 incurs 16.60 ms per query on CPU.\n"
        "  - Too sluggish for low-power ARM microcontrollers and battery devices.\n\n"
        "• High Energy & Hardware Costs:\n"
        "  - Requires expensive GPU accelerators ($500–$2000).\n"
        "  - Infeasible for edge cameras, robotics, and embedded micro-servers."
    )
    p_dl_b = tb_dl.text_frame.add_paragraph()
    p_dl_b.space_before = Pt(6)
    r_dl_b = p_dl_b.add_run()
    r_dl_b.text = dl_points
    r_dl_b.font.name = "Times New Roman"
    r_dl_b.font.size = Pt(12)
    r_dl_b.font.color.rgb = C_DARK
    
    add_card(s4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_prop = s4.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tb_prop.text_frame.word_wrap = True
    p_pr_h = tb_prop.text_frame.paragraphs[0]
    r_pr_h = p_pr_h.add_run()
    r_pr_h.text = "Our Proposed Ultra-Low-Latency Edge Paradigm\n"
    r_pr_h.font.name = "Times New Roman"
    r_pr_h.font.size = Pt(16)
    r_pr_h.font.bold = True
    r_pr_h.font.color.rgb = C_NAVY
    
    prop_points = (
        "• Ultra-Compact Handcrafted Descriptor:\n"
        "  - Exactly 13 dimensions (52 bytes per image).\n"
        "  - 157.5x smaller storage than ResNet-50; 39.4x smaller than CLIP.\n\n"
        "• Sub-Millisecond CPU Latency:\n"
        "  - Full feature extraction executes in 0.45 ms on CPU (53.4x faster).\n"
        "  - Distance ranking across 7,000 images takes < 0.5 ms.\n\n"
        "• Tiny Neural Checkpoint:\n"
        "  - 378-parameter MLP serialized to just 1.76 KB.\n"
        "  - Fits entirely into on-chip cache of microcontrollers (e.g. ESP32, STM32).\n\n"
        "• Zero Cloud Dependency: Completely private, on-device offline indexing."
    )
    p_pr_b = tb_prop.text_frame.add_paragraph()
    p_pr_b.space_before = Pt(6)
    r_pr_b = p_pr_b.add_run()
    r_pr_b.text = prop_points
    r_pr_b.font.name = "Times New Roman"
    r_pr_b.font.size = Pt(12)
    r_pr_b.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 5: PROBLEM FORMULATION
    # =============================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Problem Formulation: The Flaws of Static Fusion")
    
    add_card(s5, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.3))
    tb_f1 = s5.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.333), Inches(2.1))
    p_f1 = tb_f1.text_frame.paragraphs[0]
    r_f1_h = p_f1.add_run()
    r_f1_h.text = "1. Dimensional Dominance in Feature Concatenation\n"
    r_f1_h.font.name = "Times New Roman"
    r_f1_h.font.size = Pt(15)
    r_f1_h.font.bold = True
    r_f1_h.font.color.rgb = C_NAVY
    
    p_f1_b = tb_f1.text_frame.add_paragraph()
    p_f1_b.space_before = Pt(4)
    r_f1_b = p_f1_b.add_run()
    r_f1_b.text = (
        "In classical feature concatenation, vectors x = [f_C^T, f_T^T]^T are stacked into a shared space. Differences in dimensionality "
        "(9-D colour vs 4-D texture) and numerical variances cause the higher-dimensional modality to dominate Euclidean distances, "
        "obscuring critical texture signals. Distance-level fusion preserves the native metric topology of each modality."
    )
    r_f1_b.font.name = "Times New Roman"
    r_f1_b.font.size = Pt(12)
    r_f1_b.font.color.rgb = C_DARK
    
    add_card(s5, Inches(0.8), Inches(4.2), Inches(11.733), Inches(2.7))
    tb_f2 = s5.shapes.add_textbox(Inches(1.0), Inches(4.3), Inches(11.333), Inches(2.5))
    p_f2 = tb_f2.text_frame.paragraphs[0]
    r_f2_h = p_f2.add_run()
    r_f2_h.text = "2. Mathematical Formulation of Adaptive Distance Fusion\n"
    r_f2_h.font.name = "Times New Roman"
    r_f2_h.font.size = Pt(15)
    r_f2_h.font.bold = True
    r_f2_h.font.color.rgb = C_CRIMSON
    
    math_text = (
        "• Modality Distances:  D_C(q, d) = ||f_C(q) - f_C(d)||_2  and  D_T(q, d) = ||f_T(q) - f_T(d)||_2\n"
        "• Query-Adaptive Distance:  D(q, d) = w_C(q) * D_C(q, d) + w_T(q) * D_T(q, d)\n"
        "• Geometric Unit Simplex Constraint:  w_C(q) >= 0,  w_T(q) >= 0,  and  w_C(q) + w_T(q) = 1.0\n\n"
        "Rigid heuristic weights (e.g. M4: 0.7 C / 0.3 T) degrade retrieval fidelity (mAP 0.4337 vs 0.4650 for equal fusion M3). "
        "Dynamic query-adaptive weighting eliminates static compromises and adapts metric priority to each individual query."
    )
    p_f2_b = tb_f2.text_frame.add_paragraph()
    p_f2_b.space_before = Pt(4)
    r_f2_b = p_f2_b.add_run()
    r_f2_b.text = math_text
    r_f2_b.font.name = "Times New Roman"
    r_f2_b.font.size = Pt(12)
    r_f2_b.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 6: OBJECTIVES OF THE WORK
    # =============================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Objectives of the Work")
    
    objectives_list = [
        ("Objective 1", "Ultra-Compact Multi-Modal Feature Extraction",
         "Formulate an ultra-compact 13-dimensional handcrafted visual descriptor combining nine statistical HSV colour moments and four Haralick GLCM texture moments (exactly 52 bytes/image)."),
        ("Objective 2", "Metric-Learning MLP Query-Adaptive Weighting",
         "Design an ultra-tiny 13–16–8–2 MLP (378 trainable parameters, 1.76 KB checkpoint) trained under batch-hard triplet loss (m=0.15) to predict Softmax-constrained metric weights on the unit simplex."),
        ("Objective 3", "Leakage-Controlled Duplicate-Aware Benchmark Protocol",
         "Audit Wang's Corel-1K via 64-bit perceptual hashing, purge duplicate horses/761.jpg, fit Min-Max normalisation bounds exclusively over gallery training images, and evaluate across 100 held-out queries."),
        ("Objective 4", "Inferential Statistical Validation & Edge Profiling",
         "Conduct paired Wilcoxon signed-rank tests with Holm–Bonferroni correction, benchmark against deep representations (ResNet-50, EfficientNet-B0, CLIP), and establish transfer limits on Oxford-IIIT Pet.")
    ]
    for idx, (tag, title, desc) in enumerate(objectives_list):
        top = Inches(1.6 + idx * 1.35)
        add_card(s6, Inches(0.8), top, Inches(11.733), Inches(1.2))
        tb = s6.shapes.add_textbox(Inches(1.0), top + Inches(0.1), Inches(11.333), Inches(1.0))
        tb.text_frame.word_wrap = True
        
        p = tb.text_frame.paragraphs[0]
        r_tag = p.add_run()
        r_tag.text = f"{tag}: "
        r_tag.font.name = "Times New Roman"
        r_tag.font.size = Pt(14)
        r_tag.font.bold = True
        r_tag.font.color.rgb = C_CRIMSON
        
        r_title = p.add_run()
        r_title.text = f"{title}\n"
        r_title.font.name = "Times New Roman"
        r_title.font.size = Pt(14)
        r_title.font.bold = True
        r_title.font.color.rgb = C_NAVY
        
        p_d = tb.text_frame.add_paragraph()
        p_d.space_before = Pt(2)
        r_d = p_d.add_run()
        r_d.text = desc
        r_d.font.name = "Times New Roman"
        r_d.font.size = Pt(11.5)
        r_d.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 7: METHODOLOGY - FEATURE EXTRACTION
    # =============================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Methodology: 13-D Visual Feature Extraction")
    
    add_card(s7, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_c = s7.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tb_c.text_frame.word_wrap = True
    p_ch = tb_c.text_frame.paragraphs[0]
    r_ch = p_ch.add_run()
    r_ch.text = "Colour Modality: 9-D HSV Moments\n"
    r_ch.font.name = "Times New Roman"
    r_ch.font.size = Pt(16)
    r_ch.font.bold = True
    r_ch.font.color.rgb = C_NAVY
    
    c_desc = (
        "• Converted to HSV (Hue, Saturation, Value) space.\n\n"
        "• Three low-order central moments extracted per channel:\n"
        "  1. Mean (μ_c): Average chromatic intensity.\n"
        "  2. Standard Deviation (σ_c): Chromatic spread/variance.\n"
        "  3. Skewness (s_c): Asymmetry of colour distribution (cubic root).\n\n"
        "• Vector formulation:  f_C = [μ_H, σ_H, s_H, μ_S, σ_S, s_S, μ_V, σ_V, s_V]^T in R^9\n\n"
        "• Advantages: Highly compact, illumination-tolerant, and invariant to scale, rotation, and translation."
    )
    p_cb = tb_c.text_frame.add_paragraph()
    p_cb.space_before = Pt(6)
    r_cb = p_cb.add_run()
    r_cb.text = c_desc
    r_cb.font.name = "Times New Roman"
    r_cb.font.size = Pt(12)
    r_cb.font.color.rgb = C_DARK
    
    add_card(s7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_t = s7.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tb_t.text_frame.word_wrap = True
    p_th = tb_t.text_frame.paragraphs[0]
    r_th = p_th.add_run()
    r_th.text = "Texture Modality: 4-D GLCM Haralick Moments\n"
    r_th.font.name = "Times New Roman"
    r_th.font.size = Pt(16)
    r_th.font.bold = True
    r_th.font.color.rgb = C_CRIMSON
    
    t_desc = (
        "• Converted to grayscale and quantized into 16 gray levels.\n\n"
        "• GLCM computed at displacement d=1 across four directions (0°, 45°, 90°, 135°) and averaged for rotational invariance.\n\n"
        "• Four Haralick moments computed:\n"
        "  1. Contrast: Intensity variation between neighbouring pixels.\n"
        "  2. Correlation: Linear gray-level dependencies.\n"
        "  3. Energy (ASM): Uniformity / orderliness of texture.\n"
        "  4. Homogeneity (IDM): Closeness of element distribution.\n\n"
        "• Vector formulation:  f_T = [Contrast, Correlation, Energy, Homogeneity]^T in R^4\n\n"
        "• Combined Descriptor: Exactly 13 dimensions (52 bytes)."
    )
    p_tb = tb_t.text_frame.add_paragraph()
    p_tb.space_before = Pt(6)
    r_tb = p_tb.add_run()
    r_tb.text = t_desc
    r_tb.font.name = "Times New Roman"
    r_tb.font.size = Pt(12)
    r_tb.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 8: METHODOLOGY - MLP ARCHITECTURE & METRIC LEARNING
    # =============================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Methodology: Metric-Learning MLP Architecture")
    
    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_mlp = s8.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tb_mlp.text_frame.word_wrap = True
    p_mlph = tb_mlp.text_frame.paragraphs[0]
    r_mlph = p_mlph.add_run()
    r_mlph.text = "MLP Topology & Layer Specifications\n"
    r_mlph.font.name = "Times New Roman"
    r_mlph.font.size = Pt(16)
    r_mlph.font.bold = True
    r_mlph.font.color.rgb = C_NAVY
    
    mlp_spec = (
        "• Input Layer: 13 neurons (normalized descriptor x_hat in [0, 1]^13)\n\n"
        "• Hidden Layer 1: 16 neurons with LeakyReLU (slope 0.1)\n"
        "  - Parameters: 13 x 16 + 16 = 224 parameters\n\n"
        "• Hidden Layer 2: 8 neurons with LeakyReLU (slope 0.1)\n"
        "  - Parameters: 16 x 8 + 8 = 136 parameters\n\n"
        "• Output Layer: 2 neurons with Softmax activation\n"
        "  - Parameters: 8 x 2 + 2 = 18 parameters\n\n"
        "• Total Trainable Parameters: Exactly 378 parameters!\n\n"
        "• Serialized Checkpoint: 1.76 KB disk space (SRAM deployable)."
    )
    p_mlpb = tb_mlp.text_frame.add_paragraph()
    p_mlpb.space_before = Pt(6)
    r_mlpb = p_mlpb.add_run()
    r_mlpb.text = mlp_spec
    r_mlpb.font.name = "Times New Roman"
    r_mlpb.font.size = Pt(12)
    r_mlpb.font.color.rgb = C_DARK
    
    add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_loss = s8.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tb_loss.text_frame.word_wrap = True
    p_lh = tb_loss.text_frame.paragraphs[0]
    r_lh = p_lh.add_run()
    r_lh.text = "Batch-Hard Triplet Margin Optimization\n"
    r_lh.font.name = "Times New Roman"
    r_lh.font.size = Pt(16)
    r_lh.font.bold = True
    r_lh.font.color.rgb = C_CRIMSON
    
    loss_desc = (
        "• Objective Function:\n"
        "  L_triplet = (1/|B|) Σ max(0, D(a, p) - D(a, n) + m)\n\n"
        "• Triplet Mining Strategy:\n"
        "  - Mini-batch size: 64 images.\n"
        "  - Anchor (a): Query image.\n"
        "  - Hardest Positive (p): Intra-class image maximizing D(a, p).\n"
        "  - Hardest Negative (n): Inter-class image minimizing D(a, n).\n\n"
        "• Margin Hyperparameter: m = 0.15\n\n"
        "• Optimization Details:\n"
        "  - Adam Optimizer (learning rate 10^-3, weight decay 10^-5).\n"
        "  - Early stopping evaluated on 199 validation images (optimal epoch 8)."
    )
    p_lb = tb_loss.text_frame.add_paragraph()
    p_lb.space_before = Pt(6)
    r_lb = p_lb.add_run()
    r_lb.text = loss_desc
    r_lb.font.name = "Times New Roman"
    r_lb.font.size = Pt(12)
    r_lb.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 9: SYSTEM ARCHITECTURE DIAGRAM
    # =============================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "System Architecture: End-to-End Operational Pipeline")
    
    if os.path.exists("publication_figures/fig_system_architecture.png"):
        s9.shapes.add_picture("publication_figures/fig_system_architecture.png", Inches(1.2), Inches(1.6), width=Inches(10.933))
        
    tb_cap9 = s9.shapes.add_textbox(Inches(1.2), Inches(6.2), Inches(10.933), Inches(0.9))
    p_cap9 = tb_cap9.text_frame.paragraphs[0]
    p_cap9.alignment = PP_ALIGN.CENTER
    r_c9 = p_cap9.add_run()
    r_c9.text = "Four-Stage Operational Pipeline: (1) Duplicate Auditing (64-bit pHash) → (2) 13-D Feature Extraction with Gallery Normalisation → (3) Metric-Learning MLP Dynamic Weight Inference → (4) Query-Adaptive Euclidean Distance Ranking."
    r_c9.font.name = "Times New Roman"
    r_c9.font.size = Pt(12)
    r_c9.font.bold = True
    r_c9.font.color.rgb = C_NAVY

    # =============================================================
    # SLIDE 10: EXPERIMENTAL SETUP & LEAKAGE CONTROL
    # =============================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Experimental Setup & Leakage-Controlled Protocol")
    
    setup_boxes = [
        ("Wang Corel-1K Benchmark", "Primary Evaluation Dataset",
         "• 1,000 natural images across 10 classes (100 images/class):\n"
         "  Africa, Beaches, Buildings, Buses, Dinosaurs, Elephants, Flowers, Horses, Mountains, Food.\n\n"
         "• Duplicate Check (64-bit pHash): Identified exact duplicate pair (horses/760.jpg & horses/761.jpg, Hamming dist 0).\n\n"
         "• Purging: Removed horses/761.jpg from gallery to eliminate artificial rank-1 bias.", C_NAVY),
        ("Deterministic Splitting", "Zero-Leakage Guarantee",
         "• Deterministic Random Seed: 42.\n\n"
         "• 100 Held-Out Queries: Exactly 10 queries per category.\n\n"
         "• 899-Image Gallery: 700 train (70/class) and 199 validation (20/class, 19 for Horses).\n\n"
         "• Gallery-Only Normalisation: Bounds fitted strictly on gallery training set. Zero query exposure during training.", C_CRIMSON),
        ("Oxford-IIIT Pet Benchmark", "Zero-Shot Cross-Dataset Transfer",
         "• 7,390 natural images across 37 cat and dog breeds.\n\n"
         "• Evaluation Split: 10 queries per breed (370 queries) and 6,979 gallery images.\n\n"
         "• Zero-Shot Application: Corel-trained MLP applied directly without retraining to establish domain boundary.", C_NAVY)
    ]
    for idx, (title, sub, body, color) in enumerate(setup_boxes):
        left = Inches(0.8 + idx * 3.95)
        add_card(s10, left, Inches(1.6), Inches(3.75), Inches(5.3))
        tb = s10.shapes.add_textbox(left + Inches(0.2), Inches(1.8), Inches(3.35), Inches(4.9))
        tb.text_frame.word_wrap = True
        
        p_h = tb.text_frame.paragraphs[0]
        r_h = p_h.add_run()
        r_h.text = title + "\n"
        r_h.font.name = "Times New Roman"
        r_h.font.size = Pt(16)
        r_h.font.bold = True
        r_h.font.color.rgb = color
        
        p_s = tb.text_frame.add_paragraph()
        p_s.space_before = Pt(2)
        r_s = p_s.add_run()
        r_s.text = sub + "\n\n"
        r_s.font.name = "Times New Roman"
        r_s.font.size = Pt(11)
        r_s.font.bold = True
        r_s.font.color.rgb = C_DARK
        
        p_b = tb.text_frame.add_paragraph()
        r_b = p_b.add_run()
        r_b.text = body
        r_b.font.name = "Times New Roman"
        r_b.font.size = Pt(12)
        r_b.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 11: RESULTS & OUTPUTS - MAIN RETRIEVAL PERFORMANCE
    # =============================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Results: Handcrafted Configurations on Corel-1K")
    
    # Left: Results Table
    add_card(s11, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.3))
    tb_t11 = s11.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.6), Inches(4.9))
    p_t11 = tb_t11.text_frame.paragraphs[0]
    r_t11 = p_t11.add_run()
    r_t11.text = "Benchmark Retrieval Summary (899 Gallery)\n"
    r_t11.font.name = "Times New Roman"
    r_t11.font.size = Pt(15)
    r_t11.font.bold = True
    r_t11.font.color.rgb = C_NAVY
    
    # Table inside slide
    t11 = s11.shapes.add_table(7, 5, Inches(1.0), Inches(2.3), Inches(5.6), Inches(2.8)).table
    t11_hdrs = ["Model", "Modality", "P@10", "F1@10", "mAP"]
    t11_data = [
        ("M1", "Colour Only (9-D)", "0.6190", "0.1240", "0.4045"),
        ("M2", "Texture Only (4-D)", "0.4680", "0.0937", "0.3508"),
        ("M3", "Equal Fusion (0.5/0.5)", "0.6990", "0.1400", "0.4650"),
        ("M4", "Heuristic (0.7/0.3)", "0.6730", "0.1348", "0.4337"),
        ("M5", "Proposed Adaptive", "0.7100", "0.1422", "0.4798"),
        ("M6", "Mean Weight Control", "0.6960", "0.1394", "0.4605")
    ]
    for c_idx, h in enumerate(t11_hdrs):
        cell = t11.cell(0, c_idx)
        cell.text = h
        cell.text_frame.paragraphs[0].runs[0].font.bold = True
        cell.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(230, 235, 245)
    for r_idx, row_data in enumerate(t11_data, start=1):
        for c_idx, val in enumerate(row_data):
            cell = t11.cell(r_idx, c_idx)
            cell.text = val
            cell.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
            if "M5" in row_data[0]:
                cell.text_frame.paragraphs[0].runs[0].font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(255, 240, 240)
                
    # Text notes below table
    tb_notes11 = s11.shapes.add_textbox(Inches(1.0), Inches(5.2), Inches(5.6), Inches(1.6))
    p_n11 = tb_notes11.text_frame.paragraphs[0]
    r_n11 = p_n11.add_run()
    r_n11.text = (
        "Key Insights:\n"
        "• Proposed M5 achieved the highest score across every metric.\n"
        "• Fusion (M3) outperforms colour alone by +15.0% relative gain.\n"
        "• Heuristic fixed fusion (M4) degrades mAP to 0.4337 (worse than equal fusion).\n"
        "• Proposed M5 outperforms equal fusion (+3.17%) and mean control (+4.2%)."
    )
    r_n11.font.name = "Times New Roman"
    r_n11.font.size = Pt(11)
    r_n11.font.color.rgb = C_DARK
    
    # Right: Chart Image
    add_card(s11, Inches(7.0), Inches(1.6), Inches(5.5), Inches(5.3))
    if os.path.exists("publication_figures/fig_main_performance_comparison.png"):
        s11.shapes.add_picture("publication_figures/fig_main_performance_comparison.png", Inches(7.2), Inches(2.0), width=Inches(5.1))

    # =============================================================
    # SLIDE 12: RESULTS & OUTPUTS - INFERENTIAL STATISTICAL VALIDATION
    # =============================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Results: Rigorous Inferential Statistical Validation")
    
    add_card(s12, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    tb_stat = s12.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.9))
    p_sh = tb_stat.text_frame.paragraphs[0]
    r_sh = p_sh.add_run()
    r_sh.text = "Paired Statistical Contrasts Across 100 Queries (Holm-Bonferroni FWER Control)\n"
    r_sh.font.name = "Times New Roman"
    r_sh.font.size = Pt(16)
    r_sh.font.bold = True
    r_sh.font.color.rgb = C_NAVY
    
    t12 = s12.shapes.add_table(8, 7, Inches(1.0), Inches(2.3), Inches(11.333), Inches(2.8)).table
    t12_hdrs = ["Paired Contrast", "Mean Δ", "Median Δ", "95% Bootstrap CI", "Wilcoxon W", "Holm p-value", "W / T / L"]
    t12_data = [
        ("M5 vs CLIP ViT-B/32 (D5)", "-0.4687", "-0.5263", "[-0.5197, -0.4177]", "24.0", "0.0000*", "2 / 2 / 96"),
        ("M5 vs ResNet-50 (D2)", "-0.4305", "-0.4882", "[-0.4830, -0.3781]", "102.0", "0.0000*", "8 / 0 / 92"),
        ("M5 vs EfficientNet-B0 (D3)", "-0.4081", "-0.4611", "[-0.4580, -0.3582]", "71.0", "0.0000*", "7 / 0 / 93"),
        ("M5 vs Uniform LBP (D1)", "+0.1064", "+0.1005", "[+0.0703, +0.1426]", "838.0", "0.0000*", "81 / 0 / 19"),
        ("M5 vs Fixed Heuristic (M4)", "+0.0461", "+0.0441", "[+0.0238, +0.0687]", "943.0", "0.0001*", "76 / 0 / 24"),
        ("M5 vs Equal Fusion (M3)", "+0.0148", "+0.0002", "[+0.0017, +0.0278]", "1972.0", "0.6519", "51 / 6 / 43"),
        ("M5 vs Mean Control (M6)", "+0.0193", "+0.0002", "[+0.0041, +0.0344]", "2353.0", "0.6519", "53 / 0 / 47")
    ]
    for c_idx, h in enumerate(t12_hdrs):
        cell = t12.cell(0, c_idx)
        cell.text = h
        cell.text_frame.paragraphs[0].runs[0].font.bold = True
        cell.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(230, 235, 245)
    for r_idx, row_data in enumerate(t12_data, start=1):
        for c_idx, val in enumerate(row_data):
            cell = t12.cell(r_idx, c_idx)
            cell.text = val
            cell.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
            if "*" in val:
                cell.text_frame.paragraphs[0].runs[0].font.bold = True
                
    tb_exp12 = s12.shapes.add_textbox(Inches(1.0), Inches(5.3), Inches(11.333), Inches(1.5))
    p_exp12 = tb_exp12.text_frame.paragraphs[0]
    r_exp12 = p_exp12.add_run()
    r_exp12.text = (
        "Rigorous Scientific Conclusions:\n"
        "• Confirmed Statistical Superiority: M5 is statistically significantly superior to classical LBP (p < 0.0001, 81 wins vs 19 losses) "
        "and heuristic fixed fusion M4 (p = 0.0001, 76 wins vs 24 losses).\n"
        "• Descriptive Classification: Against equal fusion M3 and mean control M6, differences do not reach statistical significance after "
        "Holm-Bonferroni correction (Holm p = 0.6519). Gains over static hybrid models are formally classified as descriptive."
    )
    r_exp12.font.name = "Times New Roman"
    r_exp12.font.size = Pt(12)
    r_exp12.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 13: RESULTS - CATEGORY DECOMPOSITION & FLOWERS EFFECT
    # =============================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Results: Category Breakdown & The Flowers Driver Effect")
    
    add_card(s13, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_f_txt = s13.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
    p_fh = tb_f_txt.text_frame.paragraphs[0]
    r_fh = p_fh.add_run()
    r_fh.text = "Structural Retrieval Phenomena\n"
    r_fh.font.name = "Times New Roman"
    r_fh.font.size = Pt(16)
    r_fh.font.bold = True
    r_fh.font.color.rgb = C_NAVY
    
    f_text = (
        "1. Uniform Ceiling in Dinosaurs:\n"
        "   - All hybrid models attain mAP = 0.9934.\n"
        "   - Driven by stark silhouette foreground-background contrast.\n\n"
        "2. The Flowers Driver Effect (Core Empirical Discovery):\n"
        "   - M5 recorded an mAP of 0.7410 in Flowers vs 0.5307 for M6 and M3 (+0.2103 mAP gain).\n"
        "   - The MLP dynamically assigns high colour priority (mean w_C = 0.812) to floral queries.\n"
        "   - When Flowers is excluded, the mean paired difference across the remaining 90 queries is slightly negative (-0.00197).\n\n"
        "Takeaway: Adaptive distance weighting delivers decisive gains when a category's optimal feature balance sharply diverges from the global mean."
    )
    p_fb = tb_f_txt.text_frame.add_paragraph()
    p_fb.space_before = Pt(6)
    r_fb = p_fb.add_run()
    r_fb.text = f_text
    r_fb.font.name = "Times New Roman"
    r_fb.font.size = Pt(12)
    r_fb.font.color.rgb = C_DARK
    
    add_card(s13, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    if os.path.exists("publication_figures/fig_category_performance.png"):
        s13.shapes.add_picture("publication_figures/fig_category_performance.png", Inches(7.0), Inches(1.9), width=Inches(5.3))

    # =============================================================
    # SLIDE 14: RESULTS - QUALITATIVE RETRIEVAL EXAMPLES
    # =============================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Results: Qualitative Visual Retrieval Proof")
    
    if os.path.exists("publication_figures/fig_qualitative_retrieval_examples.png"):
        s14.shapes.add_picture("publication_figures/fig_qualitative_retrieval_examples.png", Inches(1.0), Inches(1.6), width=Inches(11.333))
        
    tb_q14 = s14.shapes.add_textbox(Inches(1.0), Inches(6.1), Inches(11.333), Inches(1.1))
    p_q14 = tb_q14.text_frame.paragraphs[0]
    p_q14.alignment = PP_ALIGN.CENTER
    r_q14 = p_q14.add_run()
    r_q14.text = (
        "Qualitative Visual Retrieval Proof: Top Row: For a colorful floral query, the MLP dynamically predicts high colour priority (wC = 0.812, wT = 0.188), "
        "retrieving five correct floral matches. Bottom Row: For an architectural query, the network shifts priority to structural texture (wT = 0.638, wC = 0.362), "
        "retrieving relevant stone facades despite varying illumination."
    )
    r_q14.font.name = "Times New Roman"
    r_q14.font.size = Pt(11.5)
    r_q14.font.bold = True
    r_q14.font.color.rgb = C_NAVY

    # =============================================================
    # SLIDE 15: RESULTS - EDGE VIABILITY BENCHMARKING
    # =============================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Edge Viability: Handcrafted vs Deep Representations")
    
    add_card(s15, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.3))
    t15 = s15.shapes.add_table(8, 5, Inches(1.0), Inches(1.8), Inches(5.6), Inches(3.2)).table
    t15_hdrs = ["Model", "Dim", "mAP", "Storage / Img", "CPU Latency"]
    t15_data = [
        ("D1 (LBP)", "10", "0.3734", "40 Bytes", "0.38 ms"),
        ("M3 (Equal)", "13", "0.4650", "52 Bytes", "0.45 ms"),
        ("M6 (Control)", "13", "0.4605", "52 Bytes", "0.45 ms"),
        ("Proposed M5", "13", "0.4798", "52 Bytes", "0.45 ms"),
        ("D3 (EffNet)", "1280", "0.8879", "5,120 Bytes", "12.18 ms"),
        ("D2 (ResNet)", "2048", "0.9103", "8,192 Bytes", "24.04 ms"),
        ("D5 (CLIP)", "512", "0.9485", "2,048 Bytes", "16.60 ms")
    ]
    for c_idx, h in enumerate(t15_hdrs):
        cell = t15.cell(0, c_idx)
        cell.text = h
        cell.text_frame.paragraphs[0].runs[0].font.bold = True
        cell.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(230, 235, 245)
    for r_idx, row_data in enumerate(t15_data, start=1):
        for c_idx, val in enumerate(row_data):
            cell = t15.cell(r_idx, c_idx)
            cell.text = val
            cell.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
            if "Proposed M5" in row_data[0]:
                cell.text_frame.paragraphs[0].runs[0].font.bold = True
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(255, 240, 240)
                
    tb_n15 = s15.shapes.add_textbox(Inches(1.0), Inches(5.2), Inches(5.6), Inches(1.6))
    p_n15 = tb_n15.text_frame.paragraphs[0]
    r_n15 = p_n15.add_run()
    r_n15.text = (
        "Practical Engineering Niche:\n"
        "• 157.5x Storage Reduction vs ResNet-50 (52 B vs 8,192 B).\n"
        "• 53.4x Faster Feature Extraction on CPU (0.45 ms vs 24.04 ms).\n"
        "• 1.76 KB Checkpoint enables on-chip persistence in microcontrollers."
    )
    r_n15.font.name = "Times New Roman"
    r_n15.font.size = Pt(11.5)
    r_n15.font.bold = True
    r_n15.font.color.rgb = C_NAVY
    
    add_card(s15, Inches(7.0), Inches(1.6), Inches(5.5), Inches(5.3))
    if os.path.exists("publication_figures/fig_deep_learning_comparison.png"):
        s15.shapes.add_picture("publication_figures/fig_deep_learning_comparison.png", Inches(7.2), Inches(1.9), width=Inches(5.1))

    # =============================================================
    # SLIDE 16: GALLERY SCALING & TRANSFER LIMITS ON OXFORD PET
    # =============================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Scaling Sensitivity & Transfer Limits on Oxford-IIIT Pet")
    
    add_card(s16, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_sc = s16.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
    p_sch = tb_sc.text_frame.paragraphs[0]
    r_sch = p_sch.add_run()
    r_sch.text = "Gallery Scaling Sensitivity (Phase 7)\n"
    r_sch.font.name = "Times New Roman"
    r_sch.font.size = Pt(16)
    r_sch.font.bold = True
    r_sch.font.color.rgb = C_NAVY
    
    sc_text = (
        "• Evaluated across nested stratified gallery subsets:\n"
        "  - 25% (N=220), 50% (N=449), 75% (N=679), 100% (N=899).\n\n"
        "• Mathematical Metric Phenomenon:\n"
        "  - Recall@10 increases from 0.0790 to 0.2882 as gallery shrinks.\n"
        "  - F1@10 increases from 0.1422 to 0.3662.\n\n"
        "• Explanation: Direct consequence of a smaller relevance denominator (|Rel| approx 22 vs approx 90), not improved retrieval fidelity.\n\n"
        "• Ranking consistency remains robust across gallery sizes."
    )
    p_scb = tb_sc.text_frame.add_paragraph()
    p_scb.space_before = Pt(6)
    r_scb = p_scb.add_run()
    r_scb.text = sc_text
    r_scb.font.name = "Times New Roman"
    r_scb.font.size = Pt(12)
    r_scb.font.color.rgb = C_DARK
    
    add_card(s16, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_ox = s16.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    p_oxh = tb_ox.text_frame.paragraphs[0]
    r_oxh = p_oxh.add_run()
    r_oxh.text = "Zero-Shot Transfer on Oxford Pet (Phase 9)\n"
    r_oxh.font.name = "Times New Roman"
    r_oxh.font.size = Pt(16)
    r_oxh.font.bold = True
    r_oxh.font.color.rgb = C_CRIMSON
    
    ox_text = (
        "• Benchmark: Oxford-IIIT Pet (37 breeds, 370 queries, 6,979 gallery).\n\n"
        "• Zero-Shot Retrieval Results:\n"
        "  - Proposed M5: mAP = 0.0333 (Precision@10 = 0.0546)\n"
        "  - Equal Fusion M3: mAP = 0.0333\n"
        "  - Mean Control M6: mAP = 0.0332\n"
        "  - Random Chance Baseline: 1/37 approx 0.0270\n\n"
        "• Fundamental Domain Boundary Identified:\n"
        "  - Global statistical moments cannot disambiguate fine-grained biological breeds without local keypoint descriptors.\n"
        "  - Proves the importance of honest academic reporting of failure modes!"
    )
    p_oxb = tb_ox.text_frame.add_paragraph()
    p_oxb.space_before = Pt(6)
    r_oxb = p_oxb.add_run()
    r_oxb.text = ox_text
    r_oxb.font.name = "Times New Roman"
    r_oxb.font.size = Pt(12)
    r_oxb.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 17: ACHIEVEMENTS & OUTCOMES
    # =============================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Project Outcomes & Research Achievements")
    
    add_card(s17, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_pub = s17.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.3), Inches(4.9))
    p_pbh = tb_pub.text_frame.paragraphs[0]
    r_pbh = p_pbh.add_run()
    r_pbh.text = "International Conference Research Publication\n"
    r_pbh.font.name = "Times New Roman"
    r_pbh.font.size = Pt(16)
    r_pbh.font.bold = True
    r_pbh.font.color.rgb = C_CRIMSON
    
    pub_info = (
        "• Title: Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features\n\n"
        "• Conference: 2026 International Conference on Artificial Intelligence and Networking (ICAIN 2026)\n\n"
        "• Publication Series: Springer Lecture Notes in Networks and Systems (LNNS)\n\n"
        "• Paper ID: 1567\n\n"
        "• Authors: Shalok Dadhwal, Prakriti Sharma, Sujal Jain, Vijay Bhardwaj, Jaspreet Singh\n\n"
        "• Status: Formally Accepted for Oral Presentation & Springer publication\n\n"
        "• Indexing: Scopus, EI Compendex, INSPEC, SCImago."
    )
    p_pbb = tb_pub.text_frame.add_paragraph()
    p_pbb.space_before = Pt(6)
    r_pbb = p_pbb.add_run()
    r_pbb.text = pub_info
    r_pbb.font.name = "Times New Roman"
    r_pbb.font.size = Pt(12)
    r_pbb.font.color.rgb = C_DARK
    
    add_card(s17, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb_soft = s17.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    p_sfh = tb_soft.text_frame.paragraphs[0]
    r_sfh = p_sfh.add_run()
    r_sfh.text = "Software Deliverables & Quality Assurance\n"
    r_sfh.font.name = "Times New Roman"
    r_sfh.font.size = Pt(16)
    r_sfh.font.bold = True
    r_sfh.font.color.rgb = C_NAVY
    
    soft_info = (
        "• Public Open-Source Repository:\n"
        "  https://github.com/Shalokexe/CBIR-Research-Paper.git\n\n"
        "• Complete Automated Testing Suite:\n"
        "  - 58/58 unit tests passing via Pytest (100% test pass rate).\n"
        "  - Full test coverage of feature extraction, MLP training, and ranking.\n\n"
        "• Full University Documentation Package:\n"
        "  - Project Proposal (CU AIT format)\n"
        "  - Research Synopsis (CU Format)\n"
        "  - Spiral-Bound Final Project Report (with User Manual)\n"
        "  - Professional Evaluation Presentation (PPTX)"
    )
    p_sfb = tb_soft.text_frame.add_paragraph()
    p_sfb.space_before = Pt(6)
    r_sfb = p_sfb.add_run()
    r_sfb.text = soft_info
    r_sfb.font.name = "Times New Roman"
    r_sfb.font.size = Pt(12)
    r_sfb.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 18: CONCLUSION
    # =============================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Conclusion")
    
    add_card(s18, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    tb_c18 = s18.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.9))
    tb_c18.text_frame.word_wrap = True
    
    p_ch18 = tb_c18.text_frame.paragraphs[0]
    r_ch18 = p_ch18.add_run()
    r_ch18.text = "Summary of Project Contributions & Impact\n"
    r_ch18.font.name = "Times New Roman"
    r_ch18.font.size = Pt(18)
    r_ch18.font.bold = True
    r_ch18.font.color.rgb = C_NAVY
    
    conc_points = (
        "1. Dynamic Distance Fusion Solves Static Flaws:\n"
        "   Eliminates rigid global coefficients by dynamically adjusting modality weights per query image subject to the unit simplex constraint.\n\n"
        "2. Superior Handcrafted Retrieval Fidelity:\n"
        "   Proposed M5 attains P@10 of 0.7100 and mAP of 0.4798 on Corel-1K, outperforming colour-only, texture-only, equal fusion, heuristic fixed, and mean control.\n\n"
        "3. Rigorous Inferential Statistical Validation:\n"
        "   Paired Wilcoxon signed-rank tests with Holm-Bonferroni correction confirm statistically significant gains over classical LBP (p < 0.0001) and heuristic fusion (p < 0.0001).\n\n"
        "4. Practical Edge Computing Breakthrough:\n"
        "   Requires only 52 bytes per image descriptor (157.5x smaller than ResNet-50) and extracts in 0.45 ms on CPU (53.4x faster than ResNet-50) with a 1.76 KB checkpoint.\n\n"
        "5. Complete University Deliverables Ready:\n"
        "   Full documentation, 58/58 unit tests passing, functional prototype demonstration, and accepted Springer LNNS publication."
    )
    p_cb18 = tb_c18.text_frame.add_paragraph()
    p_cb18.space_before = Pt(8)
    r_cb18 = p_cb18.add_run()
    r_cb18.text = conc_points
    r_cb18.font.name = "Times New Roman"
    r_cb18.font.size = Pt(13)
    r_cb18.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 19: FUTURE SCOPE
    # =============================================================
    s19 = prs.slides.add_slide(blank_layout)
    add_header(s19, "Future Scope & Technological Roadmap")
    
    future_cards = [
        ("01", "Dense Local Patch Descriptors",
         "• Augment global statistical moments with ultra-compact binary local descriptors (e.g. ORB, BRIEF).\n\n"
         "• Address fine-grained biological morphology limitations identified on the Oxford-IIIT Pet dataset.", C_NAVY),
        ("02", "INT8 Microcontroller Quantization",
         "• Quantize the 378-parameter MLP weights from float32 to INT8.\n\n"
         "• Compress checkpoint from 1.76 KB to < 450 bytes for deployment on ARM Cortex-M microcontrollers.", C_CRIMSON),
        ("03", "Cross-Modal Text-to-Image Edge Search",
         "• Build a lightweight projection head mapping natural language queries into the compact 13-D moment space.\n\n"
         "• Enable zero-cloud edge keyword search for smart camera networks.", C_NAVY)
    ]
    for idx, (num, title, body, color) in enumerate(future_cards):
        left = Inches(0.8 + idx * 3.95)
        add_card(s19, left, Inches(1.6), Inches(3.75), Inches(5.3))
        tb = s19.shapes.add_textbox(left + Inches(0.2), Inches(1.8), Inches(3.35), Inches(4.9))
        tb.text_frame.word_wrap = True
        
        p = tb.text_frame.paragraphs[0]
        r_n = p.add_run()
        r_n.text = f"[{num}]  {title}\n\n"
        r_n.font.name = "Times New Roman"
        r_n.font.size = Pt(16)
        r_n.font.bold = True
        r_n.font.color.rgb = color
        
        p_b = tb.text_frame.add_paragraph()
        r_b = p_b.add_run()
        r_b.text = body
        r_b.font.name = "Times New Roman"
        r_b.font.size = Pt(12.5)
        r_b.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 20: REFERENCES (Part 1 - Times New Roman 16pt)
    # =============================================================
    s20 = prs.slides.add_slide(blank_layout)
    add_header(s20, "References (Part 1)")
    
    add_card(s20, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    tb_r20 = s20.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.9))
    tb_r20.text_frame.word_wrap = True
    
    refs_p1 = [
        "[1] S. R. Dubey, 'A decade survey of content based image retrieval using deep learning,' IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 5, pp. 2687–2704, 2022.",
        "[2] Y. D. Chun, N. C. Kim, and I. H. Jang, 'Content-based image retrieval using multiresolution color and texture features,' IEEE Transactions on Multimedia, vol. 10, no. 6, pp. 1073–1084, 2008.",
        "[3] M. A. Stricker and M. Orengo, 'Similarity of color images,' in Storage and Retrieval for Image and Video Databases III, Proc. SPIE, vol. 2420, pp. 381–392, 1995.",
        "[4] A. Singh, E. Walia, and K. P. Kaur, 'Color texture description with novel local binary patterns for effective image retrieval,' Pattern Recognition, vol. 76, pp. 50–66, 2018."
    ]
    for idx, r in enumerate(refs_p1):
        p = tb_r20.text_frame.add_paragraph() if idx > 0 else tb_r20.text_frame.paragraphs[0]
        p.space_after = Pt(14)
        run = p.add_run()
        run.text = r
        run.font.name = "Times New Roman"
        run.font.size = Pt(16)  # Exactly 16pt as required by guidelines!
        run.font.color.rgb = C_DARK

    # =============================================================
    # SLIDE 21: REFERENCES (Part 2 - Times New Roman 16pt)
    # =============================================================
    s21 = prs.slides.add_slide(blank_layout)
    add_header(s21, "References (Part 2)")
    
    add_card(s21, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.3))
    tb_r21 = s21.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.9))
    tb_r21.text_frame.word_wrap = True
    
    refs_p2 = [
        "[5] J. Yue, Z. Li, L. Liu, and Z. Fu, 'Content-based image retrieval using color and texture fused features,' Mathematical and Computer Modelling, vol. 54, no. 3–4, pp. 1121–1127, 2011.",
        "[6] S. Tyagi, P. Shukla, P. Singh, and H. Tekchandani, 'Enhanced content-based image retrieval via hybrid color, texture, and deep learning features,' Scientific Reports, vol. 16, p. 14888, 2026.",
        "[7] K. Naveen and R. M. S. Parvathi, 'AI powered multi-feature fusion framework for retrieving images using color, texture and shape descriptors,' Scientific Reports, vol. 15, p. 45601, 2025.",
        "[8] X.-Y. Wang, B.-B. Zhang, and H.-Y. Yang, 'Content-based image retrieval by integrating color and texture features,' Multimedia Tools and Applications, vol. 68, no. 3, pp. 545–569, 2014."
    ]
    for idx, r in enumerate(refs_p2):
        p = tb_r21.text_frame.add_paragraph() if idx > 0 else tb_r21.text_frame.paragraphs[0]
        p.space_after = Pt(14)
        run = p.add_run()
        run.text = r
        run.font.name = "Times New Roman"
        run.font.size = Pt(16)  # Exactly 16pt as required by guidelines!
        run.font.color.rgb = C_DARK

    prs.save(output_path)
    print(f"Presentation successfully created and saved to: {output_path}")

if __name__ == "__main__":
    create_presentation("Research_Project_Presentation.pptx")
