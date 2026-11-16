import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_project_proposal(output_path):
    doc = docx.Document()
    
    # Page setup
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
        sec.page_width = Inches(8.27)  # A4
        sec.page_height = Inches(11.69)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    
    # Header: DEPARTMENT OF APEX INSTITUTE OF TECHNOLOGY
    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_dept = p_dept.add_run("DEPARTMENT OF APEX INSTITUTE OF TECHNOLOGY")
    r_dept.font.name = 'Times New Roman'
    r_dept.font.size = Pt(16)
    r_dept.font.bold = True
    r_dept.font.color.rgb = RGBColor(180, 0, 0) # CU Crimson
    p_dept.paragraph_format.space_after = Pt(4)
    
    # Institution Subtitle
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("CHANDIGARH UNIVERSITY, GHARUAN, MOHALI - 140413, PUNJAB")
    r_inst.font.name = 'Times New Roman'
    r_inst.font.size = Pt(11)
    r_inst.font.bold = True
    p_inst.paragraph_format.space_after = Pt(14)
    
    # Title: PROJECT PROPOSAL
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("PROJECT PROPOSAL")
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.underline = True
    p_title.paragraph_format.space_after = Pt(18)
    
    # 1. Project Title: -
    p1 = doc.add_paragraph()
    r1_lbl = p1.add_run("1. Project Title: - ")
    r1_lbl.bold = True
    r1_lbl.font.size = Pt(13)
    r1_val = p1.add_run("Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and Texture Features")
    r1_val.bold = True
    r1_val.font.size = Pt(13)
    r1_val.font.color.rgb = RGBColor(0, 32, 96)
    p1.paragraph_format.space_after = Pt(14)
    
    # 2. Project Scope: - (Max 500 words)
    p2_hdr = doc.add_paragraph()
    r2_lbl = p2_hdr.add_run("2. Project Scope: - ")
    r2_lbl.bold = True
    r2_lbl.font.size = Pt(13)
    r2_sub = p2_hdr.add_run("(Max 500 words)")
    r2_sub.italic = True
    r2_sub.font.size = Pt(11)
    p2_hdr.paragraph_format.space_after = Pt(6)
    
    scope_text = (
        "Content-Based Image Retrieval (CBIR) is a fundamental computer vision task enabling automated search and discovery "
        "across massive visual databases based solely on visual content. While modern CBIR is increasingly dominated by heavy deep learning "
        "backbones (such as ResNet-50 and Vision-Language models), their substantial storage overhead (up to 8,192 bytes per image) and "
        "prohibitive feature extraction latencies (12 to 25 ms on CPU) create critical bottlenecks for battery-powered, resource-constrained edge hardware. "
        "Conversely, classical handcrafted multi-feature frameworks conventionally apply rigid, predetermined weighting coefficients "
        "(such as equal weighting or heuristic 0.7 colour / 0.3 texture ratios) to combine visual descriptors. Static weighting fundamentally "
        "ignores the visual reality that modality discriminability varies dramatically across diverse image queries (e.g., colour dominates floral queries, "
        "whereas spatial texture characterizes architectural structures).\n\n"
        "The primary scope of this project is to develop, validate, and deploy an ultra-lightweight, query-adaptive distance-level fusion CBIR framework "
        "engineered specifically for edge computing environments. The project encompasses four primary operational phases: (1) Designing an ultra-compact "
        "13-dimensional visual representation combining nine statistical HSV colour moments and four Gray-Level Co-occurrence Matrix (GLCM) Haralick texture moments; "
        "(2) Developing an ultra-tiny metric-learning Multi-Layer Perceptron (MLP) containing just 378 trainable parameters (1.76 KB disk checkpoint) "
        "trained via batch-hard triplet loss (margin m = 0.15) to predict Softmax-constrained metric weights on the unit simplex (wC + wT = 1.0); "
        "(3) Executing a leakage-controlled empirical benchmark on Wang's Corel-1K dataset under an audited protocol that purges near-duplicate images "
        "and fits Min-Max normalisation strictly over gallery training images; and (4) Conducting formal inferential statistical validation (paired Wilcoxon signed-rank tests, "
        "10,000 sign-flip permutations, and Holm-Bonferroni step-down correction) alongside zero-shot cross-dataset transfer boundary evaluation on the 37-breed Oxford-IIIT Pet benchmark.\n\n"
        "The anticipated project deliverables include: a complete, verifiable software prototype with 58/58 passing unit tests; empirical verification of superior retrieval fidelity "
        "(Precision@10 = 0.7100, mAP = 0.4798) outperforming all handcrafted baselines; formal proof of edge viability demonstrating 52 bytes/image descriptor storage "
        "(157.5x smaller than ResNet-50) and 0.45 ms CPU latency (53.4x faster than ResNet-50); and accepted academic research publication in the Springer LNNS series (ICAIN 2026, Paper ID 1567)."
    )
    
    p2_body = doc.add_paragraph()
    p2_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r2_txt = p2_body.add_run(scope_text)
    r2_txt.font.size = Pt(11.5)
    p2_body.paragraph_format.line_spacing = 1.25
    p2_body.paragraph_format.space_after = Pt(14)
    
    # 3. Requirements: -
    p3_hdr = doc.add_paragraph()
    r3_lbl = p3_hdr.add_run("3. Requirements: -")
    r3_lbl.bold = True
    r3_lbl.font.size = Pt(13)
    p3_hdr.paragraph_format.space_after = Pt(6)
    
    # Hardware Requirements
    p_hw = doc.add_paragraph()
    r_hw = p_hw.add_run("Hardware Requirements:")
    r_hw.bold = True
    r_hw.underline = True
    p_hw.paragraph_format.space_after = Pt(3)
    
    hw_items = [
        "Development Host Processor: Intel Core i7-14650HX (16 Cores, 24 Threads, 30 MB Cache, up to 5.2 GHz) / compatible x86_64 architecture.",
        "System Memory (RAM): 16 GB DDR5 RAM (5600 MHz) for batch caching and fast array manipulation.",
        "Primary Storage: 512 GB PCIe 4.0 NVMe M.2 SSD (minimum 5 GB allocated for image datasets, feature databases, and checkpoints).",
        "Target Edge Deployment Platform: Single-board edge computers and embedded micro-servers (e.g., Raspberry Pi 4/5, NVIDIA Jetson Nano, Intel NUC) requiring <= 512 MB available RAM and <= 50 MB total disk footprint."
    ]
    for idx, item in enumerate(hw_items):
        p_item = doc.add_paragraph()
        p_item.paragraph_format.left_indent = Inches(0.25)
        p_item.paragraph_format.space_after = Pt(2)
        r_num = p_item.add_run(f"{idx+1}. ")
        r_num.bold = True
        p_item.add_run(item)
        
    # Software Requirements
    p_sw = doc.add_paragraph()
    p_sw.paragraph_format.space_before = Pt(6)
    r_sw = p_sw.add_run("Software Requirements:")
    r_sw.bold = True
    r_sw.underline = True
    p_sw.paragraph_format.space_after = Pt(3)
    
    sw_items = [
        "Operating System: Windows 11 64-bit / Linux (Ubuntu 22.04 LTS / Debian 12) with PowerShell and bash shell environments.",
        "Core Programming Runtime: Python 3.10+ (specifically executed and validated on Python 3.11.9).",
        "Machine Learning & Mathematical Libraries: PyTorch 2.15+ (CPU/CUDA), NumPy 2.4+, SciPy 1.17+, scikit-image 0.26+, OpenCV (headless).",
        "Statistical Testing & Validation Suite: SciPy stats module (Wilcoxon signed-rank, permutation tests), Custom Bootstrap Resampling engine.",
        "Benchmarking & Quality Assurance: Pytest 9.1+ testing framework (58 automated unit tests verifying complete system integrity).",
        "Documentation & Typesetting Tools: Springer Nature LaTeX Template (sn-jnl / Overleaf), python-docx 1.2+, python-pptx 1.0+."
    ]
    for idx, item in enumerate(sw_items):
        p_item = doc.add_paragraph()
        p_item.paragraph_format.left_indent = Inches(0.25)
        p_item.paragraph_format.space_after = Pt(2)
        r_num = p_item.add_run(f"{idx+1}. ")
        r_num.bold = True
        p_item.add_run(item)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    
    # STUDENTS DETAILS
    p_sd = doc.add_paragraph()
    r_sd = p_sd.add_run("STUDENTS DETAILS")
    r_sd.bold = True
    r_sd.font.size = Pt(13)
    p_sd.paragraph_format.space_after = Pt(6)
    
    # Table 1: Students details (6 rows x 3 cols)
    t1 = doc.add_table(rows=6, cols=3)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False
    
    col_widths = [Inches(2.5), Inches(2.0), Inches(2.0)]
    for row in t1.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width
            
    # Headers
    headers = ["Name", "UID", "Signature"]
    for idx, text in enumerate(headers):
        cell = t1.cell(0, idx)
        cell.text = text
        set_cell_background(cell, "F2F2F2")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.bold = True
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        
    students_data = [
        ("Shalok Dadhwal", "23BCS10042", ""),
        ("Prakriti Sharma", "23BCS10038", ""),
        ("Sujal Jain", "23BCS10055", ""),
        ("Jaspreet Singh", "23BCS10080", ""),
        ("—", "—", "")
    ]
    
    for r_idx, (name, uid, sig) in enumerate(students_data, start=1):
        row = t1.rows[r_idx]
        row.cells[0].text = name
        row.cells[1].text = uid
        row.cells[2].text = sig
        for c_idx in range(3):
            cell = row.cells[c_idx]
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            p = cell.paragraphs[0]
            if c_idx == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # APPROVAL AND AUTHORITY TO PROCEED
    p_ap = doc.add_paragraph()
    r_ap = p_ap.add_run("APPROVAL AND AUTHORITY TO PROCEED")
    r_ap.bold = True
    r_ap.font.size = Pt(13)
    p_ap.paragraph_format.space_after = Pt(4)
    
    p_auth = doc.add_paragraph()
    r_auth = p_auth.add_run("We approve the project as described above, and authorize the team to proceed.")
    r_auth.font.size = Pt(11.5)
    r_auth.italic = True
    p_auth.paragraph_format.space_after = Pt(8)
    
    # Table 2: Approval (2 rows x 3 cols)
    t2 = doc.add_table(rows=2, cols=3)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False
    
    for row in t2.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width
            
    auth_headers = ["Name", "Title", "Signature (With Date)"]
    for idx, text in enumerate(auth_headers):
        cell = t2.cell(0, idx)
        cell.text = text
        set_cell_background(cell, "F2F2F2")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.bold = True
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        
    t2.cell(1, 0).text = "Prof. Dr. Vijay Bhardwaj"
    t2.cell(1, 1).text = "Project Supervisor / Professor"
    t2.cell(1, 2).text = ""
    for c_idx in range(3):
        cell = t2.cell(1, c_idx)
        set_cell_margins(cell, top=140, bottom=140, left=150, right=150)
        
    doc.save(output_path)
    print(f"Project Proposal successfully saved to: {output_path}")

if __name__ == "__main__":
    create_project_proposal("Project_Proposal_Final.docx")
