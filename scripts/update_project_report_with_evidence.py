import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_margins(cell, top=40, bottom=40, left=60, right=60):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def update_report(doc_path, out_paths):
    doc = docx.Document(doc_path)
    
    # 1. Update Table 4 (List of Figures)
    t_lof = doc.tables[3] # Table 4
    new_figs = [
        ("Figure A.1: Official acceptance notification email from the 2026 International Conference on Artificial Intelligence and Networking (ICAIN 2026) confirming acceptance of Paper ID 1567 for publication in Springer Lecture Notes in Networks and Systems (LNNS).", "41"),
        ("Figure A.2: Official registration fee payment receipt confirming author registration for ICAIN 2026 (Paper ID 1567).", "42")
    ]
    for cap, pg in new_figs:
        row = t_lof.add_row()
        row.cells[0].width = Inches(5.4)
        row.cells[1].width = Inches(1.1)
        row.cells[0].text = cap
        row.cells[1].text = pg
        row.cells[0].paragraphs[0].runs[0].font.name = "Times New Roman"
        row.cells[0].paragraphs[0].runs[0].font.size = Pt(10)
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        row.cells[1].paragraphs[0].runs[0].font.name = "Times New Roman"
        row.cells[1].paragraphs[0].runs[0].font.size = Pt(10)
        for c in row.cells:
            set_cell_margins(c, top=40, bottom=40, left=60, right=60)
            
    # 2. Update Table 3 (Table of Contents)
    t_toc = doc.tables[2] # Table 3
    new_toc_items = [
        ("    A.1 ICAIN 2026 Official Paper Acceptance Letter", "41"),
        ("    A.2 ICAIN 2026 Conference Registration Payment Receipt", "42")
    ]
    for title, pg in new_toc_items:
        row = t_toc.add_row()
        row.cells[0].width = Inches(5.4)
        row.cells[1].width = Inches(1.1)
        row.cells[0].text = title
        row.cells[1].text = pg
        row.cells[0].paragraphs[0].runs[0].font.name = "Times New Roman"
        row.cells[0].paragraphs[0].runs[0].font.size = Pt(10.5)
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        row.cells[1].paragraphs[0].runs[0].font.name = "Times New Roman"
        row.cells[1].paragraphs[0].runs[0].font.size = Pt(10.5)
        for c in row.cells:
            set_cell_margins(c, top=35, bottom=35, left=60, right=60)

    # 3. Add documentary evidence subsections and images under ACHIEVEMENTS & PUBLICATIONS
    # Find paragraph of ACHIEVEMENTS & PUBLICATIONS
    ach_idx = -1
    for idx, p in enumerate(doc.paragraphs):
        if "ACHIEVEMENTS & PUBLICATIONS" in p.text:
            ach_idx = idx
            break
            
    print(f"Found ACHIEVEMENTS & PUBLICATIONS at paragraph {ach_idx}")
    
    # We will append the new dedicated sections at the end of the document
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_before = Pt(18)
    
    # Heading: Documentary Evidence of Research Outcome
    p_doc_h = doc.add_paragraph()
    p_doc_h.paragraph_format.space_before = Pt(14)
    p_doc_h.paragraph_format.space_after = Pt(6)
    r_doc_h = p_doc_h.add_run("Documentary Evidence of Research Outcome (Mandatory Evaluation Proof)")
    r_doc_h.font.name = "Times New Roman"
    r_doc_h.font.size = Pt(14)
    r_doc_h.font.bold = True
    r_doc_h.font.color.rgb = RGBColor(0, 32, 96)
    
    p_doc_intro = doc.add_paragraph()
    p_doc_intro.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_doc_intro.paragraph_format.line_spacing = 1.5
    p_doc_intro.paragraph_format.space_after = Pt(10)
    r_intro = p_doc_intro.add_run(
        "In compliance with the mandatory project evaluation guidelines issued by the Department of Computer Science & Engineering / "
        "Apex Institute of Technology, Chandigarh University for the Final External Viva-Voce examination on 21st November 2026, "
        "the following subsections provide verified documentary proof of research outcome fulfilment for Paper ID 1567: "
        "(1) the official conference paper acceptance notification from ICAIN 2026 (Springer LNNS Series), and (2) the official "
        "conference registration fee payment receipt (Rs. 16,500/-) confirming complete author registration."
    )
    r_intro.font.name = "Times New Roman"
    r_intro.font.size = Pt(12)

    # Subsection A.1: Official Acceptance Letter
    p_sub1 = doc.add_paragraph()
    p_sub1.paragraph_format.space_before = Pt(14)
    p_sub1.paragraph_format.space_after = Pt(4)
    r_sub1 = p_sub1.add_run("A.1 ICAIN 2026 Official Paper Acceptance Letter (Email Screenshot)")
    r_sub1.font.name = "Times New Roman"
    r_sub1.font.size = Pt(13)
    r_sub1.font.bold = True
    r_sub1.font.color.rgb = RGBColor(180, 0, 0) # CU Crimson
    
    p_desc1 = doc.add_paragraph()
    p_desc1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_desc1.paragraph_format.line_spacing = 1.3
    p_desc1.paragraph_format.space_after = Pt(8)
    r_desc1 = p_desc1.add_run(
        "Notification Details: The notification was officially communicated via email on Tuesday, October 27, 2026 at 11:27 PM "
        "from the ICAIN-2026 Conference Committee (icain.conf@gmail.com) to corresponding author Prakriti Sharma (sprakriti0193@gmail.com). "
        "The letter confirms that Paper ID '1567' titled 'Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and "
        "Texture Features' was selected among the top 20% of submissions for publication in the Proceedings of ICAIN 2026 — "
        "Springer Lecture Notes in Networks and Systems (LNNS) Series, indexed in Scopus, zbMATH, and DBLP. "
        "The letter also confirms the mandatory Springer camera-ready manuscript ceiling of 12 pages."
    )
    r_desc1.font.name = "Times New Roman"
    r_desc1.font.size = Pt(11.5)
    
    # Embed acceptance email image
    img_acc_path = "publication_figures/acceptance_email.png"
    if os.path.exists(img_acc_path):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_before = Pt(6)
        p_img1.paragraph_format.space_after = Pt(4)
        run_img1 = p_img1.add_run()
        run_img1.add_picture(img_acc_path, width=Inches(5.6))
        
        # Caption Figure A.1
        p_cap1 = doc.add_paragraph()
        p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap1.paragraph_format.space_after = Pt(16)
        r_lbl1 = p_cap1.add_run("Figure A.1: ")
        r_lbl1.font.name = "Times New Roman"
        r_lbl1.font.size = Pt(10)
        r_lbl1.font.bold = True
        r_txt1 = p_cap1.add_run("Official acceptance notification email from the 2026 International Conference on Artificial Intelligence and Networking (ICAIN 2026) confirming acceptance of Paper ID 1567 for publication in Springer Lecture Notes in Networks and Systems (LNNS).")
        r_txt1.font.name = "Times New Roman"
        r_txt1.font.size = Pt(10)
        r_txt1.font.bold = True
    else:
        print("Warning: acceptance_email.png not found!")

    doc.add_page_break()

    # Subsection A.2: Conference Registration Payment Receipt
    p_sub2 = doc.add_paragraph()
    p_sub2.paragraph_format.space_before = Pt(14)
    p_sub2.paragraph_format.space_after = Pt(4)
    r_sub2 = p_sub2.add_run("A.2 ICAIN 2026 Conference Registration Payment Receipt")
    r_sub2.font.name = "Times New Roman"
    r_sub2.font.size = Pt(13)
    r_sub2.font.bold = True
    r_sub2.font.color.rgb = RGBColor(180, 0, 0)
    
    p_desc2 = doc.add_paragraph()
    p_desc2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_desc2.paragraph_format.line_spacing = 1.3
    p_desc2.paragraph_format.space_after = Pt(8)
    r_desc2 = p_desc2.add_run(
        "Payment Details: The official conference registration fee was executed on October 30, 2026. "
        "The receipt confirms payment of Rs. 16,500/- (Net Amount Inclusive of Taxes) made by author Prakriti Sharma, "
        "Chandigarh University, for Paper ID 1567 ('Enhanced Content-Based Image Retrieval Using AI-Weighted Hybrid Colour and "
        "Texture Features'; Authors: Prakriti Sharma, Shalok Dadhwal, Sujal Jain, Vijay Bhardwaj, Jaspreet Singh). "
        "Payment was credited to the conference account (UI Consultants, City Union Bank, Rohini Branch, Account No: 510909010118897). "
        "This official receipt fulfils the institutional requirement for verified project outcome marks allocation."
    )
    r_desc2.font.name = "Times New Roman"
    r_desc2.font.size = Pt(11.5)
    
    # Embed registration receipt image
    img_rec_path = "publication_figures/registration_receipt.png"
    if os.path.exists(img_rec_path):
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.paragraph_format.space_before = Pt(6)
        p_img2.paragraph_format.space_after = Pt(4)
        run_img2 = p_img2.add_run()
        run_img2.add_picture(img_rec_path, width=Inches(5.6))
        
        # Caption Figure A.2
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap2.paragraph_format.space_after = Pt(16)
        r_lbl2 = p_cap2.add_run("Figure A.2: ")
        r_lbl2.font.name = "Times New Roman"
        r_lbl2.font.size = Pt(10)
        r_lbl2.font.bold = True
        r_txt2 = p_cap2.add_run("Official registration fee payment receipt confirming author registration for ICAIN 2026 (Paper ID 1567).")
        r_txt2.font.name = "Times New Roman"
        r_txt2.font.size = Pt(10)
        r_txt2.font.bold = True
    else:
        print("Warning: registration_receipt.png not found!")

    for pth in out_paths:
        doc.save(pth)
        print(f"Saved updated report with evidence to: {pth} (Size: {os.path.getsize(pth)} bytes)")

if __name__ == "__main__":
    update_report("PJ Report CBIR.docx", ["PJ Report CBIR.docx", "Final_Project_Report.docx"])
