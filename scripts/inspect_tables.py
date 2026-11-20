import docx

doc = docx.Document('PJ Report CBIR.docx')

print('--- Tables in PJ Report CBIR.docx ---')
for t_idx, t in enumerate(doc.tables):
    txt0 = t.cell(0, 0).text if len(t.rows) > 0 and len(t.columns) > 0 else ''
    print(f'Table {t_idx+1}: {len(t.rows)}x{len(t.columns)}, cell(0,0)=\"{txt0[:30]}\"')
