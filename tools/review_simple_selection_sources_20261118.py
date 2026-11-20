"""Read original PDF/PPTX pages behind changed evidence; never alter corpus."""
import json
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import fitz
from tools.validate_simple_selection_20261118 import OUT, exp, check_frozen
from tools.validate_selection_factorial import sha, write
from local_full_text_search.rag.content_identity import canonical_content


def main():
    check_frozen()
    audit = json.loads((OUT/'source_bound_audit.json').read_text(encoding='utf8'))
    targets = []
    for n in audit['new_negative_quotes']:
        for m in n['matches']:
            targets.append(dict(id=n['id'], entry=m['entry'], quote=n['gold_annotation']['quote'], role='new_hard_negative'))
    rows = {r['id']:r for r in exp.read_rows(OUT/'simple_native.jsonl')}
    cases = {c.case_id:c for c in exp.load_cases(True)}
    for cid in ['rag-row-holdout2-001', 'vector-source-independent-003']:
        r=rows[cid]
        for i,evidence in enumerate(cases[cid].gold_evidence,1):
            eid=f'{cid}:positive:{i}'
            rank=r['scores']['context']['gold_completion_ranks'][eid]
            targets.append(dict(id=cid,entry=r['entries'][rank-1],quote=evidence['quote'],role='restored_gold'))
    output=[]
    render=OUT/'source_checks'
    render.mkdir(exist_ok=True)
    for n,t in enumerate(targets,1):
        e=t['entry'];ref=e['sources'][0];path=Path(ref['file_path'])
        if path.suffix.lower()=='.pdf':
            page=int(re.search(r'第\s*(\d+)\s*页',ref['location'])[1])
            with fitz.open(path) as pdf:
                p=pdf[page-1]
                text=p.get_text()
                png=render/f'{n:02}_{t["id"]}_p{page}.png'
                p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(png)
            mechanism='native PDF page text + rendered original page'
        elif path.suffix.lower()=='.pptx':
            page=int(re.search(r'第\s*(\d+)\s*张',ref['location'])[1])
            with ZipFile(path) as z:
                xml=ET.fromstring(z.read(f'ppt/slides/slide{page}.xml'))
                # Formatting runs (e.g. 1 + superscript 3S) are not separate lines.
                ns='{http://schemas.openxmlformats.org/drawingml/2006/main}'
                text='\n'.join(''.join(x.text or '' for x in p.iter(ns+'t')) for p in xml.iter(ns+'p'))
            png=None
            mechanism='native PPTX XML; no slide rendering in this audit'
        else:
            raise ValueError(path)
        exists=canonical_content(t['quote']) in canonical_content(text)
        output.append(dict(id=t['id'],role=t['role'],file=str(path),sha256=sha(path),page_or_slide_1based=page,
            mechanism=mechanism,quote=t['quote'],quote_in_original_page=exists,
            quote_in_actual_context=canonical_content(t['quote']) in canonical_content(e['included_text']),
            page_text=text,render=str(png) if png else None))
    write(OUT/'original_source_checks.json',dict(checks=output,read_only=True,
        limitation='PPTX XML proves quote location, not all table-cell semantic ownership; business interpretation retains existing Gold and source headings.'))
    print('ORIGINAL_CHECKS',len(output),'quotes_found',sum(x['quote_in_original_page'] for x in output),flush=True)
    check_frozen()


if __name__=='__main__':main()
