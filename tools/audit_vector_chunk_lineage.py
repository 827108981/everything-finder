"""Read-only source -> published blocks -> parents -> vector-unit lineage audit.

Header co-occurrence is a lexical audit, not an automatic judgment of table meaning.
No source, parser, index, model or production slicing setting is modified.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from pptx import Presentation
from tools.validate_selection_factorial import ROOT, write, sha, no_network
from tools.diagnose_vector_recall import ro
from local_full_text_search.rag.content_identity import canonical_content as canon
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.retrieval.vector_units import VectorUnitBuilder, VectorUnitPolicy
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.core.text_blob_codec import decode_text_blob
from local_full_text_search.parsers.ooxml.pptx_stream_parser import _paragraph_texts
from local_full_text_search.parsers.ooxml.common import xml_parts
from local_full_text_search.parsers.pptx_parser import _extract_shape_texts

OUT=ROOT/'reports/rag_gold_material_repair_20261116'
PRIOR=ROOT/'reports/rag_vector_diagnosis_20261116'
SNAP=Path('E:/everythingfinder-db/search_index.gen-000033.db')


def shapes_recursive(shapes):
    for shape in shapes:
        yield shape
        if hasattr(shape,'shapes'):yield from shapes_recursive(shape.shapes)


def main():
    from transformers import AutoTokenizer
    from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
    from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
    protocol=json.loads((PRIOR/'protocol.json').read_text(encoding='utf8'))
    protected={**protocol['protected'],str(SNAP):sha(SNAP)}
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    assert bundle.bundle_id==protocol['bundle']
    with store.connect_read_only(bundle.metadata_path) as conn:
        parents={r[0]:store.load_chunk(conn,r[0]) for r in conn.execute('select stable_chunk_id from rag_chunks')}
    with ro(protocol['vector_index']['database_path']) as conn:
        units=[dict(r) for r in conn.execute('select * from vector_units order by parent_chunk_id,ordinal')]
    by_parent=defaultdict(list);by_page=defaultdict(list)
    for u in units:by_parent[u['parent_chunk_id']].append(u)
    for p in parents.values():by_page[(str(Path(p.source.file_path)),p.source.slide_number)].append(p)
    snap=ro(SNAP);snap.create_function('decode_text',2,decode_text_blob)
    files={str(Path(r['path'])):dict(r) for r in snap.execute('select * from files')}
    selected=[json.loads(x) for x in (PRIOR/'diagnosis.jsonl').read_text(encoding='utf8').splitlines()]
    selected_ids={p['chunk_id'] for r in selected for t in r['targets'] for p in t['parents']}
    rebuild=[];lineage=[]
    with no_network() as attempts:
        path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(protocol['model'],protocol['revision'],'embedding'))
        tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        count=lambda t:len(tokenizer(t,truncation=False)['input_ids'])
        policy=VectorUnitPolicy();assert policy.fingerprint==protocol['vector_index']['policy_fingerprint']
        builder=VectorUnitBuilder(count,policy)
        for cid in sorted(selected_ids):
            p=parents[cid]
            built=[dict(unit_id=u.unit_id,parent_chunk_id=u.parent_chunk_id,ordinal=u.ordinal,text=u.text,embedding_text=u.embedding_text,
                        kind=u.kind,start_char=u.start_char,end_char=u.end_char,token_count=u.token_count) for u in builder.build(p)]
            saved=by_parent[cid]
            rebuild.append(dict(parent=cid,equal=built==saved,count=len(saved)))
            raw=[dict(r) for r in snap.execute('select * from search_blocks where id in ('+','.join('?' for _ in p.source.block_ids)+')',p.source.block_ids)]
            file=files.get(str(Path(p.source.file_path)),{})
            lineage.append(dict(parent=cid,file=p.source.file_path,page=p.source.page_number,slide=p.source.slide_number,
                parser_name=file.get('parser_name'),parser_version=file.get('parser_version'),source_generation=p.source.source_generation,
                block_ids=list(p.source.block_ids),published_blocks=raw,parent_text=p.text,units=saved,
                parent_text_in_joined_blocks=canon(p.text) in canon('\n'.join(r['raw_text'] for r in raw))))
        assert all(r['equal'] for r in rebuild)
        assert not attempts
    print('Reproduced selected persisted units:',len(rebuild),flush=True)
    # Scan every accessible indexed native PPTX, not only the diagnostic questions.
    pptpaths=sorted({str(Path(p.source.file_path)) for p in parents.values() if p.source.extension=='.pptx'})
    stats=Counter();tables=[];errors=[];orders=[];parser_replay=[]
    for fi,file in enumerate(pptpaths,1):
        if not Path(file).is_file():errors.append(dict(file=file,error='missing source'));continue
        try:
            presentation=Presentation(file)
            with zipfile.ZipFile(file) as archive:
                parts=xml_parts(archive,r'ppt/slides/slide\d+\.xml')
                actual=[str(s.part.partname).lstrip('/') for s in presentation.slides]
                diffs=[dict(page=i+1,presentation_part=a,stream_part=b) for i,(a,b) in enumerate(zip(actual,parts)) if a!=b]
                if diffs:orders.append(dict(file=file,differences=diffs))
                for page,slide in enumerate(presentation.slides,1):
                    ps=by_page.get((file,page),[])
                    if not ps:continue
                    stats['indexed_slides']+=1
                    us=[u for p in ps for u in by_parent[p.stable_chunk_id]]
                    uc=[canon(u['text']) for u in us]
                    parentbody=canon('\n'.join(p.text for p in ps))
                    table_shapes=[(i,s) for i,s in enumerate(shapes_recursive(slide.shapes)) if s.has_table]
                    if not table_shapes:continue
                    stats['indexed_slides_with_native_tables']+=1
                    # Source and published parser lineage checks on all selected table pages.
                    if any(p.stable_chunk_id in selected_ids for p in ps):
                        stream='\n'.join(_paragraph_texts(archive,parts[page-1],CancelToken()))
                        blocks=[dict(r) for r in snap.execute('select * from search_blocks where file_id=? and slide_number=?',(ps[0].source.file_id,page))]
                        parser_replay.append(dict(file=file,page=page,parser=files.get(file,{}).get('parser_name'),
                            presentation_part=actual[page-1],stream_part=parts[page-1],stream_text=stream,
                            regular_parser_text='\n'.join(_extract_shape_texts(slide.shapes)),published_blocks=blocks,
                            stream_equal_to_published=any(canon(stream)==canon(b['raw_text']) for b in blocks)))
                    for si,shape in table_shapes:
                        stats['native_tables']+=1
                        cells=[[c.text.strip() for c in row.cells] for row in shape.table.rows]
                        merged=any(c.is_merge_origin or c.is_spanned for row in shape.table.rows for c in row.cells)
                        headers=[h for h in cells[0] if h] if cells else []
                        # Deliberately call this a first-row lexical proxy, not a verified semantic header.
                        eligible=not merged and len(cells)>1 and len(headers)>=2 and len(set(headers))==len(headers)
                        rec=dict(file=file,page=page,shape=si,rows=len(cells),columns=len(cells[0]) if cells else 0,
                            merged=merged,first_row_lexical_proxy_eligible=eligible,first_row=headers,row_checks=[])
                        if eligible:
                            stats['unmerged_first_row_proxy_tables']+=1
                            for ri,row in enumerate(cells[1:],1):
                                if not any(row):continue
                                stats['proxy_data_rows']+=1
                                rowtext=canon('\n'.join(x for x in row if x))
                                row_units=[u for u,t in zip(us,uc) if rowtext and rowtext in t]
                                header_units=[u for u in row_units if all(canon(h) in canon(u['text']) for h in headers)]
                                inparent=rowtext in parentbody
                                if inparent:stats['proxy_rows_in_parent']+=1
                                if row_units:stats['proxy_rows_complete_in_some_unit']+=1
                                if row_units and not header_units:stats['proxy_complete_rows_without_all_first_row_labels']+=1
                                if header_units:stats['proxy_rows_with_all_first_row_labels']+=1
                                if inparent and not row_units:stats['proxy_rows_split_across_units']+=1
                                rec['row_checks'].append(dict(row=ri,cells=row,row_in_parent=inparent,
                                    row_unit_ids=[u['unit_id'] for u in row_units],row_and_first_row_unit_ids=[u['unit_id'] for u in header_units]))
                        else:stats['semantic_header_not_auto_assessed_tables']+=1
                        tables.append(rec)
            stats['accessible_files']+=1
        except Exception as exc:errors.append(dict(file=file,error=repr(exc)))
        if fi%10==0:print('PPTX scanned',fi,'/',len(pptpaths),'tables',stats['native_tables'],flush=True)
    snap.close()
    drift=[p for p,h in protected.items() if sha(ROOT/p)!=h]
    assert not drift
    audit=dict(bundle=bundle.bundle_id,snapshot=str(SNAP),source_generation=33,api_calls=0,
        parent_count=len(parents),unit_count=len(units),indexed_pptx_files=len(pptpaths),stats=dict(stats),errors=errors,
        counts_are='All indexed native PPTX original table objects; row/header results only unmerged unique first-row lexical proxy, not semantic/table correctness rate.',
        unit_kinds=dict(Counter(u['kind'] for u in units)),source_prefix_omitted=sum(u['embedding_text']==u['text'] for u in units),
        rebuilt_selected_parents=rebuild,selected_lineage=lineage,selected_parser_replays=parser_replay,
        pptx_slide_order_mismatches=orders,table_checks=tables,protected_drift=drift,
        limitations=['PDF visually drawn tables / images and legacy PPT tables are not counted as native PPTX table objects.',
            'Co-occurrence of first-row strings is necessary for many tables but does not prove column binding or semantic header correctness.',
            'Local source representation controls do not establish full-index rebuild gains.'])
    write(OUT/'chunk_lineage_audit.json',audit)
    print(json.dumps(dict(stats=stats,errors=errors,slide_order_mismatch_files=len(orders),rebuilt=len(rebuild),drift=drift),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
