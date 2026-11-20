from dataclasses import replace
import hashlib
import sqlite3
from pathlib import Path
import pytest
from pptx import Presentation
from pptx.util import Inches
from local_full_text_search.rag.retrieval.embedding_context import EmbeddingContextPolicy, VerifiedBackground, fuse_parent_view_ranks
from local_full_text_search.rag.retrieval.vector_units import VectorUnitBuilder, VectorUnitPolicy
from local_full_text_search.rag.retrieval.source_representation import inspect_pptx, SourceBackgroundProvider, StructuredUnitBuilder
from local_full_text_search.rag.vector_index_builder import VectorIndexBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from tests.rag.test_vector_unit_v2 import _chunk, _count_tokens, _Embedding


def test_optional_views_keep_original_windows_spans_and_legacy_ids():
    chunk=_chunk(1,'型号X-900M\n参数\n24V±5%\n'+('操作说明。'*35),'维修手册V2.3.pdf')
    policy=VectorUnitPolicy(36,64,8)
    baseline=VectorUnitBuilder(_count_tokens,policy).build(chunk)
    legacy_fingerprint=VectorUnitBuilder(_count_tokens,policy).fingerprint
    assert legacy_fingerprint==policy.fingerprint
    for i in range(6):
        candidate=VectorUnitBuilder(_count_tokens,policy,context_policy=EmbeddingContextPolicy(f'W{i}')).build(chunk)
        assert [(u.text,u.start_char,u.end_char,u.ordinal) for u in candidate]==[(u.text,u.start_char,u.end_char,u.ordinal) for u in baseline]
        assert all(u.token_count<=64 for u in candidate)
        if i==0:
            assert [(u.unit_id,u.embedding_text) for u in candidate]==[(u.unit_id,u.embedding_text) for u in baseline]
        else:
            assert set(u.unit_id for u in candidate).isdisjoint(u.unit_id for u in baseline)


def test_background_budget_never_truncates_body_model_or_version():
    chunk=_chunk(1,'正常供电范围22.8～25.2V','BS-2800M维修手册软件V1.9.2.pdf')
    policy=EmbeddingContextPolicy('W5',32)
    text,meta=policy.render(chunk,chunk.text,_count_tokens,192)
    assert 'BS-2800M维修手册软件V1.9.2' in text
    assert text.endswith(chunk.text) and meta['scope_authority']=='filename_hint'
    text,meta=EmbeddingContextPolicy('W5',4).render(chunk,chunk.text,_count_tokens,192)
    assert text==chunk.text and 'exceeds_budget' in meta['context_omitted_reason']


def test_verified_title_is_deduplicated_without_broader_filename_injection():
    chunk=_chunk(1,'本地标题\n参数24V','另一个型号维修手册V1.0.pdf')
    text,meta=EmbeddingContextPolicy('W5').render(chunk,chunk.text,_count_tokens,192,VerifiedBackground('本地标题','shape:7'))
    assert text==chunk.text and 'already_in_body' in meta['context_omitted_reason']


def test_identical_views_and_repeated_windows_do_not_gain_duplicate_votes():
    baseline=fuse_parent_view_ranks([('view-hash',['a','b','c'])])
    repeated=fuse_parent_view_ranks([('view-hash',['a','a','b','b','c']),('view-hash',['a','b','c'])])
    assert baseline==repeated
    other=fuse_parent_view_ranks([('body',['a','b']),('context',['b','c'])])
    assert other[0][0]=='b' and len(other[2]['b'])==2


def _deck(tmp_path):
    path=tmp_path/'unknown-model-V3.1.pptx'
    p=Presentation();slide=p.slides.add_slide(p.slide_layouts[5]);slide.shapes.title.text='通信参数'
    table=slide.shapes.add_table(3,3,Inches(1),Inches(2),Inches(8),Inches(3)).table
    for row,values in zip(table.rows,[['设备','接口','单位'],['X-9','以太网','Mb/s'],['Y-8','串口','bit/s']]):
        for cell,value in zip(row.cells,values):cell.text=value
    p.save(path)
    return p,path


def test_native_rows_preserve_header_units_and_source_coordinates(tmp_path):
    _,path=_deck(tmp_path)
    doc=inspect_pptx(path);docs={str(path):doc}
    chunk=_chunk(1,'设备\n接口\n单位\nX-9\n以太网\nMb/s\nY-8\n串口\nbit/s')
    chunk=replace(chunk,source=replace(chunk.source,file_path=str(path),filename=path.name,extension='.pptx',page_number=None,slide_number=1))
    base=VectorUnitBuilder(_count_tokens,VectorUnitPolicy(64,128,8),context_policy=EmbeddingContextPolicy('W5'),background_provider=SourceBackgroundProvider(docs))
    units=StructuredUnitBuilder(base,docs).build(chunk)
    rows=[u for u in units if u.kind=='native_table_row']
    assert len(rows)==2
    assert rows[0].text=='设备：X-9\n接口：以太网\n单位：Mb/s'
    assert rows[0].start_char==rows[0].end_char==0
    assert rows[0].representation_metadata['source_span_kind']=='noncontiguous_original_cells'
    assert rows[0].representation_metadata['cells'][2]=={'row':1,'column':2,'header':'单位','value':'Mb/s'}
    assert len(StructuredUnitBuilder(base,docs,replacement=True).build(chunk))==len(base.build(chunk))
    other=replace(chunk,text='另一个对象\n只有一个值X-9')
    assert all(u.kind!='native_table_row' for u in StructuredUnitBuilder(base,docs).build(other))


@pytest.mark.parametrize('invalid',['merged','blank','duplicate','unstyled'])
def test_ambiguous_native_tables_are_audited_and_not_guessed(tmp_path,invalid):
    p,path=_deck(tmp_path);table=p.slides[0].shapes[-1].table
    if invalid=='merged':table.cell(0,0).merge(table.cell(0,1))
    if invalid=='blank':table.cell(0,1).text=''
    if invalid=='duplicate':table.cell(0,1).text='设备'
    if invalid=='unstyled':table.first_row=False
    p.save(path)
    table=inspect_pptx(path)['slides']['1']['tables'][0]
    assert not table['eligible'] and table['skip_reason']


def test_isolated_index_never_writes_production_cache_or_pointer(tmp_path):
    store=RagIndexStore(tmp_path/'rag');bundle,connection=store.create_candidate(1,
        embedding_repo_id=_Embedding.model_id,embedding_revision=_Embedding.model_revision,embedding_dimension=3,
        reranker_repo_id='test/r',reranker_revision='r1')
    chunk=_chunk(1,'未知设备的通信参数。')
    store.add_chunk(connection,chunk);connection.commit();connection.close();bundle=replace(bundle,chunk_count=1)
    root=tmp_path/'experiment'
    settings=AppSettings(rag_vector_unit_target_tokens=16,rag_vector_unit_max_tokens=48,rag_vector_unit_overlap_tokens=4)
    unitbuilder=VectorUnitBuilder(_count_tokens,VectorUnitPolicy.from_settings(settings),context_policy=EmbeddingContextPolicy('W4'))
    index,_=VectorIndexBuilder(store,_Embedding(),settings,vector_root=root,unit_builder=unitbuilder).prepare(bundle,CancelToken(),publish_pointer=False)
    assert (root/'vector/embedding_cache.db').is_file()
    assert not (store.root/'embedding_cache.db').exists()
    assert not (root/'vector'/bundle.bundle_id/'active.json').exists()
    with sqlite3.connect(index.database_path) as c:
        ids=[r[0] for r in c.execute('SELECT unit_id FROM vector_units')]
    loaded=VectorIndexStore.load_units(index.database_path,ids)
    assert all(r['embedding_text']==r['text'] and r['representation_metadata']['view']=='W4' for r in loaded.values())
