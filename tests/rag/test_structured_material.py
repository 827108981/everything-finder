import xml.etree.ElementTree as ET
from tools import validate_structured_material as v

def run_extract(tmp_path,monkeypatch,table,index_text):
    path=tmp_path/'source.pptx';path.write_bytes(b'fixture')
    xml=ET.fromstring('<p:sld xmlns:p="'+v.NS['p']+'" xmlns:a="'+v.NS['a']+'">'+table+'</p:sld>')
    monkeypatch.setattr(v,'slides',lambda p:iter([(1,xml)]))
    return v.extract([{'file_path':str(path),'slide_number':1,'text':index_text,'chunk_id':'parent'}])

def cell(t,attrs=''):
    return '<a:tc '+attrs+'><a:txBody><a:p><a:r><a:t>'+t+'</a:t></a:r></a:p></a:txBody></a:tc>'

def table(attrs=''):
    return '<a:tbl><a:tblPr firstRow="1"/><a:tr>'+cell('参数')+cell('值（μL/s）')+'</a:tr><a:tr>'+cell('流量',attrs)+cell('140±5')+'</a:tr></a:tbl>'

def test_units_and_source_provenance_survive(tmp_path,monkeypatch):
    units,audit,_=run_extract(tmp_path,monkeypatch,table(),'参数\n值（μL/s）\n流量\n140±5')
    assert len(units)==1 and units[0]['cells']==['流量','140±5']
    assert 'μL/s' in units[0]['text'] and units[0]['parent_ids']==['parent']
    assert audit[0]['status']=='accepted'

def test_merged_cells_fall_back_instead_of_copying_conditions(tmp_path,monkeypatch):
    units,audit,_=run_extract(tmp_path,monkeypatch,table('gridSpan="2"'),'参数\n值（μL/s）\n流量\n140±5')
    assert units==[] and audit[0]['reason']=='merged_cells_require_review'

def test_source_index_disagreement_blocks_unit(tmp_path,monkeypatch):
    units,audit,_=run_extract(tmp_path,monkeypatch,table(),'参数\n值（μL/s）\n流量\n145±5')
    assert units==[] and audit[0]['reason']=='source_cell_not_found_in_index'
