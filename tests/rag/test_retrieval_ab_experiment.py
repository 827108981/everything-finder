"""Experiment safeguards: no cross-document links or invented query models."""
from tools.validate_retrieval_ab import continuation_links, validate_queries


def page(cid, number, text, path='one.pptx'):
    return dict(chunk_id=cid, slide_number=number, page_number=None, text=text,
                file_path=path, ordinal=0, location=f'第{number}张')


def test_scenario_and_causes_can_link_without_rewriting_source():
    rows=[page('a',1,'多个测试失败\n故障特点：多项目报警'),page('b',2,'可能原因：\n1.污染\n解决思路：清洗')]
    before=[r.copy() for r in rows]
    assert continuation_links(rows)['a'][0][0]=='b'
    assert rows==before


def test_neighbour_is_not_enough_and_documents_do_not_mix():
    rows=[page('a',1,'故障：报警'),page('b',2,'可能原因：污染',path='other.pptx')]
    assert not continuation_links(rows)
    assert not continuation_links([page('a',1,'故障：报警'),page('b',2,'另一种机型故障\n可能原因：污染')])


def test_answer_page_does_not_expand_again():
    assert not continuation_links([page('a',1,'故障原因：污染\n解决思路：清洗'),page('b',2,'可能原因：供电')])


def test_query_model_tokens_must_be_retained_without_invention():
    question='CL-8000i诊断失败，怎么检查？能确定损坏吗？'
    assert validate_queries(question,{'queries':['CL-8000i诊断失败怎么检查？','CL-8000i诊断失败能确定损坏吗？']})
    assert not validate_queries(question,{'queries':['CL-6000i诊断失败怎么检查？','CL-8000i能确定损坏吗？']})
    assert not validate_queries(question,{'queries':['怎么检查？']})
