import json
import threading
import urllib.error
import urllib.request
import zipfile
from urllib.parse import urlsplit, parse_qs

from local_full_text_search.rag.evaluation.reporting.comparison import write_comparison_report
from tools.serve_rag_report import make_server, physical_source


def build(tmp_path, sources, gold):
    case = {'id': 'example', 'question': '<script>alert(1)</script>', 'review_status': 'pending', 'gold_evidence': gold}
    capture = {'context': 'actual text', 'entries': [], 'scores': {}}
    stats = write_comparison_report(tmp_path, [case], {'off': {'example': capture}, 'on': {'example': capture}}, {}, {}, sources)
    return stats


def test_ambiguous_basename_never_opens_arbitrary_file(tmp_path):
    paths = []
    for folder in ('a', 'b'):
        path = tmp_path / folder / 'same.pptx'
        path.parent.mkdir()
        path.touch()
        paths.append(path)
    stats = build(tmp_path, paths, [{'file': 'same.pptx', 'page': 14}])
    assert stats['unresolved'] == 1
    assert not json.loads((tmp_path / 'report_sources.json').read_text())
    text = (tmp_path / 'report.html').read_text(encoding='utf8')
    assert '<script>alert(1)</script>' not in text
    assert '仅诊断，不计分' in text


def test_slide_pdf_locations_and_http_guard(tmp_path):
    ppt = tmp_path / 'demo.pptx'
    pdf = tmp_path / 'demo.pdf'
    ppt.touch()
    pdf.touch()
    stats = build(tmp_path, [ppt, pdf], [{'file': ppt.name, 'page': 14}, {'file': pdf.name, 'page': 19}])
    assert stats == {'cases': 1, 'resolved_locations': 2, 'unresolved': 0}
    sources = json.loads((tmp_path / 'report_sources.json').read_text(encoding='utf8'))
    assert [s['location']['slide_number'] for s in sources.values()] == [14, None]
    assert [s['location']['page_number'] for s in sources.values()] == [None, 19]
    calls = []
    server, url = make_server(tmp_path / 'report.html', lambda p, loc: calls.append((p, loc)) or 'opened')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = 'http://' + urlsplit(url).netloc
    token = parse_qs(urlsplit(url).query)['token'][0]

    def post(payload, auth=token, request_origin=origin):
        req = urllib.request.Request(origin + '/open-source', json.dumps(payload).encode(), {'X-Report-Token': auth, 'Origin': request_origin}, method='POST')
        try:
            with urllib.request.urlopen(req) as res:
                return res.status
        except urllib.error.HTTPError as exc:
            return exc.code

    try:
        assert post({'id': next(iter(sources))}, auth='wrong') == 403
        assert post({'id': next(iter(sources))}, request_origin='https://other.example') == 403
        assert post({'id': '../secret', 'path': 'C:/secret'}) == 404
        assert calls == []
        assert post({'id': next(iter(sources))}) == 200
        assert calls[0][0] == ppt
        assert calls[0][1]['slide_number'] == 14
        with urllib.request.urlopen(url) as res:
            assert res.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_archive_location_is_retained(tmp_path):
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('folder/test.pdf', b'%PDF test')
    raw = str(archive) + ' > folder/test.pdf'
    stats = build(tmp_path, [raw], [{'file_path': raw, 'location': '第 11 页'}])
    assert stats['unresolved'] == 0
    source = next(iter(json.loads((tmp_path / 'report_sources.json').read_text(encoding='utf8')).values()))
    assert source['location']['page_number'] == 11
    path = physical_source(source)
    assert path.name == 'test.pdf'
    assert path.read_bytes() == b'%PDF test'


def test_archive_member_cannot_choose_extraction_directory(tmp_path):
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../../outside.pdf', b'test')
    path = physical_source({'archive': str(archive), 'member': '../../outside.pdf'})
    assert path.parent.parent.name == 'rag_report_sources'
    assert path.name == 'outside.pdf'
