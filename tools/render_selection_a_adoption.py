"""Publish verified production A with the unchanged historical comparison data."""
import json
import shutil
import xml.etree.ElementTree as ET
from tools.validate_selection_a_adoption import OUT, EXPERIMENT, ROOT


def main():
    replay = json.loads((OUT/'production_replay.json').read_text(encoding='utf8'))
    assert replay['passed'] and replay['cases'] == 91
    tests = [t for name in ('unit_results.xml', 'settings_unit_results.xml')
             for t in ET.parse(OUT/name).iter('testcase')]
    failures = [(t.get('classname'), t.get('name')) for t in tests if t.find('failure') is not None]
    previous = [(t.get('classname'), t.get('name'))
                for t in ET.parse(EXPERIMENT/'unit_results.xml').iter('testcase') if t.find('failure') is not None]
    assert sorted(failures) == sorted(previous)
    assert len(tests) == 474
    audit = {'adopted': 'A', 'runtime_default_document_cap': None, 'final_top_k': 8,
             'context_budget': 12000, 'tests_passed': 463, 'existing_failures': failures,
             'new_failures': 0, 'skipped': 1, 'production_matches_experiment': replay['passed'],
             'api_calls': 0, 'adoption_basis': 'User authorized selection-layer gain; upstream applicability risks tracked separately'}
    (OUT/'final_acceptance.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    page = (EXPERIMENT/'report.html').read_text(encoding='utf8')
    replacements = {
        '最终选段八组对照：暂不采用，不改变应用默认行为': '已采用A：取消单文档配额；其余方案仅作历史对照',
        '所有组合均未满足采用门槛。': '本轮按用户决定采用A，保留其已验证的选段收益；上游已有范围识别缺口单独修复，B/C及组合未采用。',
        '单元/兼容回归：459通过，10个既有失败，1跳过；新增失败0。Gold及运行默认配置未改。没有合格候选进入独立验收，不能宣称泛化已通过。':
            '接入核验：生产策略与实验A的Context、条目、评分91/91一致。单元/兼容回归463通过，10个既有失败，1跳过；新增失败0。Gold及持久化设置未改，应用选段已采用A。此为冻结候选重放，未重跑上游或生成答案，不宣称独立泛化验收通过。',
        '<option value="AC" selected>': '<option value="AC">',
        '<option value="A">': '<option value="A" selected>',
        "let activeArm='AC';": "let activeArm='A';",
        '正式完整证据': '正式题全部Gold引文覆盖',
        '诊断完整证据': '诊断题全部Gold引文覆盖',
    }
    for old, new in replacements.items():
        assert old in page, old
        page = page.replace(old, new)
    (OUT/'report.html').write_text(page, encoding='utf8')
    shutil.copyfile(EXPERIMENT/'report_sources.json', OUT/'report_sources.json')
    shutil.copyfile(ROOT/'docs/rag_selection_a_adoption_20261113.md', OUT/'conclusion.md')
    cmd = '@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n'
    (OUT/'open_report.cmd').write_text(cmd, encoding='utf8')
    print('Published production A adoption report; historical comparison bodies retained unchanged.')


if __name__ == '__main__':
    main()
