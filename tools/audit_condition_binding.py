"""Source verification and frozen, pending independent applicability probes."""
from pathlib import Path
from dataclasses import asdict
import json
from tools.validate_condition_binding import OUT, ARMS
from tools.validate_selection_factorial import sha, write, ROOT, no_network
from tools.rag_trace_audit import original_text
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.models import EvidenceChunk
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher

# Questions and source IDs are evaluation fixtures, never imported by runtime.
PROBES=[
 ('SDM单机配置时，如何检查所有通道的调度和条码识别？','56fbe173a8d54bb403ad69d84f03cae063820797d5e99abdd48d7fa54eff157d','在1~7号所有通道放满试管架', 'instruction_not_failure'),
 ('SDM普通通道设为快速加急后，这个设置一直有效吗？','56fbe173a8d54bb403ad69d84f03cae063820797d5e99abdd48d7fa54eff157d','快速加急通道，设置后，单次有效','time_condition'),
 ('BS-5000系列的试剂项目都配有质控品吗？哪些没有校准品？','1ddfbddba288b83fbc3ecd4b660ef984312b1c844fd6835b3072f52603d25998','质控品：所有项目均有质控品','enumeration_not_fault'),
 ('生化全开放系统和封闭系统，使用第三方试剂的项目范围有什么区别？','f949e5b34f332e14f6fb48f615eb49fc365f251e1a1413395476549111626a05','全开放系统：所有项目通道都可以使用第三方生化试剂的系统','multiple_branches'),
 ('Medica ISE定标失败，所有项目斜率都为负数，先检查哪里？','d4c178b28b31092b081aede0988887f0ebb57b38ea3378d1d0e3c9438644e43a','检查A液与B液的连接管路—发现接反','negative_numeric_direction'),
 ('压力传感器只能检测正压，不能检测负压吗？','0da17fe1248bdd89272b8f62e86c7426a9d650c12fc5eec4cdd574b623510279','压力传感器能检测正压和负压','negation_and_two_conditions'),
 ('血清和全血短期保存的温度与时间分别是多少？','59d2d54d5321c13d0828a13a8cdbdff185fa0f87aa2efeac7fe70733eac367ea','短期保存（1-7天）','object_and_units'),
 ('什么时候全面使用TEST规格试剂包装？集采和非集采区域有什么区别？','8dec87a7507860e2c283c186252ae38def4cb628f316128685fc3d4fdfea3c1d','当肾功集采TEST规格增补完成后','time_and_scope_branches'),
]


def main():
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    audit=[]; fixtures=[]; probes=[]
    with store.connect_read_only(bundle.metadata_path) as conn:
        rows={a:{r['id']:r for r in map(json.loads,(OUT/(a+'.jsonl')).read_text(encoding='utf8').splitlines())} for a in ARMS}
        for cid,r in rows['combined'].items():
            changed=set(r['removed_chunks']+r['added_chunks'])
            for chunk_id in sorted(changed):
                ch=store.load_chunk(conn,chunk_id);src=ch.source;page=src.slide_number or src.page_number
                original=original_text(Path(src.file_path),page)
                current=next(e for e in r['evidence'] if e['chunk_id']==chunk_id)
                quotes=[j['quote'] for j in current['query_plan_diagnostics']['local_conditions']['checks'] if j['quote']]
                audit.append({'id':cid,'chunk_id':chunk_id,'change':'removed' if chunk_id in r['removed_chunks'] else 'added',
                    'file':src.file_path,'page':page,'location':src.location_text,'source_sha256':sha(src.file_path),
                    'text':ch.text,'original_text':original,'checks':current['query_plan_diagnostics']['local_conditions']['checks'],
                    'quotes_verified':all(canonical_content(q) in canonical_content(original) and q in ch.text for q in quotes),
                    'reason':current['reason'],'selected':current['selected']})
        for i,(q,cid,quote,category) in enumerate(PROBES,1):
            ch=store.load_chunk(conn,cid);src=ch.source;page=src.slide_number or src.page_number
            original=original_text(Path(src.file_path),page)
            assert canonical_content(quote) in canonical_content(original) and quote in ch.text
            fixtures.append({'id':f'condition-independent-{i:02}','question':q,'chunk_id':cid,'file':src.file_path,
                'page':page,'quote':quote,'review_status':'pending','category':category,'expected_candidate_accepted':True,
                'original_verified':True,'index_verified':True,'source_sha256':sha(src.file_path),
                'independence':'New questions and source pages, not a fully independent document split; acceptance checks only, no answer Gold.'})
        path=OUT/'independent_pending.json'
        if path.exists(): assert json.loads(path.read_text(encoding='utf8'))==fixtures
        else: write(path,fixtures)
        # Fixture is written and hashed before executing probes; never retuned.
        frozen=sha(path)
        with no_network() as attempts:
            for f in fixtures:
                for arm,(extract,local) in ARMS.items():
                    plan=QueryInterpreter(bound_conditions=extract).interpret(f['question'])
                    item=EvidenceChunk(chunk=store.load_chunk(conn,f['chunk_id']),lexical_rank=1,rerank_score=.8,vector_score=.8)
                    checked=ConstraintMatcher(local_conditions=local).evaluate(plan,[item])[0]
                    probes.append({'id':f['id'],'arm':arm,'accepted':checked.constraint_decision.accepted,
                        'passed':checked.constraint_decision.accepted==f['expected_candidate_accepted'],
                        'plan':asdict(plan),'decision':asdict(checked.constraint_decision)})
        assert sha(path)==frozen and not attempts
    write(OUT/'source_audit.json',audit)
    write(OUT/'independent_results.json',{'fixture_sha256':frozen,'records':probes,'cases':len(fixtures),
          'scope':'Known-candidate applicability only. Not retrieval recall or answer validation; no claims of conflict-recall generalization.',
          'passed_by_arm':{a:sum(r['passed'] for r in probes if r['arm']==a) for a in ARMS}})
    print('Changed sources',len(audit),'quote verification',all(x['quotes_verified'] for x in audit))
    print('Independent', {a:sum(r['passed'] for r in probes if r['arm']==a) for a in ARMS})


if __name__=='__main__':main()
