"""Native validation against the frozen, user-approved 91-case v5 suite."""
import json
from tools import validate_query_plan_adapter as native
from tools.rag_eval_common import ROOT

native.OUT=ROOT/'reports/rag_business_repair_20261112'
manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
native.FILES=[ROOT/d['path'] for d in manifest['datasets']]
if __name__=='__main__': native.main()
