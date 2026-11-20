"""Native validation using the unchanged historical case cohort and scorer."""
from tools import validate_query_plan_adapter as native
from tools.rag_eval_common import ROOT

if __name__=='__main__':
    native.OUT=ROOT/'reports/rag_adapter_robustness_20261111'
    native.main()
