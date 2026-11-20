"""Frozen-suite validation wrapper. Gold is used only by the evaluator."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--structured', action='store_true')
    parser.add_argument('--support-graph', action='store_true')
    args, remaining = parser.parse_known_args()
    from tools import validate_query_plan_adapter as native
    native.OUT = args.output_dir.resolve()
    manifest = json.loads((args.suite_root / 'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    native.FILES = [args.suite_root / row['path'] for row in manifest['datasets']]
    if args.structured or args.support_graph:
        original = native.Experiment
        class StructuredExperiment(original):
            def __init__(self, *positional, **keywords):
                super().__init__(*positional, **keywords)
                self.settings.rag_structured_answer_enabled = True
                if args.support_graph:
                    self.settings.rag_answer_review_enabled = True
                    self.settings.rag_answer_review_strict = False
                    self.settings.rag_semantic_clarification_enabled = True
        native.Experiment = StructuredExperiment
    sys.argv = [sys.argv[0], *remaining]
    native.main()


if __name__ == '__main__':
    main()
