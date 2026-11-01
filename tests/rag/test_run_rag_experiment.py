from __future__ import annotations

import json

import pytest

from tools.run_rag_experiment import load_experiment_config


def test_experiment_config_requires_bundle_and_disallows_credentials(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({
        "generation_base_path": "E:/index/search_index.db",
        "expected_bundle_id": "g000008-example",
        "rag_lexical_multi_route_enabled": True,
    }), encoding="utf-8")
    assert load_experiment_config(path)["rag_lexical_multi_route_enabled"] is True

    path.write_text(json.dumps({
        "generation_base_path": "E:/index/search_index.db",
        "expected_bundle_id": "g000008-example",
        "api_key": "should-never-be-in-config",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_experiment_config(path)
