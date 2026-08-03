from __future__ import annotations

from tools.run_frozen_validations import VALIDATIONS
from tools.run_frozen_validations import PARAMETERIZED_VALIDATIONS


def test_s0_03r_frozen_validation_matrix_includes_hang_and_semantic_progress() -> None:
    assert "--validate-hang-recovery" in VALIDATIONS
    assert "--validate-semantic-progress" in VALIDATIONS


def test_u0_02v_frozen_validation_matrix_includes_single_eta() -> None:
    assert "--validate-single-eta" in VALIDATIONS


def test_schema_v8_frozen_validation_matrix_uses_latest_schema_gate() -> None:
    assert "--validate-schema-v8" in VALIDATIONS


def test_fts_02r_frozen_validation_matrix_includes_manual_exclusion() -> None:
    assert "--validate-manual-exclusion" in VALIDATIONS


def test_failure_fallback_demo_has_a_frozen_validation_command() -> None:
    assert "--validate-failure-demo" in VALIDATIONS


def test_phase2_frozen_validation_matrix_contains_every_special_gate() -> None:
    required = {
        "--validate-hang-recovery",
        "--validate-semantic-progress",
        "--validate-pdf-page-pipeline",
        "--validate-ocr-adaptive-v2",
        "--validate-ocr-backend",
        "--validate-index-status-layout",
        "--validate-single-eta",
        "--validate-safe-pause",
        "--validate-paused-mode-switch",
        "--validate-schema-v8",
        "--validate-manual-exclusion",
    }
    assert required.issubset(set(VALIDATIONS))


def test_p1_04r_declares_parameterized_cold_and_compare_commands() -> None:
    assert (
        "--benchmark-cold-index <folder> --output <json> [--performance]"
        in PARAMETERIZED_VALIDATIONS
    )
    assert (
        "--compare-index <baseline-db> <candidate-db> --queries <json>"
        in PARAMETERIZED_VALIDATIONS
    )
