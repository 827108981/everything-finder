from __future__ import annotations

from local_full_text_search.ocr.adaptive_region_planner import (
    AdaptiveRegionPlanner,
)


def test_p0_03r_planner_starts_with_one_region_and_splits_lazily() -> None:
    planner = AdaptiveRegionPlanner(
        source_sha256="a" * 64,
        width=13_837,
        height=9_637,
        target_side=1_280,
        model_fingerprint="model-v1",
        strategy_version="adaptive-v2",
        anchors=[(12_000, 8_000, 13_000, 9_000)],
    )

    assert planner.created_count == 1
    root = planner.pop_next()
    assert root is not None
    children = planner.split(root.region_id)

    assert len(children) == 4
    assert planner.created_count == 5
    assert planner.pending_count == 4
    assert children[0].level == 1


def test_p0_03r_region_ids_and_processing_order_are_deterministic() -> None:
    def run() -> list[str]:
        planner = AdaptiveRegionPlanner(
            source_sha256="b" * 64,
            width=4_096,
            height=3_072,
            target_side=960,
            model_fingerprint="model-v1",
            strategy_version="adaptive-v2",
            anchors=[(100, 100, 600, 500), (3_000, 2_000, 3_900, 2_900)],
        )
        order: list[str] = []
        while planner.pending_count:
            region = planner.pop_next()
            assert region is not None
            order.append(region.region_id)
            if region.width > 960 or region.height > 960:
                planner.split(region.region_id)
            else:
                planner.resolve(region.region_id, "inspected")
        return order

    assert run() == run()


def test_p0_03r_checkpoint_restores_only_unresolved_regions() -> None:
    planner = AdaptiveRegionPlanner(
        source_sha256="c" * 64,
        width=2_000,
        height=2_000,
        target_side=960,
        model_fingerprint="model-v1",
        strategy_version="adaptive-v2",
        anchors=[],
    )
    root = planner.pop_next()
    assert root is not None
    children = planner.split(root.region_id)
    planner.resolve(children[0].region_id, "blank")
    checkpoint = planner.checkpoint()

    restored = AdaptiveRegionPlanner.from_checkpoint(checkpoint)

    assert restored.resolved_count == 1
    assert restored.pending_count == 3
    assert children[0].region_id not in {
        restored.pop_next().region_id,
    }


def test_p0_03r_checkpoint_preserves_batches_lines_order_and_cache_refs() -> None:
    planner = AdaptiveRegionPlanner(
        source_sha256="d" * 64,
        width=1_200,
        height=800,
        target_side=960,
        model_fingerprint="model-v1",
        strategy_version="adaptive-v2",
        anchors=[(10, 20, 300, 100)],
    )
    region = planner.pop_next()
    assert region is not None
    planner.record_detection_batch("detect-1")
    planner.record_recognition_batch("recognize-1")
    planner.confirm_line(
        line_id="line-1",
        text="确认文字",
        confidence=0.93,
        ordering_key=(1, 20, 10, "line-1"),
    )
    planner.add_cache_reference("cache-key-1")
    planner.resolve(region.region_id, "resolved")

    restored = AdaptiveRegionPlanner.from_checkpoint(planner.checkpoint())

    checkpoint = restored.checkpoint()
    assert checkpoint["completed_detection_batches"] == ["detect-1"]
    assert checkpoint["completed_recognition_batches"] == ["recognize-1"]
    assert checkpoint["confirmed_lines"] == [
        {
            "line_id": "line-1",
            "text": "确认文字",
            "confidence": 0.93,
            "ordering_key": [1, 20, 10, "line-1"],
        }
    ]
    assert checkpoint["cache_references"] == ["cache-key-1"]
