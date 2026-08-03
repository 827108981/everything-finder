from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AdaptiveRegion:
    region_id: str
    level: int
    left: int
    top: int
    right: int
    bottom: int
    heat: float
    state: str = "pending"

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height


class AdaptiveRegionPlanner:
    """Deterministic unresolved-region quadtree for recall-first OCR."""

    def __init__(
        self,
        *,
        source_sha256: str,
        width: int,
        height: int,
        target_side: int,
        model_fingerprint: str,
        strategy_version: str,
        anchors: list[tuple[int, int, int, int]],
    ) -> None:
        self.source_sha256 = str(source_sha256)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.target_side = max(64, int(target_side))
        self.model_fingerprint = str(model_fingerprint)
        self.strategy_version = str(strategy_version)
        self.anchors = [
            tuple(int(value) for value in anchor)
            for anchor in anchors
        ]
        self._regions: dict[str, AdaptiveRegion] = {}
        self._pending: set[str] = set()
        self._active: set[str] = set()
        self._resolved: dict[str, str] = {}
        self._split: set[str] = set()
        self._completed_detection_batches: set[str] = set()
        self._completed_recognition_batches: set[str] = set()
        self._confirmed_lines: dict[str, dict[str, Any]] = {}
        self._cache_references: set[str] = set()
        self._checkpoint_version = 0
        root = self._make_region(0, 0, 0, self.width, self.height)
        self._regions[root.region_id] = root
        self._pending.add(root.region_id)

    @property
    def created_count(self) -> int:
        return len(self._regions)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def resolved_count(self) -> int:
        return len(self._resolved)

    @property
    def split_count(self) -> int:
        return len(self._split)

    @property
    def checkpoint_version(self) -> int:
        return self._checkpoint_version

    @property
    def completed_recognition_batches(self) -> frozenset[str]:
        return frozenset(self._completed_recognition_batches)

    @property
    def confirmed_lines(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(self._confirmed_lines[line_id])
            for line_id in sorted(self._confirmed_lines)
        )

    @property
    def coverage_ratio(self) -> float:
        resolved_area = sum(
            self._regions[region_id].area
            for region_id in self._resolved
        )
        return min(1.0, resolved_area / max(1, self.width * self.height))

    def pop_next(self) -> AdaptiveRegion | None:
        if not self._pending:
            return None
        region_id = min(
            self._pending,
            key=lambda value: self._sort_key(self._regions[value]),
        )
        self._pending.remove(region_id)
        self._active.add(region_id)
        return self._regions[region_id]

    def split(self, region_id: str) -> list[AdaptiveRegion]:
        region = self._require_unresolved(region_id)
        if region.width <= 1 and region.height <= 1:
            raise ValueError("Region cannot be split further")
        mid_x = region.left + max(1, region.width // 2)
        mid_y = region.top + max(1, region.height // 2)
        rectangles = (
            (region.left, region.top, mid_x, mid_y),
            (mid_x, region.top, region.right, mid_y),
            (region.left, mid_y, mid_x, region.bottom),
            (mid_x, mid_y, region.right, region.bottom),
        )
        children: list[AdaptiveRegion] = []
        for left, top, right, bottom in rectangles:
            if right <= left or bottom <= top:
                continue
            child = self._make_region(
                region.level + 1,
                left,
                top,
                right,
                bottom,
            )
            self._regions[child.region_id] = child
            self._pending.add(child.region_id)
            children.append(child)
        self._pending.discard(region_id)
        self._active.discard(region_id)
        self._split.add(region_id)
        self._checkpoint_version += 1
        return sorted(children, key=self._sort_key)

    def resolve(self, region_id: str, state: str) -> None:
        if state not in {"inspected", "blank", "resolved", "cached"}:
            raise ValueError(f"Invalid resolved region state: {state}")
        self._require_unresolved(region_id)
        self._pending.discard(region_id)
        self._active.discard(region_id)
        self._resolved[region_id] = state
        self._checkpoint_version += 1

    def record_detection_batch(self, batch_id: str) -> None:
        self._completed_detection_batches.add(str(batch_id))
        self._checkpoint_version += 1

    def record_recognition_batch(self, batch_id: str) -> None:
        self._completed_recognition_batches.add(str(batch_id))
        self._checkpoint_version += 1

    def confirm_line(
        self,
        *,
        line_id: str,
        text: str,
        confidence: float | None,
        ordering_key: tuple[int, int, int, str],
        batch_id: str | None = None,
        box: list[list[float]] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "line_id": str(line_id),
            "text": str(text),
            "confidence": (
                float(confidence) if confidence is not None else None
            ),
            "ordering_key": [
                int(ordering_key[0]),
                int(ordering_key[1]),
                int(ordering_key[2]),
                str(ordering_key[3]),
            ],
        }
        if batch_id is not None:
            payload["batch_id"] = str(batch_id)
        if box is not None:
            payload["box"] = [
                [float(point[0]), float(point[1])]
                for point in box
            ]
        self._confirmed_lines[str(line_id)] = payload
        self._checkpoint_version += 1

    def add_cache_reference(self, cache_key: str) -> None:
        self._cache_references.add(str(cache_key))
        self._checkpoint_version += 1

    def checkpoint(self) -> dict[str, Any]:
        return {
            "strategy_version": self.strategy_version,
            "source_sha256": self.source_sha256,
            "width": self.width,
            "height": self.height,
            "target_side": self.target_side,
            "model_fingerprint": self.model_fingerprint,
            "anchors": [list(anchor) for anchor in self.anchors],
            "regions": [
                asdict(self._regions[region_id])
                for region_id in sorted(self._regions)
            ],
            "pending_region_ids": sorted(self._pending),
            "active_region_ids": sorted(self._active),
            "resolved_regions": dict(sorted(self._resolved.items())),
            "split_region_ids": sorted(self._split),
            "completed_detection_batches": sorted(
                self._completed_detection_batches
            ),
            "completed_recognition_batches": sorted(
                self._completed_recognition_batches
            ),
            "confirmed_lines": [
                self._confirmed_lines[line_id]
                for line_id in sorted(
                    self._confirmed_lines,
                    key=lambda value: (
                        tuple(
                            self._confirmed_lines[value]["ordering_key"]
                        ),
                        value,
                    ),
                )
            ],
            "cache_references": sorted(self._cache_references),
            "checkpoint_version": self._checkpoint_version,
        }

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict[str, Any],
    ) -> "AdaptiveRegionPlanner":
        planner = cls(
            source_sha256=str(payload["source_sha256"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            target_side=int(payload["target_side"]),
            model_fingerprint=str(payload["model_fingerprint"]),
            strategy_version=str(payload["strategy_version"]),
            anchors=[
                tuple(int(value) for value in anchor)
                for anchor in payload.get("anchors", [])
            ],
        )
        planner._regions = {
            str(item["region_id"]): AdaptiveRegion(
                region_id=str(item["region_id"]),
                level=int(item["level"]),
                left=int(item["left"]),
                top=int(item["top"]),
                right=int(item["right"]),
                bottom=int(item["bottom"]),
                heat=float(item["heat"]),
                state=str(item.get("state") or "pending"),
            )
            for item in payload.get("regions", [])
        }
        planner._pending = {
            str(value) for value in payload.get("pending_region_ids", [])
        }
        # An active region was not confirmed and must be reclaimed after
        # recovery instead of being silently lost.
        planner._pending.update(
            str(value) for value in payload.get("active_region_ids", [])
        )
        planner._active = set()
        planner._resolved = {
            str(key): str(value)
            for key, value in dict(
                payload.get("resolved_regions") or {}
            ).items()
        }
        planner._split = {
            str(value) for value in payload.get("split_region_ids", [])
        }
        planner._completed_detection_batches = {
            str(value)
            for value in payload.get("completed_detection_batches", [])
        }
        planner._completed_recognition_batches = {
            str(value)
            for value in payload.get("completed_recognition_batches", [])
        }
        planner._confirmed_lines = {}
        for item in payload.get("confirmed_lines", []):
            restored_line: dict[str, Any] = {
                "line_id": str(item["line_id"]),
                "text": str(item.get("text") or ""),
                "confidence": (
                    float(item["confidence"])
                    if item.get("confidence") is not None
                    else None
                ),
                "ordering_key": [
                    int(item["ordering_key"][0]),
                    int(item["ordering_key"][1]),
                    int(item["ordering_key"][2]),
                    str(item["ordering_key"][3]),
                ],
            }
            if item.get("batch_id") is not None:
                restored_line["batch_id"] = str(item["batch_id"])
            if isinstance(item.get("box"), list):
                restored_line["box"] = [
                    [float(point[0]), float(point[1])]
                    for point in item["box"]
                ]
            planner._confirmed_lines[
                str(item["line_id"])
            ] = restored_line
        planner._cache_references = {
            str(value) for value in payload.get("cache_references", [])
        }
        planner._checkpoint_version = int(
            payload.get("checkpoint_version") or 0
        )
        return planner

    def _make_region(
        self,
        level: int,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> AdaptiveRegion:
        identity = (
            f"{self.source_sha256}|{self.strategy_version}|"
            f"{self.model_fingerprint}|level={level}|"
            f"{left},{top},{right},{bottom}|target={self.target_side}"
        )
        region_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return AdaptiveRegion(
            region_id=region_id,
            level=level,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            heat=self._heat(left, top, right, bottom),
        )

    def _heat(self, left: int, top: int, right: int, bottom: int) -> float:
        region_area = max(1, (right - left) * (bottom - top))
        intersected = 0
        for anchor_left, anchor_top, anchor_right, anchor_bottom in self.anchors:
            overlap_width = max(
                0,
                min(right, anchor_right) - max(left, anchor_left),
            )
            overlap_height = max(
                0,
                min(bottom, anchor_bottom) - max(top, anchor_top),
            )
            intersected += overlap_width * overlap_height
        return min(1.0, intersected / region_area)

    @staticmethod
    def _sort_key(region: AdaptiveRegion) -> tuple[float, int, int, int, str]:
        return (
            -region.heat,
            region.level,
            region.top,
            region.left,
            region.region_id,
        )

    def _require_unresolved(self, region_id: str) -> AdaptiveRegion:
        region = self._regions.get(region_id)
        if region is None:
            raise KeyError(region_id)
        if region_id in self._resolved or region_id in self._split:
            raise ValueError(f"Region is already closed: {region_id}")
        return region
