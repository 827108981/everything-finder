from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_full_text_search.rag.errors import RagModelMissingError, RagModelValidationError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    repo_id: str
    revision: str
    kind: str


class LocalModelManager:
    """Downloads fixed Hugging Face revisions and resolves them offline later."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()

    def download(self, spec: ModelSpec, cancel_token, progress_callback=None) -> Path:
        cancel_token.throw_if_cancelled()
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RagModelMissingError(
                "缺少 RAG 模型下载依赖，请安装 requirements-rag.txt"
            ) from exc
        if progress_callback:
            progress_callback({"stage": "downloading_model", "model": spec.repo_id})
        try:
            path = Path(
                snapshot_download(
                    repo_id=spec.repo_id,
                    revision=spec.revision,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False,
                )
            )
        except Exception as exc:
            raise RagModelMissingError(f"下载模型 {spec.repo_id} 失败：{exc}") from exc
        cancel_token.throw_if_cancelled()
        self.validate_files(path, spec)
        if progress_callback:
            progress_callback({"stage": "model_ready", "model": spec.repo_id})
        return path

    def resolve_local_path(self, spec: ModelSpec) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RagModelMissingError(
                "缺少 RAG 模型依赖，请安装 requirements-rag.txt"
            ) from exc
        try:
            path = Path(
                snapshot_download(
                    repo_id=spec.repo_id,
                    revision=spec.revision,
                    cache_dir=str(self.cache_dir),
                    local_files_only=True,
                )
            )
        except Exception as exc:
            raise RagModelMissingError(
                f"本地没有模型 {spec.repo_id}@{spec.revision}，请先点击下载模型"
            ) from exc
        self.validate_files(path, spec)
        return path

    @staticmethod
    def validate_files(path: Path, spec: ModelSpec) -> None:
        required = ["config.json"]
        if not (path / "model.safetensors").is_file() and not (path / "pytorch_model.bin").is_file():
            required.append("model.safetensors 或 pytorch_model.bin")
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise RagModelValidationError(
                f"模型 {spec.repo_id} 缺少必要文件：{', '.join(missing)}"
            )
