from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EmbeddingModelOption:
    key: str
    label: str
    repo_id: str
    revision: str
    dimension: int
    description: str


EMBEDDING_MODEL_OPTIONS = (
    EmbeddingModelOption(
        key="bge-small-zh-v1.5",
        label="BGE Small 中文（速度优先）",
        repo_id="BAAI/bge-small-zh-v1.5",
        revision="7999e1d3359715c523056ef9478215996d62a620",
        dimension=512,
        description="适合纯 CPU 日常使用，建库和查询速度较快。",
    ),
    EmbeddingModelOption(
        key="bge-large-zh-v1.5",
        label="BGE Large 中文（质量优先）",
        repo_id="BAAI/bge-large-zh-v1.5",
        revision="79e7739b6ab944e86d6171e44d24c997fc1e0116",
        dimension=1024,
        description="检索质量更好；纯 CPU 首次完整建库可能需要数小时。",
    ),
    EmbeddingModelOption(
        key="qwen3-embedding-0.6b",
        label="Qwen3 Embedding 0.6B（实验性）",
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        dimension=1024,
        description="支持查询任务指令；本机 CPU 建库明显慢于 BGE Large，建议仅用于实验。",
    ),
)


def embedding_model_option(
    repo_id: str, revision: str = "",
) -> EmbeddingModelOption | None:
    """按固定仓库和 revision 查找受支持模型，防止同名模型版本混用。"""
    for option in EMBEDDING_MODEL_OPTIONS:
        if option.repo_id != str(repo_id):
            continue
        if revision and option.revision != str(revision):
            continue
        return option
    return None


def embedding_model_cache_path(
    cache_dir: Path, option: EmbeddingModelOption,
) -> Path:
    repository = "models--" + option.repo_id.replace("/", "--")
    return Path(cache_dir).resolve() / repository / "snapshots" / option.revision


def embedding_model_is_downloaded(
    cache_dir: Path, option: EmbeddingModelOption,
) -> bool:
    path = embedding_model_cache_path(cache_dir, option)
    if not (path / "config.json").is_file():
        return False
    return (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
