from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CodeChunk:

    chunk_id: str

    repo: str

    file_path: str

    file_name: str

    language: str

    chunk_type: str

    symbol: Optional[str]

    line_start: int

    line_end: int

    content: str

    imports: List[str] = field(
        default_factory=list
    )

    calls: List[str] = field(
        default_factory=list
    )

    dependencies: List[str] = field(
        default_factory=list
    )

    related_files: List[str] = field(
        default_factory=list
    )

    parent_symbol: Optional[str] = None

    metadata: dict = field(
        default_factory=dict
    )

    def to_dict(self):

        return {
            "chunk_id": self.chunk_id,

            "repo": self.repo,

            "file_path": self.file_path,

            "file_name": self.file_name,

            "language": self.language,

            "chunk_type": self.chunk_type,

            "symbol": self.symbol,

            "line_start": self.line_start,

            "line_end": self.line_end,

            "content": self.content,

            "imports": self.imports,

            "calls": self.calls,

            "dependencies": self.dependencies,

            "related_files": self.related_files,

            "parent_symbol": self.parent_symbol,

            "metadata": self.metadata,
        }