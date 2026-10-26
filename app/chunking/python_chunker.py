import ast
from pathlib import Path

from app.chunking.chunk_schema import CodeChunk


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "catboost_info",
}


class PythonChunker:
    """
    Structure-aware Python code chunker.

    Normal code units:
        file
        class
        method
        function

    Large code units:
        logically split into smaller
        AST-safe blocks.
    """

    # Maximum number of source lines
    # allowed in a normal chunk.
    MAX_CHUNK_LINES = 120

    def __init__(
        self,
        repo_path: str
    ):

        self.repo_path = Path(
            repo_path
        )

        self.repo_name = (
            self.repo_path.name
        )

    # --------------------------------------------------
    # READ FILE
    # --------------------------------------------------

    def read_file(
        self,
        file_path: Path
    ):

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    # --------------------------------------------------
    # GET SOURCE LINES
    # --------------------------------------------------

    def get_source_lines(
        self,
        source: str
    ):

        return source.splitlines()

    # --------------------------------------------------
    # GET NODE SOURCE
    # --------------------------------------------------

    def get_node_source(
        self,
        lines,
        node
    ):

        if not hasattr(
            node,
            "lineno"
        ):

            return ""

        if not hasattr(
            node,
            "end_lineno"
        ):

            return ""

        return "\n".join(
            lines[
                node.lineno - 1:
                node.end_lineno
            ]
        )

    # --------------------------------------------------
    # NODE SIZE
    # --------------------------------------------------

    def get_node_size(
        self,
        node
    ):

        if not hasattr(
            node,
            "lineno"
        ):

            return 0

        if not hasattr(
            node,
            "end_lineno"
        ):

            return 0

        return (
            node.end_lineno
            - node.lineno
            + 1
        )

    # --------------------------------------------------
    # CHUNK ID
    # --------------------------------------------------

    def make_chunk_id(
        self,
        file_path,
        chunk_type,
        symbol
    ):

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        return (
            f"{relative_path}"
            f"::{chunk_type}"
            f"::{symbol}"
        )

    # --------------------------------------------------
    # CREATE CHUNK
    # --------------------------------------------------

    def create_chunk(
        self,
        file_path,
        chunk_type,
        symbol,
        node,
        content,
        parent_symbol=None,
        metadata=None,
    ):

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        chunk_id = self.make_chunk_id(
            file_path,
            chunk_type,
            symbol
        )

        return CodeChunk(

            chunk_id=chunk_id,

            repo=self.repo_name,

            file_path=relative_path,

            file_name=file_path.name,

            language="python",

            chunk_type=chunk_type,

            symbol=symbol,

            line_start=node.lineno,

            line_end=node.end_lineno,

            content=content,

            parent_symbol=parent_symbol,

            metadata=metadata or {},
        )

    # --------------------------------------------------
    # CREATE LARGE NODE SUB-CHUNKS
    # --------------------------------------------------

    def split_large_node(
        self,
        file_path,
        node,
        symbol,
        chunk_type,
        lines,
        parent_symbol=None,
    ):

        """
        Split a large function/method/class
        using AST child statement boundaries.

        We do NOT split arbitrary text.
        """

        children = getattr(
            node,
            "body",
            []
        )

        if not children:

            content = (
                self.get_node_source(
                    lines,
                    node
                )
            )

            return [
                self.create_chunk(
                    file_path=file_path,
                    chunk_type=chunk_type,
                    symbol=symbol,
                    node=node,
                    content=content,
                    parent_symbol=parent_symbol,
                )
            ]

        chunks = []

        current_nodes = []

        current_start = None
        current_end = None

        block_number = 1

        for child in children:

            if not hasattr(
                child,
                "lineno"
            ):
                continue

            if not hasattr(
                child,
                "end_lineno"
            ):
                continue

            child_start = (
                child.lineno
            )

            child_end = (
                child.end_lineno
            )

            # ------------------------------------------
            # Start new logical block
            # ------------------------------------------

            if current_start is None:

                current_start = (
                    child_start
                )

                current_end = (
                    child_end
                )

                current_nodes = [
                    child
                ]

                continue

            proposed_end = child_end

            proposed_size = (
                proposed_end
                - current_start
                + 1
            )

            # ------------------------------------------
            # If adding this statement exceeds
            # threshold, close current block.
            # ------------------------------------------

            if (
                proposed_size
                > self.MAX_CHUNK_LINES
            ):

                block_content = "\n".join(
                    lines[
                        current_start - 1:
                        current_end
                    ]
                )

                block_symbol = (
                    f"{symbol}"
                    f"::block_{block_number}"
                )

                block_chunks = (
                    self._create_block_chunk(
                        file_path=file_path,
                        chunk_type=(
                            f"{chunk_type}_block"
                        ),
                        symbol=block_symbol,
                        start_line=current_start,
                        end_line=current_end,
                        content=block_content,
                        parent_symbol=(
                            parent_symbol
                            or symbol
                        ),
                    )
                )

                chunks.extend(
                    block_chunks
                )

                block_number += 1

                # Start new block
                current_start = (
                    child_start
                )

                current_end = (
                    child_end
                )

                current_nodes = [
                    child
                ]

            else:

                current_nodes.append(
                    child
                )

                current_end = (
                    child_end
                )

        # ----------------------------------------------
        # Flush final block
        # ----------------------------------------------

        if (
            current_start is not None
            and current_end is not None
        ):

            block_content = "\n".join(
                lines[
                    current_start - 1:
                    current_end
                ]
            )

            block_symbol = (
                f"{symbol}"
                f"::block_{block_number}"
            )

            block_chunks = (
                self._create_block_chunk(
                    file_path=file_path,
                    chunk_type=(
                        f"{chunk_type}_block"
                    ),
                    symbol=block_symbol,
                    start_line=current_start,
                    end_line=current_end,
                    content=block_content,
                    parent_symbol=(
                        parent_symbol
                        or symbol
                    ),
                )
            )

            chunks.extend(
                block_chunks
            )

        return chunks

    # --------------------------------------------------
    # CREATE BLOCK CHUNK
    # --------------------------------------------------

    def _create_block_chunk(
        self,
        file_path,
        chunk_type,
        symbol,
        start_line,
        end_line,
        content,
        parent_symbol=None,
    ):

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        chunk_id = (
            f"{relative_path}"
            f"::{chunk_type}"
            f"::{symbol}"
        )

        # We need a lightweight node-like
        # object for create_chunk.
        class LineNode:
            lineno: int = 0
            end_lineno: int = 0

        line_node = LineNode()

        line_node.lineno = (
            start_line
        )

        line_node.end_lineno = (
            end_line
        )

        return [
            CodeChunk(

                chunk_id=chunk_id,

                repo=self.repo_name,

                file_path=relative_path,

                file_name=file_path.name,

                language="python",

                chunk_type=chunk_type,

                symbol=symbol,

                line_start=start_line,

                line_end=end_line,

                content=content,

                parent_symbol=parent_symbol,

                metadata={
                    "is_sub_chunk": True,
                    "parent_symbol": (
                        parent_symbol
                    ),
                },
            )
        ]

    # --------------------------------------------------
    # PROCESS CODE UNIT
    # --------------------------------------------------

    def process_code_unit(
        self,
        file_path,
        node,
        chunk_type,
        symbol,
        lines,
        parent_symbol=None,
    ):

        size = self.get_node_size(
            node
        )

        # ----------------------------------------------
        # Normal chunk
        # ----------------------------------------------

        if (
            size
            <= self.MAX_CHUNK_LINES
        ):

            content = (
                self.get_node_source(
                    lines,
                    node
                )
            )

            return [
                self.create_chunk(
                    file_path=file_path,
                    chunk_type=chunk_type,
                    symbol=symbol,
                    node=node,
                    content=content,
                    parent_symbol=parent_symbol,
                )
            ]

        # ----------------------------------------------
        # Large chunk
        # ----------------------------------------------

        print(
            f"Large {chunk_type}: "
            f"{symbol} "
            f"({size} lines)"
        )

        return self.split_large_node(
            file_path=file_path,
            node=node,
            symbol=symbol,
            chunk_type=chunk_type,
            lines=lines,
            parent_symbol=parent_symbol,
        )

    # --------------------------------------------------
    # PARSE FILE
    # --------------------------------------------------

    def chunk_file(
        self,
        file_path: Path
    ):

        source = self.read_file(
            file_path
        )

        if not source.strip():

            return []

        try:

            tree = ast.parse(
                source
            )

        except SyntaxError as e:

            print(
                f"Skipping invalid Python: "
                f"{file_path}: {e}"
            )

            return []

        lines = (
            self.get_source_lines(
                source
            )
        )

        chunks = []

        # ----------------------------------------------
        # FILE CHUNK
        # ----------------------------------------------

        file_chunk = CodeChunk(

            chunk_id=(
                f"{file_path.relative_to(self.repo_path)}"
                f"::file"
            ),

            repo=self.repo_name,

            file_path=str(
                file_path.relative_to(
                    self.repo_path
                )
            ),

            file_name=file_path.name,

            language="python",

            chunk_type="file",

            symbol=None,

            line_start=1,

            line_end=len(lines),

            content=source,
        )

        chunks.append(
            file_chunk
        )

        # ----------------------------------------------
        # TOP LEVEL NODES
        # ----------------------------------------------

        for node in tree.body:

            # ==========================================
            # CLASS
            # ==========================================

            if isinstance(
                node,
                ast.ClassDef
            ):

                class_chunks = (
                    self.process_code_unit(
                        file_path=file_path,
                        node=node,
                        chunk_type="class",
                        symbol=node.name,
                        lines=lines,
                    )
                )

                chunks.extend(
                    class_chunks
                )

                # --------------------------------------
                # METHODS
                # --------------------------------------

                for child in node.body:

                    if not isinstance(
                        child,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        )
                    ):
                        continue

                    qualified_name = (
                        f"{node.name}."
                        f"{child.name}"
                    )

                    method_chunks = (
                        self.process_code_unit(
                            file_path=file_path,
                            node=child,
                            chunk_type="method",
                            symbol=qualified_name,
                            lines=lines,
                            parent_symbol=node.name,
                        )
                    )

                    chunks.extend(
                        method_chunks
                    )

            # ==========================================
            # TOP LEVEL FUNCTION
            # ==========================================

            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                )
            ):

                function_chunks = (
                    self.process_code_unit(
                        file_path=file_path,
                        node=node,
                        chunk_type="function",
                        symbol=node.name,
                        lines=lines,
                    )
                )

                chunks.extend(
                    function_chunks
                )

        return chunks

    # --------------------------------------------------
    # CHUNK REPOSITORY
    # --------------------------------------------------

    def chunk_repository(self):

        all_chunks = []

        for file_path in self.repo_path.rglob(
            "*.py"
        ):

            if any(
                part in IGNORED_DIRS
                for part in file_path.parts
            ):
                continue

            print(
                f"Chunking: "
                f"{file_path}"
            )

            chunks = self.chunk_file(
                file_path
            )

            all_chunks.extend(
                chunks
            )

        return all_chunks


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    chunker = PythonChunker(
        repo_path
    )

    chunks = (
        chunker.chunk_repository()
    )

    print(
        "\nPYTHON CHUNKS\n"
    )

    for chunk in chunks:

        print(
            f"{chunk.chunk_type:18} "
            f"{str(chunk.symbol):60} "
            f"{chunk.line_start}-"
            f"{chunk.line_end}"
        )

    print(
        f"\nTotal chunks: "
        f"{len(chunks)}"
    )