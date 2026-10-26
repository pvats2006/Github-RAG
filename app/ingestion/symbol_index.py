import ast
from pathlib import Path


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "catboost_info",
}


class SymbolIndex:

    def __init__(self, repo_path: str):

        self.repo_path = Path(repo_path)

        self.symbols = {}

        self._build_index()

    # --------------------------------------------------
    # Build symbol index
    # --------------------------------------------------

    def _build_index(self):

        for file_path in self.repo_path.rglob("*.py"):

            if any(
                part in IGNORED_DIRS
                for part in file_path.parts
            ):
                continue

            self._parse_file(file_path)

    # --------------------------------------------------
    # Parse one Python file
    # --------------------------------------------------

    def _parse_file(self, file_path: Path):

        try:

            source = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            tree = ast.parse(source)

        except Exception:
            return

        relative_file = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        for node in tree.body:

            # -----------------------------
            # Top-level function
            # -----------------------------

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                )
            ):

                self._add_symbol(
                    name=node.name,
                    symbol_type="function",
                    file=relative_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                )

            # -----------------------------
            # Class
            # -----------------------------

            elif isinstance(
                node,
                ast.ClassDef
            ):

                self._add_symbol(
                    name=node.name,
                    symbol_type="class",
                    file=relative_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                )

                # -------------------------
                # Class methods
                # -------------------------

                for child in node.body:

                    if isinstance(
                        child,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        )
                    ):

                        full_name = (
                            f"{node.name}."
                            f"{child.name}"
                        )

                        self._add_symbol(
                            name=child.name,
                            qualified_name=full_name,
                            symbol_type="method",
                            file=relative_file,
                            line_start=child.lineno,
                            line_end=child.end_lineno,
                        )

    # --------------------------------------------------
    # Add symbol
    # --------------------------------------------------

    def _add_symbol(
        self,
        name,
        symbol_type,
        file,
        line_start,
        line_end,
        qualified_name=None,
    ):

        symbol = {
            "name": name,
            "type": symbol_type,
            "file": file,
            "line_start": line_start,
            "line_end": line_end,
        }

        if qualified_name:
            symbol["qualified_name"] = (
                qualified_name
            )

        if name not in self.symbols:

            self.symbols[name] = []

        self.symbols[name].append(
            symbol
        )

    # --------------------------------------------------
    # Lookup
    # --------------------------------------------------

    def lookup(self, name):

        return self.symbols.get(
            name,
            []
        )


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    index = SymbolIndex(
        repo_path
    )

    print("\nSYMBOL INDEX\n")

    for name, locations in (
        index.symbols.items()
    ):

        print(f"\n{name}")

        for location in locations:

            print(
                f"  {location['type']} → "
                f"{location['file']} "
                f""
                f"({location['line_start']}-"
                f"{location['line_end']})"
            )