import ast
from pathlib import Path


class DependencyResolver:
    """
    Resolve Python imports to actual files
    inside a repository.
    """

    def __init__(self, repo_path: str):

        self.repo_path = Path(repo_path)

        self.python_files = []

        self.module_map = {}

        self._scan_repository()

    # --------------------------------------------------
    # STEP 1: Scan Python files
    # --------------------------------------------------

    def _scan_repository(self):

        for file_path in self.repo_path.rglob("*.py"):

            if any(
                part in {
                    ".git",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "node_modules",
                    "catboost_info",
                }
                for part in file_path.parts
            ):
                continue

            self.python_files.append(file_path)

            module_name = self._file_to_module(
                file_path
            )

            self.module_map[module_name] = file_path

    # --------------------------------------------------
    # STEP 2: Convert file path to Python module
    # --------------------------------------------------

    def _file_to_module(self, file_path: Path) -> str:

        relative_path = file_path.relative_to(
            self.repo_path
        )

        parts = list(relative_path.parts)

        # Remove .py
        parts[-1] = parts[-1][:-3]

        # __init__.py represents the package itself
        if parts[-1] == "__init__":

            parts = parts[:-1]

        return ".".join(parts)

    # --------------------------------------------------
    # STEP 3: Extract imports
    # --------------------------------------------------

    def _extract_imports(self, file_path: Path):

        try:

            source = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            tree = ast.parse(source)

        except Exception:

            return []

        imports = []

        for node in ast.walk(tree):

            # import src.utils

            if isinstance(node, ast.Import):

                for alias in node.names:

                    imports.append({
                        "module": alias.name,
                        "line": node.lineno,
                    })

            # from src.utils import evaluate_models

            elif isinstance(node, ast.ImportFrom):

                module = node.module or ""

                imports.append({
                    "module": module,
                    "symbols": [
                        alias.name
                        for alias in node.names
                    ],
                    "line": node.lineno,
                })

        return imports

    # --------------------------------------------------
    # STEP 4: Resolve module → actual file
    # --------------------------------------------------

    def resolve_module(self, module_name: str):

        # Direct match
        if module_name in self.module_map:

            return self.module_map[module_name]

        # Try parent modules
        parts = module_name.split(".")

        while parts:

            candidate = ".".join(parts)

            if candidate in self.module_map:

                return self.module_map[candidate]

            parts.pop()

        return None

    # --------------------------------------------------
    # STEP 5: Resolve dependencies of one file
    # --------------------------------------------------

    def resolve_file(self, file_path: Path):

        imports = self._extract_imports(
            file_path
        )

        dependencies = []

        for item in imports:

            module = item["module"]

            target = self.resolve_module(
                module
            )

            if target:

                dependency = {
                    "source_file": str(
                        file_path.relative_to(
                            self.repo_path
                        )
                    ),

                    "import": module,

                    "target_file": str(
                        target.relative_to(
                            self.repo_path
                        )
                    ),

                    "line": item["line"],
                }

                if "symbols" in item:

                    dependency["symbols"] = (
                        item["symbols"]
                    )

                dependencies.append(
                    dependency
                )

        return dependencies

    # --------------------------------------------------
    # STEP 6: Resolve complete repository
    # --------------------------------------------------

    def build_dependency_graph(self):

        graph = {}

        for file_path in self.python_files:

            dependencies = self.resolve_file(
                file_path
            )

            if dependencies:

                relative_file = str(
                    file_path.relative_to(
                        self.repo_path
                    )
                )

                graph[relative_file] = dependencies

        return graph


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    resolver = DependencyResolver(
        repo_path
    )

    graph = resolver.build_dependency_graph()

    print("\nCROSS-FILE DEPENDENCIES\n")

    for source, dependencies in graph.items():

        print(f"\n📄 {source}")

        for dependency in dependencies:

            print(
                f"   └── imports "
                f"{dependency['import']}"
            )

            print(
                f"       → "
                f"{dependency['target_file']}"
            )

            if "symbols" in dependency:

                print(
                    f"       symbols: "
                    f"{', '.join(dependency['symbols'])}"
                )

            print(
                f"       line: "
                f"{dependency['line']}"
            )