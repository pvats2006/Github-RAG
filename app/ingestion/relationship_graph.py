from pathlib import Path

from dependency_resolver import DependencyResolver
from code_parser import parse_python_file
from symbol_index import SymbolIndex


class RepositoryGraph:
    """
    Build a code-aware relationship graph
    for a Python repository.
    """

    def __init__(self, repo_path: str):

        self.repo_path = Path(repo_path)

        self.resolver = DependencyResolver(
            repo_path
        )

        self.symbol_index = SymbolIndex(
            repo_path
        )

        self.nodes = {}
        self.edges = []

        # file -> imported symbols
        self.file_imports = {}

    # --------------------------------------------------
    # NODE
    # --------------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_type: str,
        **metadata
    ):

        if node_id not in self.nodes:

            self.nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                **metadata,
            }

    # --------------------------------------------------
    # EDGE
    # --------------------------------------------------

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        **metadata
    ):

        edge = {
            "source": source,
            "relation": relation,
            "target": target,
            **metadata,
        }

        if edge not in self.edges:
            self.edges.append(edge)

    # --------------------------------------------------
    # IMPORT INDEX
    # --------------------------------------------------

    def build_import_index(self):

        """
        Build:

        file
          ↓
        imported symbol
          ↓
        source file
        """

        self.file_imports = {}

        dependency_graph = (
            self.resolver.build_dependency_graph()
        )

        for source, dependencies in (
            dependency_graph.items()
        ):

            self.file_imports[source] = {}

            for dependency in dependencies:

                target_file = dependency[
                    "target_file"
                ]

                symbols = dependency.get(
                    "symbols",
                    []
                )

                for symbol in symbols:

                    self.file_imports[
                        source
                    ][symbol] = target_file

    # --------------------------------------------------
    # FILE + CODE STRUCTURE
    # --------------------------------------------------

    def add_code_structure(
        self,
        file_path: Path
    ):

        result = parse_python_file(
            file_path
        )

        if not result:
            return

        relative_file = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        self.add_node(
            relative_file,
            "file"
        )

        # ----------------------------------------------
        # CLASSES
        # ----------------------------------------------

        for class_info in result["classes"]:

            class_id = (
                f"{relative_file}:"
                f"class:"
                f"{class_info['name']}"
            )

            self.add_node(
                class_id,
                "class",
                name=class_info["name"],
                file=relative_file,
                line_start=class_info[
                    "line_start"
                ],
                line_end=class_info[
                    "line_end"
                ],
            )

            self.add_edge(
                relative_file,
                "contains",
                class_id,
            )

            # ------------------------------------------
            # METHODS
            # ------------------------------------------

            for method in class_info[
                "methods"
            ]:

                method_id = (
                    f"{class_id}:"
                    f"method:"
                    f"{method['name']}"
                )

                self.add_node(
                    method_id,
                    "method",
                    name=method["name"],
                    file=relative_file,
                    line_start=method[
                        "line_start"
                    ],
                    line_end=method[
                        "line_end"
                    ],
                )

                self.add_edge(
                    class_id,
                    "contains",
                    method_id,
                )

                for call in method["calls"]:

                    self._add_call(
                        parent_id=method_id,
                        relative_file=relative_file,
                        call=call,
                    )

        # ----------------------------------------------
        # TOP LEVEL FUNCTIONS
        # ----------------------------------------------

        for function in result["functions"]:

            function_id = (
                f"{relative_file}:"
                f"function:"
                f"{function['name']}"
            )

            self.add_node(
                function_id,
                "function",
                name=function["name"],
                file=relative_file,
                line_start=function[
                    "line_start"
                ],
                line_end=function[
                    "line_end"
                ],
            )

            self.add_edge(
                relative_file,
                "contains",
                function_id,
            )

            for call in function["calls"]:

                self._add_call(
                    parent_id=function_id,
                    relative_file=relative_file,
                    call=call,
                )

    # --------------------------------------------------
    # ADD CALL
    # --------------------------------------------------

    def _add_call(
        self,
        parent_id: str,
        relative_file: str,
        call: dict,
    ):

        call_name = call["name"]
        call_line = call["line"]

        call_id = (
            f"{relative_file}:"
            f"call:"
            f"{call_name}:"
            f"{call_line}"
        )

        self.add_node(
            call_id,
            "call",
            name=call_name,
            file=relative_file,
            line=call_line,
        )

        self.add_edge(
            parent_id,
            "calls",
            call_id,
            line=call_line,
        )

    # --------------------------------------------------
    # RESOLVE CALL
    # --------------------------------------------------

    def resolve_call_targets(self):

        """
        Resolve only when we have enough evidence.

        Priority:

        1. Explicit imported symbol
        2. Same-file symbol
        3. Otherwise leave unresolved

        This avoids false matches for things like:

            __init__()
            predict()
            transform()
        """

        for edge in list(self.edges):

            if edge["relation"] != "calls":
                continue

            call_node_id = edge["target"]

            call_node = self.nodes.get(
                call_node_id
            )

            if not call_node:
                continue

            call_name = call_node.get(
                "name"
            )

            if not call_name:
                continue

            source_file = call_node.get(
                "file"
            )

            if not source_file:
                continue

            # --------------------------------------------------
            # Determine actual symbol name
            # --------------------------------------------------

            lookup_name = call_name.split(
                "."
            )[-1]

            # --------------------------------------------------
            # CASE 1: Explicit imported symbol
            # --------------------------------------------------

            imported_symbols = (
                self.file_imports.get(
                    source_file,
                    {}
                )
            )

            imported_target = (
                imported_symbols.get(
                    lookup_name
                )
            )

            if imported_target:

                self._resolve_to_file(
                    call_node_id,
                    lookup_name,
                    imported_target,
                    call_node,
                )

                continue

            # --------------------------------------------------
            # CASE 2: Same-file symbol
            # --------------------------------------------------

            same_file_matches = []

            for match in self.symbol_index.lookup(
                lookup_name
            ):

                if match["file"] == source_file:

                    same_file_matches.append(
                        match
                    )

            # Only resolve if exactly ONE
            # same-file definition exists.

            if len(same_file_matches) == 1:

                match = same_file_matches[0]

                self._resolve_to_symbol(
                    call_node_id,
                    match,
                    call_node,
                )

    # --------------------------------------------------
    # RESOLVE TO FILE
    # --------------------------------------------------

    def _resolve_to_file(
        self,
        call_node_id,
        symbol_name,
        target_file,
        call_node,
    ):

        matches = []

        for match in self.symbol_index.lookup(
            symbol_name
        ):

            if match["file"] == target_file:

                matches.append(
                    match
                )

        # Imported symbol should normally
        # have exactly one definition.

        if len(matches) != 1:
            return

        self._resolve_to_symbol(
            call_node_id,
            matches[0],
            call_node,
        )

    # --------------------------------------------------
    # RESOLVE TO SYMBOL
    # --------------------------------------------------

    def _resolve_to_symbol(
        self,
        call_node_id,
        match,
        call_node,
    ):

        target_id = (
            f"{match['file']}:"
            f"{match['type']}:"
            f"{match['name']}"
        )

        self.add_node(
            target_id,
            match["type"],
            name=match["name"],
            file=match["file"],
            line_start=match[
                "line_start"
            ],
            line_end=match[
                "line_end"
            ],
        )

        self.add_edge(
            call_node_id,
            "resolves_to",
            target_id,
            line=call_node.get(
                "line"
            ),
        )

    # --------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------

    def build(self):

        # ----------------------------------------------
        # 1. Build import index
        # ----------------------------------------------

        self.build_import_index()

        # ----------------------------------------------
        # 2. File dependencies
        # ----------------------------------------------

        dependency_graph = (
            self.resolver.build_dependency_graph()
        )

        for source, dependencies in (
            dependency_graph.items()
        ):

            self.add_node(
                source,
                "file"
            )

            for dependency in dependencies:

                target = dependency[
                    "target_file"
                ]

                self.add_node(
                    target,
                    "file"
                )

                self.add_edge(
                    source,
                    "imports",
                    target,
                    line=dependency[
                        "line"
                    ],
                    symbols=dependency.get(
                        "symbols",
                        []
                    ),
                )

        # ----------------------------------------------
        # 3. Code structure
        # ----------------------------------------------

        for file_path in (
            self.resolver.python_files
        ):

            self.add_code_structure(
                file_path
            )

        # ----------------------------------------------
        # 4. Resolve calls
        # ----------------------------------------------

        self.resolve_call_targets()

        return {
            "nodes": list(
                self.nodes.values()
            ),
            "edges": self.edges,
        }


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    graph_builder = RepositoryGraph(
        repo_path
    )

    graph = graph_builder.build()

    print("\nNODES\n")

    for node in graph["nodes"]:

        print(
            f"{node['type']:10} "
            f"{node['id']}"
        )

    print("\nRELATIONSHIPS\n")

    for edge in graph["edges"]:

        print(
            f"{edge['source']}"
            f" --[{edge['relation']}]--> "
            f"{edge['target']}"
        )

        if "line" in edge:

            print(
                f"    line: "
                f"{edge['line']}"
            )

        if edge.get("symbols"):

            print(
                f"    symbols: "
                f"{', '.join(edge['symbols'])}"
            )