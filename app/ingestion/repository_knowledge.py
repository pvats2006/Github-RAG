import json
from pathlib import Path

from dependency_resolver import DependencyResolver
from code_parser import parse_python_file
from relationship_graph import RepositoryGraph


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "catboost_info",
}


SUPPORTED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".html",
    ".css",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
}


class RepositoryKnowledge:
    """
    Build one unified knowledge representation
    of an entire repository.
    """

    def __init__(self, repo_path: str):

        self.repo_path = Path(repo_path)

        self.graph_builder = RepositoryGraph(
            repo_path
        )

        self.resolver = DependencyResolver(
            repo_path
        )

        self.knowledge = {
            "repository": self.repo_path.name,
            "root_path": str(self.repo_path),
            "files": [],
            "relationships": [],
        }

    # --------------------------------------------------
    # CHECK IGNORED PATH
    # --------------------------------------------------

    def is_ignored(self, path: Path):

        return any(
            part in IGNORED_DIRS
            for part in path.parts
        )

    # --------------------------------------------------
    # READ FILE
    # --------------------------------------------------

    def read_file(self, file_path: Path):

        try:

            return file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except Exception as e:

            print(
                f"Warning: Could not read "
                f"{file_path}: {e}"
            )

            return ""

    # --------------------------------------------------
    # DETERMINE LANGUAGE
    # --------------------------------------------------

    def get_language(self, file_path: Path):

        extension = file_path.suffix.lower()

        mapping = {
            ".py": "python",
            ".md": "markdown",
            ".txt": "text",
            ".html": "html",
            ".css": "css",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
        }

        return mapping.get(
            extension,
            "unknown"
        )

    # --------------------------------------------------
    # GET PYTHON SYMBOLS
    # --------------------------------------------------

    def get_python_symbols(
        self,
        file_path: Path,
    ):

        result = parse_python_file(
            file_path
        )

        if not result:

            return []

        symbols = []

        # ----------------------------------------------
        # Classes
        # ----------------------------------------------

        for class_info in result["classes"]:

            class_name = class_info[
                "name"
            ]

            symbols.append(
                {
                    "name": class_name,
                    "type": "class",
                    "line_start": class_info[
                        "line_start"
                    ],
                    "line_end": class_info[
                        "line_end"
                    ],
                }
            )

            # ------------------------------------------
            # Methods
            # ------------------------------------------

            for method in class_info[
                "methods"
            ]:

                symbols.append(
                    {
                        "name": method[
                            "name"
                        ],
                        "qualified_name": (
                            f"{class_name}."
                            f"{method['name']}"
                        ),
                        "type": "method",
                        "line_start": method[
                            "line_start"
                        ],
                        "line_end": method[
                            "line_end"
                        ],
                    }
                )

        # ----------------------------------------------
        # Functions
        # ----------------------------------------------

        for function in result[
            "functions"
        ]:

            symbols.append(
                {
                    "name": function[
                        "name"
                    ],
                    "type": "function",
                    "line_start": function[
                        "line_start"
                    ],
                    "line_end": function[
                        "line_end"
                    ],
                }
            )

        return symbols

    # --------------------------------------------------
    # GET IMPORTS FOR FILE
    # --------------------------------------------------

    def get_file_imports(
        self,
        relative_file: str,
    ):

        dependencies = (
            self.resolver
            .build_dependency_graph()
            .get(
                relative_file,
                []
            )
        )

        imports = []

        for dependency in dependencies:

            imports.append(
                {
                    "target_file": dependency[
                        "target_file"
                    ],
                    "symbols": dependency.get(
                        "symbols",
                        []
                    ),
                    "line": dependency[
                        "line"
                    ],
                }
            )

        return imports

    # --------------------------------------------------
    # GET RELATIONSHIPS FOR FILE
    # --------------------------------------------------

    def get_file_relationships(
        self,
        relative_file: str,
        graph,
    ):

        relationships = []

        for edge in graph["edges"]:

            source = edge[
                "source"
            ]

            target = edge[
                "target"
            ]

            # ------------------------------------------
            # Relationship belongs to this file
            # ------------------------------------------

            if (
                relative_file in source
                or relative_file in target
            ):

                relationships.append(
                    {
                        "source": source,
                        "relation": edge[
                            "relation"
                        ],
                        "target": target,
                        **{
                            key: value
                            for key, value
                            in edge.items()
                            if key
                            not in {
                                "source",
                                "relation",
                                "target",
                            }
                        },
                    }
                )

        return relationships

    # --------------------------------------------------
    # BUILD FILE KNOWLEDGE
    # --------------------------------------------------

    def build_file_knowledge(
        self,
        file_path: Path,
        graph,
    ):

        relative_file = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        content = self.read_file(
            file_path
        )

        file_info = {
            "path": relative_file,

            "name": file_path.name,

            "extension": file_path.suffix,

            "language": self.get_language(
                file_path
            ),

            "size_bytes": file_path.stat().st_size,

            "line_count": (
                content.count("\n") + 1
                if content
                else 0
            ),

            "content": content,

            "symbols": [],

            "imports": [],

            "relationships": [],
        }

        # ----------------------------------------------
        # Python-specific analysis
        # ----------------------------------------------

        if file_path.suffix.lower() == ".py":

            file_info["symbols"] = (
                self.get_python_symbols(
                    file_path
                )
            )

            file_info["imports"] = (
                self.get_file_imports(
                    relative_file
                )
            )

        # ----------------------------------------------
        # Relationships
        # ----------------------------------------------

        file_info[
            "relationships"
        ] = self.get_file_relationships(
            relative_file,
            graph,
        )

        return file_info

    # --------------------------------------------------
    # BUILD
    # --------------------------------------------------

    def build(self):

        print(
            "\nBuilding repository knowledge..."
        )

        # ----------------------------------------------
        # Build relationship graph
        # ----------------------------------------------

        graph = (
            self.graph_builder.build()
        )

        # ----------------------------------------------
        # Scan repository files
        # ----------------------------------------------

        for file_path in self.repo_path.rglob("*"):

            if not file_path.is_file():
                continue

            if self.is_ignored(
                file_path
            ):
                continue

            if (
                file_path.suffix.lower()
                not in SUPPORTED_EXTENSIONS
            ):
                continue

            print(
                f"Processing: "
                f"{file_path}"
            )

            file_info = (
                self.build_file_knowledge(
                    file_path,
                    graph,
                )
            )

            self.knowledge[
                "files"
            ].append(
                file_info
            )

        # ----------------------------------------------
        # Store all relationships
        # ----------------------------------------------

        self.knowledge[
            "relationships"
        ] = graph["edges"]

        # ----------------------------------------------
        # Statistics
        # ----------------------------------------------

        self.knowledge[
            "statistics"
        ] = {
            "total_files": len(
                self.knowledge[
                    "files"
                ]
            ),

            "total_relationships": len(
                graph["edges"]
            ),

            "total_nodes": len(
                graph["nodes"]
            ),
        }

        return self.knowledge

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    def save(
        self,
        output_path: str = (
            "data/"
            "repository_knowledge.json"
        ),
    ):

        output_file = Path(
            output_path
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.knowledge,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\nKnowledge saved:"
        )

        print(
            output_file
        )


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    knowledge_builder = (
        RepositoryKnowledge(
            repo_path
        )
    )

    knowledge = (
        knowledge_builder.build()
    )

    knowledge_builder.save()

    print(
        "\nRepository Knowledge Summary"
    )

    print(
        "----------------------------"
    )

    print(
        f"Repository: "
        f"{knowledge['repository']}"
    )

    print(
        f"Files: "
        f"{knowledge['statistics']['total_files']}"
    )

    print(
        f"Nodes: "
        f"{knowledge['statistics']['total_nodes']}"
    )

    print(
        f"Relationships: "
        f"{knowledge['statistics']['total_relationships']}"
    )