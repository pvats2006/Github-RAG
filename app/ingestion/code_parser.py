import ast
from pathlib import Path


def get_imports(tree):
    """Extract imports from the AST."""

    imports = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):

            for alias in node.names:

                imports.append({
                    "name": alias.name,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                })

        elif isinstance(node, ast.ImportFrom):

            module = node.module or ""

            for alias in node.names:

                imports.append({
                    "name": f"{module}.{alias.name}",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                })

    return imports


def get_function_name(call_node):
    """
    Extract the function name from an AST Call node.
    """

    function = call_node.func

    # Example:
    # evaluate_models(...)
    if isinstance(function, ast.Name):
        return function.id

    # Example:
    # self.evaluate_models(...)
    if isinstance(function, ast.Attribute):

        if isinstance(function.value, ast.Name):

            return f"{function.value.id}.{function.attr}"

        return function.attr

    return "unknown"


def get_calls(node):
    """
    Extract function calls from a function/class/method.
    """

    calls = []

    for child in ast.walk(node):

        if isinstance(child, ast.Call):

            calls.append({
                "name": get_function_name(child),
                "line": child.lineno,
            })

    return calls


def get_function_info(node):
    """
    Extract detailed information about a function/method.
    """

    return {
        "name": node.name,
        "line_start": node.lineno,
        "line_end": node.end_lineno,
        "arguments": [
            arg.arg
            for arg in node.args.args
        ],
        "calls": get_calls(node),
    }


def get_classes(tree):
    """
    Extract classes and their methods.
    """

    classes = []

    for node in tree.body:

        if not isinstance(node, ast.ClassDef):
            continue

        class_info = {
            "name": node.name,
            "line_start": node.lineno,
            "line_end": node.end_lineno,
            "methods": [],
        }

        for child in node.body:

            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            ):

                class_info["methods"].append(
                    get_function_info(child)
                )

        classes.append(class_info)

    return classes


def get_functions(tree):
    """
    Extract top-level functions only.
    """

    functions = []

    for node in tree.body:

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):

            functions.append(
                get_function_info(node)
            )

    return functions


from typing import List, Optional


def parse_python_file(file_path: Path) -> Optional[dict]:
    """
    Parse a Python file using AST.
    """

    try:

        source_code = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        tree = ast.parse(source_code)

    except Exception as e:

        print(
            f"Could not parse {file_path}: {e}"
        )

        return None

    return {
        "file_path": str(file_path),
        "imports": get_imports(tree),
        "classes": get_classes(tree),
        "functions": get_functions(tree),
    }


def print_result(result):

    print("\nFILE:")
    print(result["file_path"])

    print("\nIMPORTS:")

    for item in result["imports"]:

        print(
            f"  {item['name']} "
            f"({item['line_start']}-"
            f"{item['line_end']})"
        )

    print("\nCLASSES:")

    for class_info in result["classes"]:

        print(
            f"\n  Class: {class_info['name']} "
            f"({class_info['line_start']}-"
            f"{class_info['line_end']})"
        )

        print("    Methods:")

        for method in class_info["methods"]:

            args = ", ".join(
                method["arguments"]
            )

            print(
                f"      {method['name']}({args}) "
                f"({method['line_start']}-"
                f"{method['line_end']})"
            )

            if method["calls"]:

                print("        Calls:")

                for call in method["calls"]:

                    print(
                        f"          {call['name']} "
                        f"(line {call['line']})"
                    )

    print("\nFUNCTIONS:")

    for function in result["functions"]:

        args = ", ".join(
            function["arguments"]
        )

        print(
            f"  {function['name']}({args}) "
            f"({function['line_start']}-"
            f"{function['line_end']})"
        )

        if function["calls"]:

            print("    Calls:")

            for call in function["calls"]:

                print(
                    f"      {call['name']} "
                    f"(line {call['line']})"
                )


if __name__ == "__main__":

    file_path = Path(
        input(
            "Enter Python file path: "
        ).strip()
    )

    result = parse_python_file(file_path)

    if result:
        print_result(result)