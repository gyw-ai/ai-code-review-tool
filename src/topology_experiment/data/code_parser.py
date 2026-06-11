"""Parse source code into token sequences and AST graphs."""

import os
import re


def parse_file(file_path):
    """Read a code file and return parsed representation.

    Returns dict with:
      - path: file path
      - language: file extension
      - content: raw text
      - lines: list of line strings
      - tokens: list of token strings
      - functions: list of function boundaries
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    lines = content.split("\n")
    ext = os.path.splitext(file_path)[1].lstrip(".")

    tokens = _tokenize(content)
    functions = _extract_functions(content, ext)

    return {
        "path": file_path,
        "language": ext,
        "content": content,
        "lines": lines,
        "tokens": tokens,
        "functions": functions,
    }


def _tokenize(text):
    import re
    return re.findall(r"[a-zA-Z_]\w*|[\d]+|\S", text)


def _extract_functions(content, language):
    """Find function/class boundaries using simple regex patterns."""
    functions = []
    if language == "py":
        pattern = r"^(def |class |async def )"
    elif language in ("js", "ts"):
        pattern = r"^(function |class |const \w+ = \(\)|export default)"
    elif language == "go":
        pattern = r"^func "
    elif language == "java":
        pattern = r"^\s*(public|private|protected).*\(.*\).*\{"
    else:
        pattern = r"^[a-zA-Z_]\w*\s*\("

    for i, line in enumerate(content.split("\n"), 1):
        if re.match(pattern, line.strip()):
            functions.append({"line": i, "name": line.strip()[:40]})
    return functions


def build_ast_graph(file_path):
    """Build adjacency list of AST nodes using tree-sitter.

    Returns dict with:
      - nodes: list of {id, type, line, text}
      - edges: list of (parent_id, child_id)
    """
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python

        lang = Language(tree_sitter_python.language())
        parser = Parser(lang)

        with open(file_path, "rb") as f:
            tree = parser.parse(f.read())
    except Exception:
        return _simple_ast_fallback(file_path)

    nodes = []
    edges = []
    _walk_tree(tree.root_node, None, nodes, edges)

    return {"nodes": nodes, "edges": edges, "root": 0}


def _walk_tree(node, parent_id, nodes, edges):
    node_id = len(nodes)
    nodes.append({
        "id": node_id,
        "type": node.type,
        "line": node.start_point[0] + 1,
        "text": node.text[:40].decode("utf-8", errors="replace") if node.text else "",
    })
    if parent_id is not None:
        edges.append((parent_id, node_id))
    for child in node.children:
        _walk_tree(child, node_id, nodes, edges)


def _simple_ast_fallback(file_path):
    """Simple indentation-based tree when tree-sitter is unavailable."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    nodes = [{"id": 0, "type": "root", "line": 0, "text": os.path.basename(file_path)}]
    edges = []
    stack = [(0, -1)]

    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        node_id = len(nodes)
        is_def = bool(re.match(r"^\s*(def |class |function |if |for |while )", stripped))

        nodes.append({
            "id": node_id,
            "type": "definition" if is_def else "statement",
            "line": i + 1,
            "text": stripped[:60],
        })

        while stack and stack[-1][1] >= indent:
            stack.pop()
        if stack:
            edges.append((stack[-1][0], node_id))
        stack.append((node_id, indent))

    return {"nodes": nodes, "edges": edges, "root": 0}
