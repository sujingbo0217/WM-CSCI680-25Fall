import ast

def extract_if_conditions(code):
    """Extract textual if-conditions from Python source using AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    conditions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            try:
                cond = ast.get_source_segment(code, node.test)
                if cond and len(cond.strip()) > 0:
                    conditions.append(cond.strip())
            except Exception:
                continue
    return conditions
    