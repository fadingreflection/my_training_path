"""tree-sitter C/C++ parse + identifier extraction."""
from __future__ import annotations

from dataclasses import dataclass

from tree_sitter import Language, Parser
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

IDENT_TYPES = {"identifier", "field_identifier", "type_identifier"}
TYPE_NODE_TYPES = {"primitive_type", "sized_type_specifier"}
PREPROC_DEF_TYPES = {"preproc_def", "preproc_function_def"}
CATEGORIES = ("param", "local", "field", "callee", "type", "macro", "other")

_C_PARSER = Parser(Language(tsc.language()))
_CPP_PARSER = Parser(Language(tscpp.language()))


@dataclass
class ParseResult:
    parse_ok: bool
    parser: str
    identifiers: list[str]
    identifier_set: set[str]
    by_category: dict[str, list[str]]
    error: str = ""


def _text(node) -> str:
    try:
        return node.text.decode("utf-8", "ignore")
    except Exception:
        return ""


def _preproc_name(node) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _text(name_node).strip()
    for child in node.children:
        if child.type == "identifier":
            return _text(child).strip()
    return ""


def _collect_idents(node, out: list[str]) -> None:
    ntype = node.type
    if ntype in IDENT_TYPES | TYPE_NODE_TYPES:
        text = _text(node).strip()
        if text:
            out.append(text)
    elif ntype in PREPROC_DEF_TYPES:
        name = _preproc_name(node)
        if name:
            out.append(name)
    for child in node.children:
        _collect_idents(child, out)


def _is_call_callee(node, parent) -> bool:
    if parent is None or parent.type != "call_expression":
        return False
    fn = parent.child_by_field_name("function")
    return fn is not None and (fn == node or node.start_byte == fn.start_byte)


def categorize_tree(root) -> dict[str, list[str]]:
    buckets = {k: [] for k in CATEGORIES}

    def add(cat: str, name: str) -> None:
        if name and name not in buckets[cat]:
            buckets[cat].append(name)

    def walk(node, parent, in_param: bool, in_decl: bool) -> None:
        ntype = node.type
        in_param = in_param or ntype in {"parameter_declaration", "parameter_list"}
        in_decl = in_decl or ntype in {"declaration", "init_declarator"}
        name = _text(node) if ntype in IDENT_TYPES else ""
        if ntype in TYPE_NODE_TYPES:
            add("type", _text(node).strip())
        elif ntype in PREPROC_DEF_TYPES:
            add("macro", _preproc_name(node))
        elif ntype == "field_identifier" and name:
            add("field", name)
        elif ntype == "type_identifier" and name:
            add("type", name)
        elif ntype == "identifier" and name:
            if _is_call_callee(node, parent):
                add("callee", name)
            elif in_param:
                add("param", name)
            elif in_decl and not in_param:
                add("local", name)
            elif name.isupper() and "_" in name and len(name) >= 3:
                add("macro", name)
            else:
                add("other", name)
        elif ntype == "preproc_arg":
            ident = _text(node).strip()
            if ident and ident.replace("_", "").isalnum():
                add("macro", ident)
        for child in node.children:
            walk(child, node, in_param, in_decl)

    walk(root, None, False, False)
    return buckets


def parse_function(code: str) -> ParseResult:
    raw = (code or "").encode("utf-8", "ignore")
    if not raw.strip():
        return ParseResult(False, "none", [], set(), {k: [] for k in CATEGORIES}, "empty")
    for name, parser in (("C", _C_PARSER), ("CPP", _CPP_PARSER)):
        tree = parser.parse(raw)
        if tree.root_node.has_error:
            continue
        idents: list[str] = []
        _collect_idents(tree.root_node, idents)
        uniq = list(dict.fromkeys(idents))
        return ParseResult(True, name, uniq, set(uniq), categorize_tree(tree.root_node), "")
    return ParseResult(False, "none", [], set(), {k: [] for k in CATEGORIES}, "has_error")
