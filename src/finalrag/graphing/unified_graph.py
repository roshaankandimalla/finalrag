import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = PROJECT_ROOT / "graphify-out" / "graph.json"
PYTHON_ROOTS = ["scripts", "src", "tests"]
ARTIFACT_ROOTS = ["data/parsed", "data/normalized", "data/chunks"]


def stable_id(*parts: str) -> str:
    value = ":".join(parts).lower()
    return re.sub(r"[^a-z0-9_:.\\/-]+", "_", value).strip("_")


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


@dataclass
class UnifiedNode:
    id: str
    type: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        label = self.meta.get("label") or self.id
        source_file = self.meta.get("source_file")
        source_location = self.meta.get("source_location")
        payload = {
            "id": self.id,
            "type": self.type,
            "meta": self.meta,
            "label": label,
            "file_type": self.type,
            "norm_label": str(label).lower(),
        }
        if source_file:
            payload["source_file"] = source_file
        if source_location:
            payload["source_location"] = source_location
        return payload


@dataclass(frozen=True)
class UnifiedEdge:
    source: str
    target: str
    type: str
    meta_items: tuple[tuple[str, Any], ...] = ()

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self.meta_items)

    def to_json(self) -> dict[str, Any]:
        meta = self.meta
        return {
            "from": self.source,
            "to": self.target,
            "type": self.type,
            "meta": meta,
            "source": self.source,
            "target": self.target,
            "relation": self.type,
            "confidence": meta.get("confidence", "EXTRACTED"),
            "confidence_score": meta.get("confidence_score", 1.0),
            **{key: value for key, value in meta.items() if key not in {"confidence", "confidence_score"}},
        }


class GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, UnifiedNode] = {}
        self.edges: set[UnifiedEdge] = set()

    def add_node(self, node_id: str, node_type: str, **meta: Any) -> str:
        existing = self.nodes.get(node_id)
        if existing:
            existing.meta.update({key: value for key, value in meta.items() if value is not None})
            return node_id
        self.nodes[node_id] = UnifiedNode(
            id=node_id,
            type=node_type,
            meta={key: value for key, value in meta.items() if value is not None},
        )
        return node_id

    def add_edge(self, source: str, target: str, edge_type: str, **meta: Any) -> None:
        if meta.get("source_file") is None:
            source_node = self.nodes.get(source)
            target_node = self.nodes.get(target)
            meta["source_file"] = (
                (source_node.meta.get("source_file") if source_node else None)
                or (target_node.meta.get("source_file") if target_node else None)
                or "graphify-out/graph.json"
            )
        self.edges.add(
            UnifiedEdge(
                source=source,
                target=target,
                type=edge_type,
                meta_items=tuple(sorted((key, value) for key, value in meta.items() if value is not None)),
            )
        )

    def to_json(self) -> dict[str, Any]:
        nodes = [node.to_json() for node in sorted(self.nodes.values(), key=lambda item: item.id)]
        edges = [edge.to_json() for edge in sorted(self.edges, key=lambda item: (item.source, item.type, item.target))]
        return {
            "schema_version": "unified-graph-v1",
            "directed": True,
            "multigraph": False,
            "graph": {
                "model": "unified_static_runtime",
                "node_types": ["function", "artifact"],
                "edge_types": ["calls", "produces", "consumes"],
            },
            "nodes": nodes,
            "edges": edges,
            "links": edges,
        }


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, source_file: str, module_name: str, builder: GraphBuilder) -> None:
        self.source_file = source_file
        self.module_name = module_name
        self.builder = builder
        self.scope: list[str] = []
        self.function_ids: dict[ast.AST, str] = {}
        self.calls: list[tuple[str, str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self.scope, node.name]) if self.scope else node.name
        node_id = stable_id("function", self.module_name, qualname)
        self.function_ids[node] = node_id
        self.builder.add_node(
            node_id,
            "function",
            label=f"{qualname}()",
            name=node.name,
            qualified_name=qualname,
            module=self.module_name,
            source_file=self.source_file,
            source_location=f"L{node.lineno}",
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self.scope:
            caller = stable_id("function", self.module_name, ".".join(self.scope))
            callee_name = call_name(node.func)
            if callee_name:
                self.calls.append((caller, callee_name, node.lineno))
        self.generic_visit(node)


def module_name(path: Path) -> str:
    relative_path = relative(path)
    if relative_path.startswith("src/"):
        return ".".join(Path(relative_path[4:]).with_suffix("").parts)
    return ".".join(Path(relative_path).with_suffix("").parts)


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def python_files() -> list[Path]:
    files = []
    for root in PYTHON_ROOTS:
        base = PROJECT_ROOT / root
        if base.exists():
            files.extend(
                path
                for path in base.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    return sorted(files)


def build_static_graph(builder: GraphBuilder) -> None:
    collectors = []
    function_by_simple_name: dict[str, list[str]] = {}
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collector = FunctionCollector(relative(path), module_name(path), builder)
        collector.visit(tree)
        collectors.append(collector)
        for node_id, node in builder.nodes.items():
            if node.type == "function" and node.meta.get("source_file") == relative(path):
                function_by_simple_name.setdefault(node.meta["name"], []).append(node_id)

    for collector in collectors:
        for caller, callee_name, line_number in collector.calls:
            candidates = function_by_simple_name.get(callee_name, [])
            if len(candidates) == 1:
                builder.add_edge(
                    caller,
                    candidates[0],
                    "calls",
                    source_file=builder.nodes[caller].meta.get("source_file"),
                    source_location=f"L{line_number}",
                    confidence="INFERRED",
                    confidence_score=0.9,
                )


def artifact_node(builder: GraphBuilder, artifact_id: str, label: str, path: str, **meta: Any) -> str:
    absolute = PROJECT_ROOT / path
    file_count = 0
    byte_count = 0
    samples: list[str] = []
    if absolute.exists():
        files = [absolute] if absolute.is_file() else sorted(item for item in absolute.rglob("*") if item.is_file())
        file_count = len(files)
        byte_count = sum(item.stat().st_size for item in files)
        samples = [relative(item) for item in files[:10]]
    return builder.add_node(
        stable_id("artifact", artifact_id),
        "artifact",
        label=label,
        path=path,
        source_file=path,
        exists=absolute.exists(),
        file_count=file_count,
        byte_count=byte_count,
        samples=samples,
        **meta,
    )


def function_ref(builder: GraphBuilder, module: str, qualname: str, label: str) -> str:
    node_id = stable_id("function", module, qualname)
    return builder.add_node(
        node_id,
        "function",
        label=label,
        name=qualname.split(".")[-1],
        qualified_name=qualname,
        module=module,
    )


def build_runtime_dataflow(builder: GraphBuilder) -> None:
    input_artifact = artifact_node(builder, "data_input", "source_documents", "data/input")
    parsed_artifact = artifact_node(builder, "parsed_output_json", "parsed_output.json", "data/parsed")
    normalized_artifact = artifact_node(builder, "normalized_output_json", "normalized_output.json", "data/normalized")
    chunk_artifact = artifact_node(builder, "chunk_output_json", "chunk_output.json", "data/chunks")

    parse_functions = [
        function_ref(builder, "scripts.03_parse_all", "parse_pdfs", "parse_pdfs()"),
        function_ref(builder, "scripts.03_parse_all", "parse_html_sources", "parse_html_sources()"),
        function_ref(builder, "scripts.03_parse_all", "parse_csv_documents", "parse_csv_documents()"),
    ]
    normalize_function = function_ref(
        builder,
        "finalrag.normalization.normalize",
        "normalize_all",
        "normalize_all()",
    )
    chunk_function = function_ref(
        builder,
        "finalrag.chunking.hierarchical_chunker",
        "create_all_chunks",
        "create_all_chunks()",
    )

    for function_id in parse_functions:
        builder.add_edge(
            input_artifact,
            function_id,
            "consumes",
            stage="parse",
            source_file=builder.nodes[function_id].meta.get("source_file"),
        )
        builder.add_edge(
            function_id,
            parsed_artifact,
            "produces",
            stage="parse",
            source_file=builder.nodes[function_id].meta.get("source_file"),
        )

    builder.add_edge(
        parsed_artifact,
        normalize_function,
        "consumes",
        stage="normalize",
        source_file=builder.nodes[normalize_function].meta.get("source_file"),
    )
    builder.add_edge(
        normalize_function,
        normalized_artifact,
        "produces",
        stage="normalize",
        source_file=builder.nodes[normalize_function].meta.get("source_file"),
    )
    builder.add_edge(
        normalized_artifact,
        chunk_function,
        "consumes",
        stage="chunk",
        source_file=builder.nodes[chunk_function].meta.get("source_file"),
    )
    builder.add_edge(
        chunk_function,
        chunk_artifact,
        "produces",
        stage="chunk",
        source_file=builder.nodes[chunk_function].meta.get("source_file"),
    )

    for path in ARTIFACT_ROOTS:
        root = PROJECT_ROOT / path
        if not root.exists():
            continue
        aggregate = {
            "data/parsed": parsed_artifact,
            "data/normalized": normalized_artifact,
            "data/chunks": chunk_artifact,
        }[path]
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            node_id = builder.add_node(
                stable_id("artifact", relative(file_path)),
                "artifact",
                label=file_path.name,
                path=relative(file_path),
                source_file=relative(file_path),
                byte_count=file_path.stat().st_size,
            )
            builder.add_edge(
                aggregate,
                node_id,
                "produces",
                stage="artifact_manifest",
                source_file=relative(file_path),
            )


def build_unified_graph() -> dict[str, Any]:
    builder = GraphBuilder()
    build_static_graph(builder)
    build_runtime_dataflow(builder)
    return builder.to_json()


def write_unified_graph(path: Path = GRAPH_PATH) -> dict[str, Any]:
    graph = build_unified_graph()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    return graph
