import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from finalrag.graphing.unified_graph import GRAPH_PATH, write_unified_graph


def main() -> None:
    graph = write_unified_graph(GRAPH_PATH)
    node_counts = Counter(node["type"] for node in graph["nodes"])
    edge_counts = Counter(edge["type"] for edge in graph["edges"])
    print(f"Unified graph written: {GRAPH_PATH}")
    print(f"Nodes: {len(graph['nodes']):,} {dict(node_counts)}")
    print(f"Edges: {len(graph['edges']):,} {dict(edge_counts)}")


if __name__ == "__main__":
    main()
