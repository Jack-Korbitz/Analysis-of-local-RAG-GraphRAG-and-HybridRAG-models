import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path
from src.graphrag.neo4j_client import Neo4jClient

OUTPUT_PATH = Path("results/visualizations/knowledge_graph.png")

# ─────────────────────────────────────────────
#  ADJUST THESE VALUES BEFORE RUNNING
# ─────────────────────────────────────────────
NUM_COMPANIES  = 20       # 1 – 306 total companies in graph
YEAR_TO_SHOW   = 2018     # 1998 – 2022
NODE_SPACING   = 2.0      # higher = more spread out (try 3.0 – 8.0)
FIGURE_SIZE    = (32, 22) # width x height in inches
# ─────────────────────────────────────────────

NODE_COLORS = {
    "Company":  "#2196F3",
    "Metric":   "#4CAF50",
    "Year":     "#FF9800",
    "Document": "#9E9E9E",
}
NODE_SIZES = {
    "Company":  1000,
    "Metric":   300,
    "Year":     600,
    "Document": 200,
}

METRIC_LABELS = {
    "revenue":            "Revenue",
    "net_income":         "Net Income",
    "interest_expense":   "Interest Exp.",
    "operating_expenses": "Op. Expenses",
    "total_assets":       "Total Assets",
    "gross_profit":       "Gross Profit",
    "operating_income":   "Op. Income",
    "cash_and_equivalents": "Cash",
}


def fetch_subgraph(neo4j: Neo4jClient, companies: list, year: int):
    """One metric-type node per company for the selected year, one shared Year node."""
    nodes = {}
    edges = []

    y_id = f"year_{year}"
    nodes[y_id] = {"type": "Year", "display": str(year)}

    rows = neo4j.query_graph("""
        MATCH (c:Company)-[:HAS_METRIC]->(m:Metric)-[:FOR_YEAR]->(y:Year {value: $year})
        WHERE c.name IN $companies
        RETURN c.name AS company, m.name AS metric
        ORDER BY c.name, m.name
    """, {"companies": companies, "year": year})

    seen = set()
    for row in rows:
        company = row["company"]
        mname   = row["metric"]

        key = (company, mname)
        if key in seen:
            continue
        seen.add(key)

        c_id = f"company_{company}"
        m_id = f"metric_{company}_{mname}"

        if c_id not in nodes:
            nodes[c_id] = {"type": "Company", "display": company}

        label = METRIC_LABELS.get(mname, mname.replace("_", " ").title())
        nodes[m_id] = {"type": "Metric", "display": label}

        edges.append((c_id, m_id, "HAS_METRIC"))
        edges.append((m_id, y_id, "FOR_YEAR"))

    return nodes, edges


def build_nx_graph(nodes, edges):
    G = nx.DiGraph()
    for node_id, attrs in nodes.items():
        G.add_node(node_id, **attrs)
    for src, dst, rel in edges:
        G.add_edge(src, dst, rel=rel)
    return G


def draw(G: nx.DiGraph, output_path: Path):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    company_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "Company"]
    metric_nodes  = [n for n, d in G.nodes(data=True) if d["type"] == "Metric"]
    year_nodes    = [n for n, d in G.nodes(data=True) if d["type"] == "Year"]

    pos = nx.spring_layout(G, k=NODE_SPACING, iterations=120, seed=42)

    has_metric_edges = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "HAS_METRIC"]
    for_year_edges   = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "FOR_YEAR"]

    nx.draw_networkx_edges(G, pos, edgelist=has_metric_edges, ax=ax,
                           edge_color="#2196F3", alpha=0.3, arrows=False, width=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=for_year_edges, ax=ax,
                           edge_color="#FF9800", alpha=0.3, arrows=False, width=0.6)

    for node_type, color in NODE_COLORS.items():
        if node_type == "Document":
            continue
        subset = [n for n, d in G.nodes(data=True) if d["type"] == node_type]
        if not subset:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=subset, ax=ax,
                               node_color=color,
                               node_size=NODE_SIZES[node_type],
                               alpha=0.9)

    # Company and Year nodes get labels; metric nodes are too numerous for full labels
    company_labels = {n: d["display"] for n, d in G.nodes(data=True) if d["type"] == "Company"}
    year_labels    = {n: d["display"] for n, d in G.nodes(data=True) if d["type"] == "Year"}
    metric_labels  = {n: d["display"] for n, d in G.nodes(data=True) if d["type"] == "Metric"}

    nx.draw_networkx_labels(G, pos, labels=company_labels, ax=ax,
                            font_size=5.5, font_color="white", font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=year_labels, ax=ax,
                            font_size=6, font_color="white", font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=metric_labels, ax=ax,
                            font_size=3.5, font_color="white")

    legend_handles = [
        mpatches.Patch(color=NODE_COLORS["Company"],  label=f"Company ({len(company_nodes)})"),
        mpatches.Patch(color=NODE_COLORS["Metric"],   label=f"Financial Metric ({len(metric_nodes)})"),
        mpatches.Patch(color=NODE_COLORS["Year"],     label=f"Year ({len(year_nodes)})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=11,
              facecolor="#2a2a4a", edgecolor="white", labelcolor="white")

    ax.set_title("GraphRAG Knowledge Graph — Financial Entity Structure",
                 fontsize=18, fontweight="bold", color="white", pad=20)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.show()


def main():
    neo4j = Neo4jClient()

    top = neo4j.query_graph("""
        MATCH (c:Company)-[:HAS_METRIC]->(m:Metric)
        RETURN c.name AS company, count(m) AS cnt
        ORDER BY cnt DESC LIMIT $n
    """, {"n": NUM_COMPANIES})
    companies = [r["company"] for r in top]
    print(f"Visualizing {len(companies)} companies | year={YEAR_TO_SHOW} | spacing={NODE_SPACING}")

    nodes, edges = fetch_subgraph(neo4j, companies, year=YEAR_TO_SHOW)
    print(f"  Nodes: {len(nodes)}  Edges: {len(edges)}")

    G = build_nx_graph(nodes, edges)
    draw(G, OUTPUT_PATH)
    neo4j.close()


if __name__ == "__main__":
    main()
