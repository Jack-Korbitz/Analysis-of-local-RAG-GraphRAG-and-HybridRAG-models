"""
Export a slide-ready knowledge graph visualization from Neo4j.
Queries a small subgraph (3 companies, their metrics and years) and
renders it as a PNG using networkx + matplotlib.

Run:
    python scripts/visualize_graph.py
Output:
    results/visualizations/knowledge_graph.png
"""
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path
from src.graphrag.neo4j_client import Neo4jClient

OUTPUT_PATH = Path("results/visualizations/knowledge_graph.png")

NODE_COLORS = {
    "Company":  "#2196F3",
    "Metric":   "#4CAF50",
    "Year":     "#FF9800",
    "Document": "#9E9E9E",
}
NODE_SIZES = {
    "Company":  2800,
    "Metric":   1400,
    "Year":     1800,
    "Document": 800,
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


def fetch_subgraph(neo4j: Neo4jClient, companies: list, max_metrics_per_company: int = 4):
    """Pull a small, readable subgraph from Neo4j."""
    nodes = {}   # id -> {label, type, display}
    edges = []   # (src_id, dst_id, rel_type)

    for company in companies:
        c_id = f"company_{company}"
        nodes[c_id] = {"type": "Company", "display": company}

        rows = neo4j.query_graph("""
            MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)-[:FOR_YEAR]->(y:Year)
            RETURN m.name AS metric, m.value AS value, y.value AS year
            ORDER BY y.value DESC, m.name
            LIMIT $limit
        """, {"company": company, "limit": max_metrics_per_company * 3})

        seen_metrics = {}
        for row in rows:
            mname = row["metric"]
            year  = int(row["year"])
            value = row["value"]

            if mname not in seen_metrics:
                if len(seen_metrics) >= max_metrics_per_company:
                    continue
                seen_metrics[mname] = True

            m_id = f"metric_{company}_{mname}_{year}"
            y_id = f"year_{year}"
            label = METRIC_LABELS.get(mname, mname.replace("_", " ").title())
            display = f"{label}\n${value:,.0f}M"

            nodes[m_id] = {"type": "Metric", "display": display}
            nodes[y_id] = {"type": "Year",   "display": str(year)}

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
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Separate nodes by type for layout
    company_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "Company"]
    metric_nodes  = [n for n, d in G.nodes(data=True) if d["type"] == "Metric"]
    year_nodes    = [n for n, d in G.nodes(data=True) if d["type"] == "Year"]

    # Manual layered layout: companies left, metrics center, years right
    pos = {}
    n_companies = len(company_nodes)
    for i, n in enumerate(company_nodes):
        pos[n] = (-2.5, (i - (n_companies - 1) / 2) * 2.5)

    n_metrics = len(metric_nodes)
    for i, n in enumerate(metric_nodes):
        pos[n] = (0, (i - (n_metrics - 1) / 2) * 0.95)

    unique_years = sorted({G.nodes[n]["display"] for n in year_nodes})
    year_y = {y: (i - (len(unique_years) - 1) / 2) * 2.2 for i, y in enumerate(unique_years)}
    for n in year_nodes:
        yr = G.nodes[n]["display"]
        pos[n] = (2.8, year_y[yr])

    # Draw edges
    has_metric_edges = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "HAS_METRIC"]
    for_year_edges   = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == "FOR_YEAR"]

    nx.draw_networkx_edges(G, pos, edgelist=has_metric_edges, ax=ax,
                           edge_color="#2196F3", alpha=0.5, arrows=True,
                           arrowsize=15, width=1.2,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(G, pos, edgelist=for_year_edges, ax=ax,
                           edge_color="#FF9800", alpha=0.5, arrows=True,
                           arrowsize=15, width=1.2,
                           connectionstyle="arc3,rad=0.05")

    # Draw nodes by type
    for node_type, color in NODE_COLORS.items():
        if node_type == "Document":
            continue
        subset = [n for n, d in G.nodes(data=True) if d["type"] == node_type]
        if not subset:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=subset, ax=ax,
                               node_color=color,
                               node_size=NODE_SIZES[node_type],
                               alpha=0.95)

    # Labels
    labels = {n: d["display"] for n, d in G.nodes(data=True) if d["type"] != "Document"}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7.5, font_color="white", font_weight="bold")

    # Edge labels
    edge_label_pos = {
        **{e: "HAS_METRIC" for e in has_metric_edges},
        **{e: "FOR_YEAR"   for e in for_year_edges},
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label_pos, ax=ax,
                                 font_size=6, font_color="#cccccc",
                                 bbox=dict(alpha=0))

    # Legend
    legend_handles = [
        mpatches.Patch(color=NODE_COLORS["Company"],  label="Company"),
        mpatches.Patch(color=NODE_COLORS["Metric"],   label="Financial Metric"),
        mpatches.Patch(color=NODE_COLORS["Year"],     label="Year"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=10,
              facecolor="#2a2a4a", edgecolor="white", labelcolor="white")

    ax.set_title("GraphRAG Knowledge Graph — Financial Entity Structure",
                 fontsize=15, fontweight="bold", color="white", pad=16)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.show()


def main():
    neo4j = Neo4jClient()

    # Pick 3 companies that have the most metrics for a rich visualization
    top = neo4j.query_graph("""
        MATCH (c:Company)-[:HAS_METRIC]->(m:Metric)
        RETURN c.name AS company, count(m) AS cnt
        ORDER BY cnt DESC LIMIT 3
    """)
    companies = [r["company"] for r in top]
    print(f"Visualizing companies: {companies}")

    nodes, edges = fetch_subgraph(neo4j, companies, max_metrics_per_company=4)
    print(f"  Nodes: {len(nodes)}  Edges: {len(edges)}")

    G = build_nx_graph(nodes, edges)
    draw(G, OUTPUT_PATH)
    neo4j.close()


if __name__ == "__main__":
    main()
