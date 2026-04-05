import sys
import textwrap
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
NUM_YEARS    = 10     # number of most recent years to show
NODE_SPACING = 3.0
FIGURE_SIZE  = (26, 18)
# ─────────────────────────────────────────────

NODE_COLORS = {
    "Company": "#003087",   # LM navy blue
    "Metric":  "#009CDE",   # LM bright blue
    "Year":    "#6D6E71",   # LM silver grey
}
NODE_SIZES = {
    "Company": 8000,
    "Metric":  4000,
    "Year":    4000,
}

METRIC_LABELS = {
    "revenue":              "Revenue",
    "net_income":           "Net Income",
    "interest_expense":     "Interest Exp.",
    "operating_expenses":   "Op. Expenses",
    "total_assets":         "Total Assets",
    "gross_profit":         "Gross Profit",
    "operating_income":     "Op. Income",
    "cash_and_equivalents": "Cash",
}


def find_company_with_most_years(neo4j: Neo4jClient, n: int):
    """Find the company with the most distinct years, return it and its top-n years."""
    rows = neo4j.query_graph("""
        MATCH (c:Company)-[:HAS_METRIC]->(:Metric)-[:FOR_YEAR]->(y:Year)
        RETURN c.name AS company, count(DISTINCT y.value) AS yr_count
        ORDER BY yr_count DESC LIMIT 1
    """, {})
    company = rows[0]["company"]

    year_rows = neo4j.query_graph("""
        MATCH (c:Company {name: $company})-[:HAS_METRIC]->(:Metric)-[:FOR_YEAR]->(y:Year)
        RETURN DISTINCT y.value AS year
        ORDER BY year DESC LIMIT $n
    """, {"company": company, "n": n})
    years = sorted(r["year"] for r in year_rows)
    return company, years


def fetch_subgraph(neo4j: Neo4jClient, company: str, years: list):
    nodes = {}
    edges = []

    c_id = f"company_{company}"
    nodes[c_id] = {"type": "Company", "display": company}

    for year in years:
        y_id = f"year_{year}"
        nodes[y_id] = {"type": "Year", "display": str(year)}

        metric_rows = neo4j.query_graph("""
            MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)-[:FOR_YEAR]->(y:Year {value: $year})
            RETURN DISTINCT m.name AS metric
            ORDER BY m.name
        """, {"company": company, "year": year})

        for row in metric_rows:
            mname = row["metric"]
            m_id  = f"metric_{mname}_{year}"
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


def draw(G: nx.DiGraph, company: str, years: list, output_path: Path):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    metric_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "Metric"]
    year_nodes   = [n for n, d in G.nodes(data=True) if d["type"] == "Year"]

    pos = nx.spring_layout(G, k=NODE_SPACING, iterations=150, seed=42)

    edge_styles = {
        "HAS_METRIC": NODE_COLORS["Metric"],
        "FOR_YEAR":   NODE_COLORS["Year"],
    }
    for rel, color in edge_styles.items():
        edgelist = [(u, v) for u, v, d in G.edges(data=True) if d["rel"] == rel]
        if edgelist:
            nx.draw_networkx_edges(G, pos, edgelist=edgelist, ax=ax,
                                   edge_color=color, alpha=0.8, arrows=True,
                                   arrowsize=14, width=1.5)

    for node_type, color in NODE_COLORS.items():
        subset = [n for n, d in G.nodes(data=True) if d["type"] == node_type]
        if not subset:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=subset, ax=ax,
                               node_color=color,
                               node_size=NODE_SIZES[node_type],
                               alpha=1.0)

    font_cfg = {
        "Company": dict(size=14, weight="bold", wrap=8),
        "Year":    dict(size=13, weight="bold", wrap=6),
        "Metric":  dict(size=10, weight="bold", wrap=10),
    }
    for n, d in G.nodes(data=True):
        ntype = d["type"]
        if ntype not in font_cfg:
            continue
        cfg = font_cfg[ntype]
        x, y = pos[n]
        wrapped = textwrap.fill(d["display"], width=cfg["wrap"])
        ax.text(x, y, wrapped, ha="center", va="center",
                fontsize=cfg["size"], fontweight=cfg["weight"],
                color="white", multialignment="center",
                zorder=5)

    legend_handles = [
        mpatches.Patch(color=NODE_COLORS["Company"], label="Company (1)"),
        mpatches.Patch(color=NODE_COLORS["Metric"],  label=f"Financial Metric ({len(metric_nodes)})"),
        mpatches.Patch(color=NODE_COLORS["Year"],    label=f"Year ({len(year_nodes)})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=20,
              facecolor="#ffffff", edgecolor="#333333", labelcolor="#333333",
              handleheight=2, handlelength=3)

    year_range = f"{years[0]}–{years[-1]}"
    ax.set_title(f"GraphRAG Knowledge Graph",
                 fontsize=40, fontweight="bold", color="#333333", pad=20)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.show()


def main():
    neo4j = Neo4jClient()

    company, years = find_company_with_most_years(neo4j, NUM_YEARS)
    print(f"Company: {company} | Years: {years}")

    nodes, edges = fetch_subgraph(neo4j, company, years)
    print(f"  Nodes: {len(nodes)}  Edges: {len(edges)}")

    G = build_nx_graph(nodes, edges)
    draw(G, company, years, OUTPUT_PATH)
    neo4j.close()


if __name__ == "__main__":
    main()
