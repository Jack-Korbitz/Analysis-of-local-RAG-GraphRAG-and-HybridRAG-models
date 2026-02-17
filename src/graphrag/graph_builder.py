"""
Graph Builder - Extracts entities and relationships from financial documents
and loads them into Neo4j
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graphrag.neo4j_client import Neo4jClient
from src.models.ollama_client import OllamaClient
from datasets import load_from_disk
from tqdm import tqdm
import json
import re


class GraphBuilder:
    """
    Builds a knowledge graph from financial documents

    Graph Schema:
        (Company)-[:HAS_METRIC]->(Metric)-[:FOR_YEAR]->(Year)
        (Company)-[:HAS_DOCUMENT]->(Document)
        (Document)-[:MENTIONS]->(Metric)
        (Document)-[:FROM_YEAR]->(Year)
    """

    def __init__(self, neo4j_client: Neo4jClient, llm_model: str = "gpt-oss:20b"):
        """
        Initialize graph builder

        Args:
            neo4j_client: Connected Neo4j client
            llm_model: Ollama model to use for entity extraction
        """
        self.neo4j = neo4j_client
        self.llm = OllamaClient(llm_model, temperature=0.0)
        print(f"Graph builder initialized with model: {llm_model}")

    def extract_entities_with_llm(self, text: str, metadata: dict) -> dict:
        """
        Use LLM to extract financial entities from text

        Args:
            text: Document text to extract from
            metadata: Document metadata (company, year, etc.)

        Returns:
            Dict with extracted entities
        """
        prompt = f"""Extract financial metrics from this text. Return ONLY a JSON object.

Text: {text[:1000]}

Company: {metadata.get('company', 'Unknown')}
Year: {metadata.get('year', 'Unknown')}

Return this exact JSON format with no other text:
{{
    "metrics": [
        {{"name": "metric_name", "value": 0.0, "unit": "millions"}}
    ]
}}

Only include metrics that have explicit numerical values in the text.
If no metrics found, return {{"metrics": []}}"""

        result = self.llm.generate(prompt, max_tokens=500)

        if not result['success']:
            return {"metrics": []}

        try:
            response_text = result['response'].strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return {"metrics": []}

    def extract_entities_rule_based(self, text: str, metadata: dict) -> dict:
        """
        Rule-based entity extraction for financial metrics

        Args:
            text: Document text
            metadata: Document metadata

        Returns:
            Dict with extracted metrics
        """
        metrics = []
        text_lower = text.lower()

        # Patterns specific to financial report format
        patterns = [
            (
                r'interest\s+expense[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'interest_expense'
            ),
            (
                r'net\s+income[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'net_income'
            ),
            (
                r'revenue[s]?[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'revenue'
            ),
            (
                r'operating\s+expense[s]?[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'operating_expenses'
            ),
            (
                r'cash\s+and\s+cash\s+equivalents[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'cash_and_equivalents'
            ),
            (
                r'net\s+revenue[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'net_revenue'
            ),
            (
                r'total\s+assets[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'total_assets'
            ),
            (
                r'earnings\s+per\s+(?:diluted\s+)?share[^$]*\$\s*([\d,\.]+)',
                'earnings_per_share'
            ),
            (
                r'depreciation\s+expense[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'depreciation'
            ),
            (
                r'capital\s+expenditure[s]?[^$]*\$\s*([\d,\.]+)\s*(million|billion|thousand)?',
                'capital_expenditures'
            ),
        ]

        for pattern, metric_name in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    value_str = match[0].replace(',', '').strip()
                    value = float(value_str)

                    # Skip values that look like years
                    if 1900 <= value <= 2100:
                        continue

                    unit = match[1] if len(match) > 1 and match[1] else 'unknown'

                    # Normalize to millions
                    if 'billion' in unit:
                        value = round(value * 1000, 2)
                    elif 'thousand' in unit:
                        value = round(value / 1000, 2)

                    metrics.append({
                        'name': metric_name,
                        'value': value,
                        'unit': 'millions'
                    })
                except (ValueError, IndexError):
                    continue

        return {"metrics": metrics}

    def build_from_dataset(
        self,
        dataset_path: str,
        max_examples: int = 50,
        use_llm: bool = False
    ):
        """
        Build knowledge graph from dataset

        Args:
            dataset_path: Path to dataset
            max_examples: Maximum number of examples to process
            use_llm: Whether to use LLM for entity extraction
        """
        print(f"\n{'='*60}")
        print("Building Knowledge Graph")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_path}")
        print(f"Max examples: {max_examples}")
        print(f"Using LLM extraction: {use_llm}")

        # Setup indexes
        self.neo4j.create_indexes()

        # Load dataset
        dataset = load_from_disk(dataset_path)
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]

        examples = data.select(range(min(max_examples, len(data))))

        companies_created = set()
        metrics_created = 0
        documents_created = 0

        for i, example in enumerate(tqdm(examples, desc="Building graph")):
            try:
                company = example.get('company_name', 'Unknown')
                year_raw = example.get('report_year', 0)
                context = example.get('context', '')
                doc_id = example.get('id', f'doc_{i}')

                if not context or len(context) < 50:
                    continue

                try:
                    year = int(str(year_raw))
                except (ValueError, TypeError):
                    year = 0

                # Create company node
                if company not in companies_created:
                    self.neo4j.create_company(company, {
                        'sector': example.get('company_sector', 'Unknown'),
                        'industry': example.get('company_industry', 'Unknown'),
                        'headquarters': example.get('company_headquarters', 'Unknown'),
                        'symbol': example.get('company_symbol', 'Unknown')
                    })
                    companies_created.add(company)

                # Create document node
                self.neo4j.create_document(
                    doc_id=doc_id,
                    text=context[:500],
                    properties={
                        'company': company,
                        'year': year,
                        'question': example.get('question', '')[:200]
                    }
                )
                documents_created += 1

                # Link document to company
                if year > 0:
                    self.neo4j.link_document_to_company(doc_id, company, year)

                # Extract metrics
                metadata = {'company': company, 'year': year}

                if use_llm:
                    entities = self.extract_entities_with_llm(context, metadata)
                else:
                    entities = self.extract_entities_rule_based(context, metadata)

                # Create metric nodes
                for metric in entities.get('metrics', []):
                    try:
                        self.neo4j.create_metric(
                            name=metric['name'],
                            value=float(metric['value']),
                            year=year,
                            company=company,
                            properties={
                                'unit': metric.get('unit', 'unknown'),
                                'source_doc': doc_id
                            }
                        )
                        metrics_created += 1
                    except Exception:
                        continue

            except Exception as e:
                print(f"\nError processing example {i}: {e}")
                continue

        print(f"\nGraph building complete!")
        print(f"   Companies created: {len(companies_created)}")
        print(f"   Documents created: {documents_created}")
        print(f"   Metrics created: {metrics_created}")

        stats = self.neo4j.get_graph_stats()
        print(f"\nGraph statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")


def main():
    """Test graph builder"""
    print("="*60)
    print("Testing Graph Builder")
    print("="*60)

    # Initialize Neo4j client
    neo4j = Neo4jClient()

    # Clear existing data
    print("\nClearing existing graph...")
    neo4j.clear_database()

    # Initialize graph builder
    builder = GraphBuilder(neo4j, llm_model="gpt-oss:20b")

    # Build graph from dataset
    builder.build_from_dataset(
        dataset_path="data/benchmarks/t2-ragbench-FinQA",
        max_examples=50,
        use_llm=False
    )

    # Test queries
    print("\n" + "="*60)
    print("Testing Graph Queries")
    print("="*60)

    # Get all companies
    companies = neo4j.query_graph(
        "MATCH (c:Company) RETURN c.name as company LIMIT 10"
    )
    print(f"\nSample companies in graph:")
    for c in companies:
        print(f"   {c['company']}")

    # Get metrics for companies
    metrics = neo4j.query_graph("""
        MATCH (c:Company)-[:HAS_METRIC]->(m:Metric)-[:FOR_YEAR]->(y:Year)
        RETURN c.name as company, m.name as metric, m.value as value, y.value as year
        ORDER BY c.name, y.value DESC
        LIMIT 10
    """)

    print(f"\nSample metrics in graph:")
    for m in metrics:
        print(f"   {m['company']} ({m['year']}): {m['metric']} = {m['value']}")

    neo4j.close()


if __name__ == "__main__":
    main()