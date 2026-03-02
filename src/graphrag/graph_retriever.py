import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graphrag.neo4j_client import Neo4jClient
from typing import List, Dict, Optional
import re


class GraphRetriever:
    """
    Retrieves context from Neo4j knowledge graph for GraphRAG
    
    Retrieval strategies:
        1. Entity-based: Find nodes matching entities in query
        2. Relationship traversal: Follow relationships to find related info
        3. Multi-hop: Traverse multiple relationships for complex queries
    """

    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize graph retriever

        Args:
            neo4j_client: Connected Neo4j client
        """
        self.neo4j = neo4j_client
        print("Graph retriever initialized")

    def extract_query_entities(self, query: str) -> Dict:
        """
        Extract key entities from a query using rules

        Args:
            query: User query string

        Returns:
            Dict with extracted entities (company, year, metric_type)
        """
        entities = {
            'company': None,
            'year': None,
            'metric_type': None
        }

        # Extract year (4 digit number between 1990-2030)
        year_match = re.search(r'\b(19[9]\d|20[0-3]\d)\b', query)
        if year_match:
            entities['year'] = int(year_match.group())

        # Extract metric type from query keywords
        query_lower = query.lower()

        metric_keywords = {
            'interest_expense': ['interest expense', 'interest cost', 'interest paid'],
            'net_income': ['net income', 'net earnings', 'net profit', 'net loss'],
            'net_revenue': ['net revenue'],
            'revenue': ['revenue', 'revenues', 'net sales', 'total sales', 'total revenue'],
            'operating_expenses': ['operating expense', 'operating cost', 'operating expenditure', 'total operating'],
            'cash_and_equivalents': ['cash and cash equivalents', 'cash equivalent', 'cash and investments'],
            'total_assets': ['total assets'],
            'earnings_per_share': ['earnings per share', 'eps', 'diluted eps', 'basic eps'],
            'depreciation': ['depreciation', 'amortization', 'depreciation and amortization', 'd&a'],
            'capital_expenditures': ['capital expenditure', 'capex', 'capital spending', 'purchases of property'],
            'gross_profit': ['gross profit', 'gross margin', 'gross income'],
            'operating_income': ['operating income', 'operating profit', 'income from operations'],
            'stock_compensation': ['stock-based compensation', 'share-based compensation', 'stock compensation'],
            'fuel_costs': ['fuel', 'fuel expense', 'fuel cost', 'aircraft fuel'],
            'long_term_debt': ['long-term debt', 'long term debt', 'total debt'],
            'available_for_sale': ['available-for-sale', 'available for sale'],
            'tax_expense': ['income tax', 'tax expense', 'provision for income tax'],
            'research_development': ['research and development', 'r&d', 'research & development'],
            'cash_flow_operations': ['cash flow from operations', 'operating cash flow', 'net cash provided by operating'],
            'dividends': ['dividends', 'dividend per share', 'dividends paid'],
        }

        for metric_name, keywords in metric_keywords.items():
            if any(kw in query_lower for kw in keywords):
                entities['metric_type'] = metric_name
                break

        return entities

    _COMPANY_SUFFIXES = [
        ', inc.', ', inc', ', corp.', ', corp', ', llc', ', ltd.', ', ltd',
        ' incorporated', ' corporation', ' company', ' group', ' holdings',
        ' inc.', ' inc', ' corp.', ' corp', ' llc', ' ltd.', ' ltd'
    ]

    def _strip_company_suffix(self, name: str) -> str:
        name_lower = name.lower().strip()
        for suffix in self._COMPANY_SUFFIXES:
            if name_lower.endswith(suffix):
                name_lower = name_lower[:-len(suffix)].strip()
                break
        return name_lower

    def find_company_in_graph(self, query: str) -> Optional[str]:
        companies = self.neo4j.query_graph(
            "MATCH (c:Company) RETURN c.name as name"
        )

        query_lower = query.lower()
        best_match = None
        best_match_len = 0

        for record in companies:
            company_name = record['name']
            company_lower = company_name.lower()
            company_stripped = self._strip_company_suffix(company_name)

            # Full name match - prefer longest match to avoid "Entergy" beating "Entergy Louisiana"
            if company_lower in query_lower:
                if len(company_lower) > best_match_len:
                    best_match = company_name
                    best_match_len = len(company_lower)

            # Stripped name match (e.g. "Analog Devices" matches "Analog Devices, Inc.")
            elif company_stripped and company_stripped in query_lower:
                if len(company_stripped) > best_match_len:
                    best_match = company_name
                    best_match_len = len(company_stripped)

        return best_match

    def retrieve_by_entity(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve context by matching entities in query to graph nodes

        Args:
            query: User query
            top_k: Maximum results to return

        Returns:
            List of relevant context dicts
        """
        entities = self.extract_query_entities(query)
        company = self.find_company_in_graph(query)

        # Strategy 1: Exact match — Company + Year + Metric.
        # Return immediately if found: the answer is in the structured record,
        # adding documents would only add noise.
        if company and entities['year'] and entities['metric_type']:
            cypher = """
            MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)
            -[:FOR_YEAR]->(y:Year {value: $year})
            WHERE m.name = $metric_type
            RETURN DISTINCT c.name as company, m.name as metric,
                   m.value as value, y.value as year,
                   'company_year_metric' as strategy
            LIMIT $top_k
            """
            records = self.neo4j.query_graph(cypher, {
                'company': company,
                'year': entities['year'],
                'metric_type': entities['metric_type'],
                'top_k': top_k
            })
            if records:
                return records

        # Strategy 2: Company + Year → go straight to document text.
        # S1 failed, meaning the needed metric is not one of the 23 canonical types.
        # Returning all S2 metrics (net_income, revenue, etc.) would add irrelevant
        # numbers that anchor the LLM to the wrong values. Documents contain the
        # full tables and are far more likely to hold the answer.
        if company and entities['year']:
            cypher = """
            MATCH (d:Document)-[:ABOUT]->(c:Company {name: $company})
            WHERE d.year = $year
            RETURN c.name as company, d.text as text,
                   d.year as year, 'document_search' as strategy
            LIMIT 3
            """
            doc_results = self.neo4j.query_graph(cypher, {
                'company': company,
                'year': entities['year'],
            })
            if doc_results:
                return doc_results

        # Strategy 3: Company + most recent docs (no year in query or no year-filtered docs found)
        if company and not entities['year']:
            cypher = """
            MATCH (d:Document)-[:ABOUT]->(c:Company {name: $company})
            RETURN c.name as company, d.text as text,
                   d.year as year, 'document_search' as strategy
            ORDER BY d.year DESC
            LIMIT 2
            """
            doc_results = self.neo4j.query_graph(cypher, {'company': company})
            if doc_results:
                return doc_results

            # Also try structured metrics when no year specified
            cypher = """
            MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)
            -[:FOR_YEAR]->(y:Year)
            RETURN DISTINCT c.name as company, m.name as metric,
                   m.value as value, y.value as year,
                   'company_only' as strategy
            ORDER BY y.value DESC
            LIMIT $top_k
            """
            records = self.neo4j.query_graph(cypher, {
                'company': company,
                'top_k': top_k
            })
            if records:
                return records

        # Strategy 4: Metric type only — no company identified
        if entities['metric_type'] and not entities['year']:
            cypher = """
            MATCH (c:Company)-[:HAS_METRIC]->(m:Metric {name: $metric_type})
            -[:FOR_YEAR]->(y:Year)
            RETURN DISTINCT c.name as company, m.name as metric,
                   m.value as value, y.value as year,
                   'metric_only' as strategy
            ORDER BY y.value DESC
            LIMIT $top_k
            """
            records = self.neo4j.query_graph(cypher, {
                'metric_type': entities['metric_type'],
                'top_k': top_k
            })
            if records:
                return records

        # Strategy 5: Last resort full-text search
        search_term = (company or entities['metric_type'] or query[:50])
        cypher = """
        MATCH (d:Document)-[:ABOUT]->(c:Company)
        WHERE toLower(d.text) CONTAINS toLower($search_term)
        RETURN c.name as company, d.text as text,
               d.year as year, 'document_search' as strategy
        LIMIT $top_k
        """
        return self.neo4j.query_graph(cypher, {
            'search_term': search_term,
            'top_k': top_k
        })

    def retrieve_documents_for_company(
        self,
        company: str,
        year: Optional[int] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve documents related to a company

        Args:
            company: Company name
            year: Optional year filter
            top_k: Maximum results

        Returns:
            List of document dicts
        """
        if year:
            cypher = """
            MATCH (d:Document)-[:ABOUT]->(c:Company {name: $company})
            MATCH (d)-[:FROM_YEAR]->(y:Year {value: $year})
            RETURN d.text as text, c.name as company,
                   y.value as year
            LIMIT $top_k
            """
            return self.neo4j.query_graph(cypher, {
                'company': company,
                'year': year,
                'top_k': top_k
            })
        else:
            cypher = """
            MATCH (d:Document)-[:ABOUT]->(c:Company {name: $company})
            RETURN d.text as text, c.name as company
            LIMIT $top_k
            """
            return self.neo4j.query_graph(cypher, {
                'company': company,
                'top_k': top_k
            })

    def format_context(self, results: List[Dict], query: str) -> str:
        """
        Format graph retrieval results as context string for LLM

        Args:
            results: List of retrieved records
            query: Original query

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found in the knowledge graph."

        context_parts = []
        context_parts.append("Information retrieved from knowledge graph:")
        context_parts.append("")

        for i, record in enumerate(results):
            strategy = record.get('strategy', 'unknown')

            if 'value' in record and 'metric' in record:
                # Metric result
                context_parts.append(
                    f"[Record {i+1}] "
                    f"Company: {record.get('company', 'Unknown')} | "
                    f"Year: {record.get('year', 'Unknown')} | "
                    f"Metric: {record.get('metric', 'Unknown')} | "
                    f"Value: ${record.get('value', 0):.2f} million"
                )
            elif 'text' in record:
                # Document result
                context_parts.append(
                    f"[Record {i+1}] "
                    f"Company: {record.get('company', 'Unknown')} | "
                    f"Year: {record.get('year', 'Unknown')}"
                )
                context_parts.append(
                    f"   Text: {record.get('text', '')[:4000]}"
                )

        return "\n".join(context_parts)

    def retrieve_context_string(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        Main retrieval method - returns formatted context string

        Args:
            query: User query
            top_k: Maximum results

        Returns:
            Formatted context string
        """
        results = self.retrieve_by_entity(query, top_k=top_k)
        return self.format_context(results, query)


def main():
    """Test graph retriever"""
    print("="*60)
    print("Testing Graph Retriever")
    print("="*60)

    # Initialize Neo4j client
    neo4j = Neo4jClient()

    # Initialize retriever
    retriever = GraphRetriever(neo4j)

    # Test queries from our benchmark
    test_queries = [
        "What was Analog Devices reported interest expense for fiscal year 2009?",
        "What were the total operating expenses for American Airlines Group in 2018?",
        "What proportion of Intel's total cash and investments as of December 29, 2012?",
        "What was the percentage change in Entergy Louisiana net revenue from 2007 to 2008?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Show extracted entities
        entities = retriever.extract_query_entities(query)
        company = retriever.find_company_in_graph(query)

        print(f"Extracted entities:")
        print(f"   Company: {company}")
        print(f"   Year: {entities['year']}")
        print(f"   Metric type: {entities['metric_type']}")

        # Retrieve context
        context = retriever.retrieve_context_string(query, top_k=5)
        print(f"\nRetrieved context:")
        print(context)

    neo4j.close()


if __name__ == "__main__":
    main()