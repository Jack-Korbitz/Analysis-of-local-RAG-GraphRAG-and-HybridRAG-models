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
            'years': [],       # all years found — enables multi-hop retrieval
            'metric_type': None
        }

        # Extract ALL years mentioned (e.g. "change from 2007 to 2008" → [2007, 2008])
        all_years = [int(y) for y in re.findall(r'\b(19[9]\d|20[0-3]\d)\b', query)]
        if all_years:
            entities['year'] = all_years[0]
            entities['years'] = all_years

        # Extract metric type from query keywords
        query_lower = query.lower()

        metric_keywords = {
            # Revenue
            'net_revenue': ['net revenue', 'total net revenue'],
            'revenue': ['total revenue', 'revenues', 'net sales', 'total sales', 'gross revenue'],
            # Net income / EPS
            'net_income': ['net income', 'net earnings', 'net profit', 'net loss'],
            'earnings_per_share': ['earnings per share', 'eps', 'diluted eps', 'basic eps'],
            # Income statement
            'gross_profit': ['gross profit', 'gross margin', 'gross income'],
            'operating_income': ['operating income', 'operating profit', 'income from operations', 'ebit'],
            'ebitda': ['ebitda', 'earnings before interest tax'],
            'income_before_tax': ['income before tax', 'income before income tax', 'pretax income', 'pre-tax income', 'earnings before tax'],
            # Expenses
            'operating_expenses': ['operating expense', 'operating cost', 'operating expenditure', 'total operating'],
            'cost_of_revenue': ['cost of revenue', 'cost of goods sold', 'cogs', 'cost of sales'],
            'research_development': ['research and development', 'r&d', 'research & development'],
            'selling_general_admin': ['selling general and administrative', 'sg&a', 'selling and marketing'],
            'interest_expense': ['interest expense', 'interest cost', 'interest paid'],
            'depreciation': ['depreciation', 'amortization', 'depreciation and amortization', 'd&a'],
            'capital_expenditures': ['capital expenditure', 'capex', 'capital spending', 'purchases of property'],
            'fuel_expense': ['fuel expense', 'fuel cost', 'aircraft fuel'],
            # Tax
            'tax_expense': ['income tax', 'tax expense', 'provision for income tax'],
            'deferred_tax_asset': ['deferred tax asset', 'deferred income tax asset'],
            'deferred_tax_liability': ['deferred tax liability', 'deferred income tax liability'],
            'deferred_tax': ['deferred tax', 'deferred income tax'],
            'effective_tax_rate': ['effective tax rate', 'tax rate'],
            # Stock
            'stock_compensation': ['stock-based compensation', 'share-based compensation', 'stock compensation'],
            'stock_awards': ['stock award', 'restricted stock', 'stock unit', 'rsu', 'restricted share'],
            # Employee benefits — pension vs OPEB are different
            'pension': ['pension', 'net periodic pension', 'defined benefit plan'],
            'post_retirement_benefits': ['postretirement', 'post-retirement benefit', 'benefit obligation', 'opeb'],
            # Balance sheet — assets
            'total_assets': ['total assets'],
            'current_assets': ['current assets', 'total current assets'],
            'cash_and_equivalents': ['cash and cash equivalents', 'cash equivalent', 'cash and investments'],
            'accounts_receivable': ['accounts receivable', 'trade receivable'],
            'inventory': ['inventory', 'inventories'],
            'goodwill': ['goodwill'],
            'intangible_assets': ['intangible asset', 'intangible assets', 'impairment'],
            # Balance sheet — equity
            'shareholders_equity': ['shareholders equity', 'stockholders equity', "shareholders' equity", 'book value'],
            'retained_earnings': ['retained earnings', 'accumulated deficit'],
            # Balance sheet — liabilities
            'total_liabilities': ['total liabilities'],
            'current_liabilities': ['current liabilities', 'total current liabilities'],
            'long_term_debt': ['long-term debt', 'long term debt', 'total long-term debt'],
            'total_debt': ['total debt', 'total indebtedness', 'recourse debt'],
            'short_term_debt': ['short-term debt', 'short term debt', 'current portion of long-term debt', 'notes payable'],
            'accounts_payable': ['accounts payable', 'trade payable'],
            # Leases — split by type
            'operating_lease': ['operating lease', 'operating right-of-use', 'operating rou'],
            'finance_lease': ['finance lease', 'financing lease', 'capital lease'],
            'lease_obligation': ['lease obligation', 'right-of-use', 'rental expense', 'rental payment', 'lease commitment'],
            # Cash flow
            'cash_flow_operations': ['cash flow from operations', 'operating cash flow', 'net cash provided by operating', 'net cash used in operating'],
            'free_cash_flow': ['free cash flow', 'fcf'],
            'dividends_paid': ['dividends paid', 'dividends', 'dividend per share'],
            # Other
            'net_assets': ['net assets', 'working capital'],
            'notional_amount': ['notional amount', 'notional value', 'cash flow hedge', 'fair value hedge', 'hedging'],
            'acquisition_cost': ['purchase price', 'acquisition price', 'acquisition cost', 'total consideration'],
            'backlog': ['backlog', 'order backlog', 'remaining performance obligation'],
            'employee_count': ['employee', 'headcount', 'workforce', 'personnel'],
            'facilities': ['square footage', 'square feet', 'leased facilities', 'owned facilities'],
            'minority_interest': ['minority interest', 'noncontrolling interest', 'non-controlling interest'],
            'deferred_revenue': ['deferred revenue', 'deferred income'],
            'available_for_sale': ['available-for-sale', 'available for sale'],
        }

        # Longest keyword match wins — prevents "revenue" beating "net revenue"
        best_match_len = 0
        for metric_name, keywords in metric_keywords.items():
            for kw in keywords:
                if kw in query_lower and len(kw) > best_match_len:
                    entities['metric_type'] = metric_name
                    best_match_len = len(kw)

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

        def _word_boundary_match(name: str, text: str) -> bool:
            """For short names (<6 chars), require word boundaries to avoid
            'AT' matching inside 'WHAT' or 'HP' inside 'SHAREHOLDER'."""
            if len(name) < 6:
                return bool(re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text))
            return name in text

        for record in companies:
            company_name = record['name']
            if not company_name or company_name == 'Unknown':
                continue
            company_lower = company_name.lower()
            company_stripped = self._strip_company_suffix(company_name)

            # Full name match - prefer longest match to avoid "Entergy" beating "Entergy Louisiana"
            if _word_boundary_match(company_lower, query_lower):
                if len(company_lower) > best_match_len:
                    best_match = company_name
                    best_match_len = len(company_lower)

            # Stripped name match (e.g. "Analog Devices" matches "Analog Devices, Inc.")
            elif company_stripped and _word_boundary_match(company_stripped, query_lower):
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

        # Strategy 1: Exact match — Company + Year(s) + Metric.
        # Queries ALL years found (e.g. 2007 and 2008 for change questions).
        # Return immediately if found: the answer is in the structured record.
        if company and entities['years'] and entities['metric_type']:
            cypher = """
            MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)
            -[:FOR_YEAR]->(y:Year)
            WHERE m.name = $metric_type AND y.value IN $years
            RETURN DISTINCT c.name as company, m.name as metric,
                   m.value as value, y.value as year,
                   'company_year_metric' as strategy
            ORDER BY y.value
            LIMIT $top_k
            """
            records = self.neo4j.query_graph(cypher, {
                'company': company,
                'years': entities['years'],
                'metric_type': entities['metric_type'],
                'top_k': top_k
            })
            if records:
                return records

        # Strategy 2: Company + Year(s) → document text.
        # First try question-relevant documents (match on stored question keywords),
        # then fall back to any document for that company/year.
        if company and entities['years']:
            year_range = sorted({y + d for y in entities['years'] for d in (-1, 0, 1)})

            # 2a: Company + Year + question keyword relevance
            # Extract distinctive terms from the query for document matching
            _stopwords = {'what', 'were', 'was', 'the', 'from', 'that', 'this', 'with',
                          'have', 'for', 'and', 'did', 'does', 'how', 'much', 'many',
                          'percent', 'percentage', 'change', 'total', 'which', 'year',
                          'given', 'based', 'their', 'company', 'reported', 'financial',
                          'statements', 'compared', 'between', 'about', 'would'}
            q_terms = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
                       if w not in _stopwords and w != company.lower().split()[0].lower()][:6]

            if q_terms:
                # Try matching 3+ question terms first (precise), then 2 (relaxed)
                for min_terms in (3, 2):
                    if len(q_terms) < min_terms:
                        continue
                    terms_to_use = q_terms[:max(min_terms + 1, 4)]
                    conditions = ' OR '.join(
                        [f"toLower(d.question) CONTAINS '{t}'" for t in terms_to_use]
                    )
                    # Count how many terms match; require at least min_terms
                    count_expr = ' + '.join(
                        [f"CASE WHEN toLower(d.question) CONTAINS '{t}' THEN 1 ELSE 0 END"
                         for t in terms_to_use]
                    )
                    cypher = f"""
                    MATCH (d:Document)-[:ABOUT]->(c:Company {{name: $company}})
                    WHERE d.year IN $year_range AND ({conditions})
                    WITH c, d, ({count_expr}) AS relevance
                    WHERE relevance >= {min_terms}
                    RETURN c.name as company, d.text as text,
                           d.year as year, 'document_search' as strategy
                    ORDER BY relevance DESC, d.year
                    LIMIT 10
                    """
                    doc_results = self.neo4j.query_graph(cypher, {
                        'company': company,
                        'year_range': year_range,
                    })
                    if doc_results:
                        return doc_results

            # 2b: Plain company + year fallback (no question relevance)
            # When multiple years are queried (e.g. "change from 2014 to 2015"),
            # fetch per-year to guarantee both years are represented.
            if len(entities['years']) > 1:
                all_docs = []
                per_year_limit = max(3, 10 // len(entities['years']))
                for yr in entities['years']:
                    yr_range = sorted({yr + d for d in (-1, 0, 1)})
                    cypher = """
                    MATCH (d:Document)-[:ABOUT]->(c:Company {name: $company})
                    WHERE d.year IN $year_range
                    RETURN c.name as company, d.text as text,
                           d.year as year, 'document_search' as strategy
                    ORDER BY abs(d.year - $target_year), d.year
                    LIMIT $per_year_limit
                    """
                    docs = self.neo4j.query_graph(cypher, {
                        'company': company,
                        'year_range': yr_range,
                        'target_year': yr,
                        'per_year_limit': per_year_limit,
                    })
                    all_docs.extend(docs)
                if all_docs:
                    return all_docs[:10]
            else:
                cypher = """
                MATCH (d:Document)-[:ABOUT]->(c:Company {name: $company})
                WHERE d.year IN $year_range
                RETURN c.name as company, d.text as text,
                       d.year as year, 'document_search' as strategy
                ORDER BY d.year
                LIMIT 10
                """
                doc_results = self.neo4j.query_graph(cypher, {
                    'company': company,
                    'year_range': year_range,
                })
                if doc_results:
                    return doc_results

        # Strategy 2b: Company found + year(s) + metric, but Strategy 1 missed
        # (metric stored under a slightly different name). Try related metric names
        # for the SAME company — never broadcast to all companies, which returns
        # irrelevant data (e.g. Lockheed Martin for a Hartford question).
        if company and entities['years'] and entities['metric_type']:
            cypher = """
            MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)
            -[:FOR_YEAR]->(y:Year)
            WHERE y.value IN $years
              AND (m.name CONTAINS $metric_root OR $metric_root CONTAINS m.name)
            RETURN DISTINCT c.name as company, m.name as metric,
                   m.value as value, y.value as year,
                   'metric_fuzzy' as strategy
            ORDER BY y.value
            LIMIT $top_k
            """
            # Use the first word of the metric as a fuzzy root
            # e.g. "net_income" → "net_income", "operating_expenses" → "operating"
            metric_root = entities['metric_type'].split('_')[0]
            records = self.neo4j.query_graph(cypher, {
                'company': company,
                'metric_root': metric_root,
                'years': entities['years'],
                'top_k': top_k
            })
            if records:
                return records

        # Strategy 3: Company + most recent docs (no year in query or no year-filtered docs found)
        if company and not entities['years']:
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

        # Strategy 4: Metric type — no company identified.
        # 4a: metric + years → scoped search across all companies for those years
        # 4b: metric only → most recent values across all companies
        if entities['metric_type'] and not company:
            if entities['years']:
                cypher = """
                MATCH (c:Company)-[:HAS_METRIC]->(m:Metric {name: $metric_type})
                -[:FOR_YEAR]->(y:Year)
                WHERE y.value IN $years
                RETURN DISTINCT c.name as company, m.name as metric,
                       m.value as value, y.value as year,
                       'metric_only' as strategy
                ORDER BY y.value
                LIMIT $top_k
                """
                records = self.neo4j.query_graph(cypher, {
                    'metric_type': entities['metric_type'],
                    'years': entities['years'],
                    'top_k': top_k
                })
                if records:
                    return records

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

        # Strategy 5a: Question keyword search — for datasets without company names
        # (FinQA/ConvFinQA), the document's stored question field is more distinctive
        # than the full document text. Extract key terms and CONTAINS-match on d.question.
        stopwords = {'what', 'were', 'was', 'the', 'from', 'that', 'this', 'with',
                     'have', 'for', 'and', 'did', 'does', 'how', 'much', 'many',
                     'percent', 'percentage', 'change', 'total', 'which', 'year'}
        key_terms = [w for w in re.findall(r'\b[a-zA-Z]{5,}\b', query.lower())
                     if w not in stopwords][:5]

        if key_terms:
            # Build a Cypher WHERE clause that requires ALL key terms to appear
            conditions = ' AND '.join(
                [f"toLower(d.question) CONTAINS '{t}'" for t in key_terms]
            )
            cypher = f"""
            MATCH (d:Document)-[:ABOUT]->(c:Company)
            WHERE {conditions}
            RETURN c.name as company, d.text as text,
                   d.year as year, 'question_keyword_search' as strategy
            LIMIT $top_k
            """
            stage_a = self.neo4j.query_graph(cypher, {'top_k': top_k})
            if stage_a:
                return stage_a

            # Relax to 3 terms, then 2 terms if all-terms match fails
            if len(key_terms) > 3:
                conditions3 = ' AND '.join(
                    [f"toLower(d.question) CONTAINS '{t}'" for t in key_terms[:3]]
                )
                cypher3 = f"""
                MATCH (d:Document)-[:ABOUT]->(c:Company)
                WHERE {conditions3}
                RETURN c.name as company, d.text as text,
                       d.year as year, 'question_keyword_search' as strategy
                LIMIT $top_k
                """
                stage_a3 = self.neo4j.query_graph(cypher3, {'top_k': top_k})
                if stage_a3:
                    return stage_a3

            if len(key_terms) > 2:
                conditions2 = ' AND '.join(
                    [f"toLower(d.question) CONTAINS '{t}'" for t in key_terms[:2]]
                )
                cypher2 = f"""
                MATCH (d:Document)-[:ABOUT]->(c:Company)
                WHERE {conditions2}
                RETURN c.name as company, d.text as text,
                       d.year as year, 'question_keyword_search' as strategy
                LIMIT $top_k
                """
                stage_a2 = self.neo4j.query_graph(cypher2, {'top_k': top_k})
                if stage_a2:
                    return stage_a2

        # Strategy 5b: Full-text fallback on document body
        search_term = (company or entities['metric_type'] or query[:80])
        cypher = """
        MATCH (d:Document)-[:ABOUT]->(c:Company)
        WHERE toLower(d.text) CONTAINS toLower($search_term)
        RETURN c.name as company, d.text as text,
               d.year as year, 'fallback_text_search' as strategy
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
                # Metric result — plain number, no $ or units.
                # Models hallucinate scale conversions when they see "million".
                val = record.get('value', 0)
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                context_parts.append(
                    f"[Record {i+1}] "
                    f"Company: {record.get('company', 'Unknown')} | "
                    f"Year: {record.get('year', 'Unknown')} | "
                    f"Metric: {record.get('metric', 'Unknown')} | "
                    f"Value: {val_str}"
                )
            elif 'text' in record:
                # Document result
                context_parts.append(
                    f"[Record {i+1}] "
                    f"Company: {record.get('company', 'Unknown')} | "
                    f"Year: {record.get('year', 'Unknown')}"
                )
                context_parts.append(
                    f"   Text: {record.get('text', '')[:8000]}"
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