import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graphrag.neo4j_client import Neo4jClient
from src.models.ollama_client import OllamaClient
from datasets import load_from_disk, concatenate_datasets
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

    def __init__(self, neo4j_client: Neo4jClient, llm_model: str = "llama3.1:8b"):
        self.neo4j = neo4j_client
        self.llm = OllamaClient(llm_model, temperature=0.0)
        print(f"Graph builder initialized with model: {llm_model}")

    def extract_entities_with_llm(self, text: str, metadata: dict) -> dict:
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
        IMPROVED rule-based entity extraction optimized for markdown tables
        """
        metrics = []
        text_lower = text.lower()
        
        metrics.extend(self._extract_from_tables(text, text_lower))
        metrics.extend(self._extract_from_sentences(text_lower))
        metrics.extend(self._extract_from_statements(text_lower))
        
        unique_metrics = []
        seen = set()
        for m in metrics:
            key = (m['name'], round(m['value'], 2))
            if key not in seen:
                seen.add(key)
                unique_metrics.append(m)
        
        return {"metrics": unique_metrics}
    
    def _extract_from_tables(self, text: str, text_lower: str) -> list:
        """Extract metrics from markdown tables - FIXED: $ is now optional"""
        metrics = []
        lines = text.split('\n')
        
        # FIXED: \$? makes dollar sign optional
        markdown_patterns = [
            r'\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*\$?\s*([-]?[\d,]+\.?\d*)',
            r'\|\s*([a-z\s]+)\s*\|\s*\$?\s*([-]?[\d,]+\.?\d*)',
        ]
        
        metric_keywords = {
            # ── Revenue ──────────────────────────────────────────────────
            'total net revenue': 'net_revenue',
            'net revenue': 'net_revenue',
            'total revenue': 'revenue',
            'revenue': 'revenue',
            'net sales': 'revenue',
            'total sales': 'revenue',
            # ── Net income ───────────────────────────────────────────────
            'net income': 'net_income',
            'net earnings': 'net_income',
            'net profit': 'net_income',
            # ── EPS ──────────────────────────────────────────────────────
            'diluted earnings per share': 'earnings_per_share',
            'basic earnings per share': 'earnings_per_share',
            'earnings per share': 'earnings_per_share',
            'diluted eps': 'earnings_per_share',
            'basic eps': 'earnings_per_share',
            # ── Gross / operating ────────────────────────────────────────
            'gross profit': 'gross_profit',
            'gross margin': 'gross_profit',
            'income from operations': 'operating_income',
            'operating income': 'operating_income',
            'ebitda': 'ebitda',
            'income before income tax': 'income_before_tax',
            'income before taxes': 'income_before_tax',
            'income before tax': 'income_before_tax',
            'pretax income': 'income_before_tax',
            # ── Expenses ─────────────────────────────────────────────────
            'total operating expenses': 'operating_expenses',
            'operating expenses': 'operating_expenses',
            'operating costs': 'operating_expenses',
            'cost of goods sold': 'cost_of_goods_sold',
            'cost of sales': 'cost_of_goods_sold',
            'cost of revenue': 'cost_of_goods_sold',
            'cogs': 'cost_of_goods_sold',
            'research and development': 'research_development',
            'research & development': 'research_development',
            'selling, general and administrative': 'selling_general_admin',
            'selling general and administrative': 'selling_general_admin',
            'sg&a': 'selling_general_admin',
            'interest expense': 'interest_expense',
            'interest paid': 'interest_expense',
            'interest cost': 'interest_expense',
            'depreciation and amortization': 'depreciation',
            'depreciation': 'depreciation',
            'amortization': 'depreciation',
            'capital expenditures': 'capital_expenditures',
            'capital expenditure': 'capital_expenditures',
            'capex': 'capital_expenditures',
            'aircraft fuel expense': 'fuel_expense',
            'fuel expense': 'fuel_expense',
            'fuel costs': 'fuel_expense',
            # ── Tax ──────────────────────────────────────────────────────
            'provision for income taxes': 'tax_expense',
            'income tax expense': 'tax_expense',
            'income tax': 'tax_expense',
            'deferred income tax assets': 'deferred_tax_asset',
            'deferred tax assets': 'deferred_tax_asset',
            'deferred income tax liabilities': 'deferred_tax_liability',
            'deferred tax liabilities': 'deferred_tax_liability',
            'deferred income tax': 'deferred_tax',
            'deferred tax': 'deferred_tax',
            # ── Stock compensation / awards ───────────────────────────────
            'stock-based compensation': 'stock_compensation',
            'share-based compensation': 'stock_compensation',
            'stock compensation expense': 'stock_compensation',
            'restricted stock units': 'stock_awards',
            'stock awards': 'stock_awards',
            'restricted stock': 'stock_awards',
            # ── Employee benefits ─────────────────────────────────────────
            'net periodic pension cost': 'pension',
            'pension expense': 'pension',
            'pension cost': 'pension',
            'other post-retirement benefit': 'post_retirement_benefits',
            'postretirement benefit': 'post_retirement_benefits',
            'benefit obligation': 'post_retirement_benefits',
            # ── Balance sheet — assets ────────────────────────────────────
            'total assets': 'total_assets',
            'total current assets': 'current_assets',
            'current assets': 'current_assets',
            'cash and cash equivalents': 'cash_and_equivalents',
            'cash and equivalents': 'cash_and_equivalents',
            'accounts receivable': 'accounts_receivable',
            'trade receivables': 'accounts_receivable',
            'inventories': 'inventory',
            'inventory': 'inventory',
            'goodwill': 'goodwill',
            'other intangible assets': 'intangible_assets',
            'intangible assets': 'intangible_assets',
            # ── Balance sheet — equity ────────────────────────────────────
            "total shareholders' equity": 'shareholders_equity',
            'total shareholders equity': 'shareholders_equity',
            'total stockholders equity': 'shareholders_equity',
            "shareholders' equity": 'shareholders_equity',
            'shareholders equity': 'shareholders_equity',
            'stockholders equity': 'shareholders_equity',
            'accumulated deficit': 'retained_earnings',
            'retained earnings': 'retained_earnings',
            # ── Balance sheet — liabilities ───────────────────────────────
            'total liabilities': 'total_liabilities',
            'total current liabilities': 'current_liabilities',
            'current liabilities': 'current_liabilities',
            'total long-term debt': 'long_term_debt',
            'long-term debt': 'long_term_debt',
            'long term debt': 'long_term_debt',
            'total debt': 'total_debt',
            'current portion of long-term debt': 'short_term_debt',
            'short-term debt': 'short_term_debt',
            'short term debt': 'short_term_debt',
            'notes payable': 'short_term_debt',
            'accounts payable': 'accounts_payable',
            'trade payables': 'accounts_payable',
            # ── Leases ────────────────────────────────────────────────────
            'operating lease right-of-use': 'operating_lease',
            'operating lease liability': 'operating_lease',
            'operating lease': 'operating_lease',
            'finance lease right-of-use': 'finance_lease',
            'finance lease liability': 'finance_lease',
            'finance lease': 'finance_lease',
            'right-of-use asset': 'operating_lease',
            'total lease obligation': 'lease_obligation',
            'lease obligation': 'lease_obligation',
            'rental expense': 'lease_obligation',
            # ── Cash flow ─────────────────────────────────────────────────
            'net cash used in operating': 'cash_flow_operations',
            'net cash provided by operating': 'cash_flow_operations',
            'cash provided by operating activities': 'cash_flow_operations',
            'free cash flow': 'free_cash_flow',
            'dividends paid': 'dividends_paid',
            'dividends': 'dividends_paid',
        }

        for line in lines:
            if '|' not in line:
                continue
                
            for pattern in markdown_patterns:
                matches = re.finditer(pattern, line.lower())
                for match in matches:
                    label = match.group(1).strip()
                    value_str = match.group(2).replace(',', '').strip()
                    
                    try:
                        value = float(value_str)

                        # Skip year values: only integers in the year range with no decimal portion
                        if value == int(value) and 1900 <= value <= 2100:
                            continue

                        # Skip very small values
                        if abs(value) < 0.01:
                            continue

                        # Longest-match wins to avoid "assets" beating "current assets"
                        best_keyword = ''
                        best_metric = None
                        for keyword, metric_name in metric_keywords.items():
                            if keyword in label and len(keyword) > len(best_keyword):
                                best_keyword = keyword
                                best_metric = metric_name

                        if best_metric:
                            normalized_value = abs(value)
                            if 'billion' in line.lower():
                                normalized_value = normalized_value * 1000
                            elif 'thousand' in line.lower():
                                normalized_value = normalized_value / 1000
                            if normalized_value > 0:
                                metrics.append({
                                    'name': best_metric,
                                    'value': round(normalized_value, 2),
                                    'unit': 'millions'
                                })
                                break
                            
                    except (ValueError, AttributeError):
                        continue
        
        return metrics
    
    def _extract_from_sentences(self, text_lower: str) -> list:
        """Extract from natural language"""
        metrics = []
        
        patterns = [
            # Interest patterns
            (r'interest\s+expense[s]?\s+would\s+(?:change|increase|decrease)\s+by\s+\$\s+([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'interest_expense'),
            (r'interest\s+expense[s]?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'interest_expense'),
            (r'interest\s+cost[s]?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'interest_expense'),
            (r'paid\s+interest\s+of\s+\$\s+([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'interest_expense'),
            (r'interest\s+paid\s+\$\s+([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'interest_expense'),
            
            # Revenue patterns
            (r'net\s+revenue[s]?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'net_revenue'),
            (r'net\s+revenue[s]?\s+(?:increased|decreased|grew)\s+(?:to|by)\s+\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'net_revenue'),
            (r'total\s+revenue[s]?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'revenue'),
            (r'revenue[s]?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'revenue'),
            (r'(?:total\s+)?sales\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'revenue'),
            (r'revenue[s]?\s+(?:increased|decreased|grew)\s+(?:to|by)\s+\$\s+([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'revenue'),

            # Net income patterns
            (r'net\s+income\s+would\s+(?:change|increase|decrease)\s+(?:by|to)\s+\$\s+([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'net_income'),
            (r'net\s+income\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'net_income'),
            (r'net\s+earnings?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'net_income'),
            (r'net\s+profit\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'net_income'),
            (r'net\s+income\s+(?:increased|decreased)\s+(?:to|by)\s+\$\s+([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'net_income'),
            
            # Operating expenses
            (r'operating\s+expense[s]?\s+would\s+(?:change|increase|decrease)\s+(?:by|to)\s+\$\s+([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'operating_expenses'),
            (r'operating\s+expense[s]?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'operating_expenses'),
            (r'operating\s+cost[s]?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'operating_expenses'),
            
            # Assets
            (r'total\s+assets\s+would\s+(?:change|increase|decrease)\s+(?:by|to)\s+\$\s+([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'total_assets'),
            (r'total\s+assets\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'total_assets'),
            (r'cash\s+and\s+cash\s+equivalents\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'cash_and_equivalents'),
            (r'cash\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'cash_and_equivalents'),
            
            # EPS
            (r'earnings\s+per\s+(?:diluted\s+)?share\s+(?:was|were|of)?\s*\$\s*([\d,]+\.?\d*)', 'earnings_per_share'),
            (r'eps\s+(?:was|were|of)?\s*\$\s*([\d,]+\.?\d*)', 'earnings_per_share'),
            (r'diluted\s+eps\s+\$\s+([\d,]+\.?\d*)', 'earnings_per_share'),
            
            # Other metrics
            (r'depreciation\s+(?:and\s+amortization\s+)?expense[s]?\s+\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'depreciation'),
            (r'capital\s+expenditure[s]?\s+\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'capital_expenditures'),
            (r'cost\s+of\s+(?:goods\s+sold|revenue|sales)\s+\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'cost_of_goods_sold'),
            (r'gross\s+profit\s+\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'gross_profit'),
            (r'operating\s+income\s+\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'operating_income'),
            (r'ebitda\s+\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'ebitda'),
            # Debt — long-term vs total are different
            (r'(?:total\s+)?long[- ]term\s+debt\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'long_term_debt'),
            (r'total\s+debt\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'total_debt'),
            (r'short[- ]term\s+(?:debt|borrowings?)\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'short_term_debt'),
            # Balance sheet
            (r'total\s+liabilities\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'total_liabilities'),
            (r'(?:total\s+)?current\s+assets\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'current_assets'),
            (r'(?:total\s+)?current\s+liabilities\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'current_liabilities'),
            (r'(?:total\s+)?(?:shareholders?|stockholders?)[\'s]?\s+equity\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'shareholders_equity'),
            (r'retained\s+earnings?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'retained_earnings'),
            (r'accounts\s+receivable\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'accounts_receivable'),
            (r'accounts\s+payable\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'accounts_payable'),
            (r'inventor(?:y|ies)\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'inventory'),
            (r'goodwill\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'goodwill'),
            (r'intangible\s+assets?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'intangible_assets'),
            # Income statement extras
            (r'income\s+before\s+(?:income\s+)?tax(?:es)?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'income_before_tax'),
            (r'research\s+and\s+development\s+(?:expense[s]?\s+)?(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'research_development'),
            (r'selling[,]?\s+general\s+and\s+administrative\s+(?:expense[s]?\s+)?(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'selling_general_admin'),
            # Tax
            (r'(?:income\s+)?tax\s+expense\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'tax_expense'),
            (r'provision\s+for\s+income\s+taxes?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'tax_expense'),
            (r'deferred\s+(?:income\s+)?tax\s+assets?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'deferred_tax_asset'),
            (r'deferred\s+(?:income\s+)?tax\s+liabilit\w+\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'deferred_tax_liability'),
            # Stock compensation
            (r'stock[- ]based\s+compensation\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'stock_compensation'),
            (r'share[- ]based\s+compensation\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'stock_compensation'),
            (r'(?:restricted\s+stock|stock\s+award|rsu)\s+(?:was|were|of|valued\s+at)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'stock_awards'),
            # Pension vs OPEB — separate
            (r'net\s+periodic\s+pension\s+cost\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'pension'),
            (r'pension\s+(?:expense|cost)\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'pension'),
            (r'(?:other\s+)?post(?:[-\s])?retirement\s+benefit\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'post_retirement_benefits'),
            (r'benefit\s+obligation\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'post_retirement_benefits'),
            # Leases — split by type
            (r'operating\s+lease\s+(?:obligation|liability|expense|cost)?\s*(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'operating_lease'),
            (r'finance\s+lease\s+(?:obligation|liability|expense|cost)?\s*(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'finance_lease'),
            (r'(?:total\s+)?rental\s+expense[s]?\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'lease_obligation'),
            # Cash flow
            (r'(?:net\s+)?cash\s+(?:provided\s+by|used\s+in|from)\s+operating\s+activit\w+\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'cash_flow_operations'),
            (r'free\s+cash\s+flow\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'free_cash_flow'),
            (r'dividends?\s+paid\s+(?:was|were|of|totaled?)?\s*\$\s*([\d,]+\.?\d*)\s*(million|billion|thousand)?', 'dividends_paid'),
        ]

        for pattern, metric_name in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    value_str = match[0].replace(',', '').strip()
                    value = float(value_str)

                    # Only skip exact integers that look like years
                    if value == int(value) and 1900 <= value <= 2100:
                        continue
                    
                    unit = match[1] if len(match) > 1 and match[1] else 'unknown'
                    value = self._normalize_value(value, unit)
                    
                    if value > 0:
                        metrics.append({
                            'name': metric_name,
                            'value': value,
                            'unit': 'millions'
                        })
                except (ValueError, IndexError):
                    continue
        
        return metrics
    
    def _extract_from_statements(self, text_lower: str) -> list:
        """Extract from structured financial statement formats"""
        metrics = []
        
        statement_patterns = [
            (r'revenue\s+\$\s+([\d,]+\.?\d*)', 'revenue'),
            (r'net\s+income\s+\$\s+([\d,]+\.?\d*)', 'net_income'),
            (r'total\s+assets\s+\$\s+([\d,]+\.?\d*)', 'total_assets'),
            (r'operating\s+expenses\s+\$\s+([\d,]+\.?\d*)', 'operating_expenses'),
        ]
        
        for pattern, metric_name in statement_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    value = float(match.replace(',', ''))
                    if 1900 <= value <= 2100:
                        continue
                    if value > 0:
                        metrics.append({
                            'name': metric_name,
                            'value': round(value, 2),
                            'unit': 'millions'
                        })
                except ValueError:
                    continue

        return metrics
    
    def _infer_unit_from_context(self, text: str, value: float) -> str:
        """Infer unit based on context and value magnitude"""
        text_lower = text.lower()
        
        if 'billion' in text_lower:
            return 'billion'
        elif 'thousand' in text_lower:
            return 'thousand'
        elif 'million' in text_lower:
            return 'million'
        
        if value < 100:
            return 'billion'
        elif value < 10000:
            return 'million'
        else:
            return 'thousand'
    
    def _normalize_value(self, value: float, unit: str) -> float:
        """Normalize all values to millions"""
        if 'billion' in unit.lower():
            return round(value * 1000, 2)
        elif 'thousand' in unit.lower():
            return round(value / 1000, 2)
        elif value < 100 and unit == 'unknown':
            return round(value * 1000, 2)
        else:
            return round(value, 2)

    # Metric keyword map reused by multi-year table parser (longest-match wins)
    _TABLE_METRIC_MAP = {
        # Revenue
        'total net revenue': 'net_revenue',
        'net revenue': 'net_revenue',
        'total revenue': 'revenue',
        'revenue': 'revenue',
        'net sales': 'revenue',
        'total sales': 'revenue',
        # Net income
        'net income': 'net_income',
        'net earnings': 'net_income',
        'net profit': 'net_income',
        # EPS
        'diluted earnings per share': 'earnings_per_share',
        'basic earnings per share': 'earnings_per_share',
        'earnings per share': 'earnings_per_share',
        # Income statement
        'gross profit': 'gross_profit',
        'gross margin': 'gross_profit',
        'income from operations': 'operating_income',
        'operating income': 'operating_income',
        'ebitda': 'ebitda',
        'income before income tax': 'income_before_tax',
        'income before taxes': 'income_before_tax',
        'income before tax': 'income_before_tax',
        'pretax income': 'income_before_tax',
        # Expenses
        'total operating expenses': 'operating_expenses',
        'operating expenses': 'operating_expenses',
        'operating costs': 'operating_expenses',
        'cost of goods sold': 'cost_of_goods_sold',
        'cost of sales': 'cost_of_goods_sold',
        'cost of revenue': 'cost_of_goods_sold',
        'research and development': 'research_development',
        'research & development': 'research_development',
        'selling, general and administrative': 'selling_general_admin',
        'selling general and administrative': 'selling_general_admin',
        'sg&a': 'selling_general_admin',
        'interest expense': 'interest_expense',
        'interest cost': 'interest_expense',
        'depreciation and amortization': 'depreciation',
        'depreciation': 'depreciation',
        'amortization': 'depreciation',
        'capital expenditures': 'capital_expenditures',
        'capital expenditure': 'capital_expenditures',
        'capex': 'capital_expenditures',
        'aircraft fuel expense': 'fuel_expense',
        'fuel expense': 'fuel_expense',
        # Tax
        'provision for income taxes': 'tax_expense',
        'income tax expense': 'tax_expense',
        'income tax': 'tax_expense',
        'deferred income tax assets': 'deferred_tax_asset',
        'deferred tax assets': 'deferred_tax_asset',
        'deferred income tax liabilities': 'deferred_tax_liability',
        'deferred tax liabilities': 'deferred_tax_liability',
        'deferred income tax': 'deferred_tax',
        'deferred tax': 'deferred_tax',
        # Stock
        'stock-based compensation': 'stock_compensation',
        'share-based compensation': 'stock_compensation',
        'restricted stock units': 'stock_awards',
        'stock awards': 'stock_awards',
        'restricted stock': 'stock_awards',
        # Employee benefits
        'net periodic pension cost': 'pension',
        'pension expense': 'pension',
        'pension cost': 'pension',
        'other post-retirement benefit': 'post_retirement_benefits',
        'postretirement benefit': 'post_retirement_benefits',
        'benefit obligation': 'post_retirement_benefits',
        # Balance sheet — assets
        'total assets': 'total_assets',
        'total current assets': 'current_assets',
        'current assets': 'current_assets',
        'cash and cash equivalents': 'cash_and_equivalents',
        'cash and equivalents': 'cash_and_equivalents',
        'accounts receivable': 'accounts_receivable',
        'trade receivables': 'accounts_receivable',
        'inventories': 'inventory',
        'inventory': 'inventory',
        'goodwill': 'goodwill',
        'other intangible assets': 'intangible_assets',
        'intangible assets': 'intangible_assets',
        # Balance sheet — equity
        "total shareholders' equity": 'shareholders_equity',
        'total shareholders equity': 'shareholders_equity',
        'total stockholders equity': 'shareholders_equity',
        "shareholders' equity": 'shareholders_equity',
        'shareholders equity': 'shareholders_equity',
        'stockholders equity': 'shareholders_equity',
        'accumulated deficit': 'retained_earnings',
        'retained earnings': 'retained_earnings',
        # Balance sheet — liabilities
        'total liabilities': 'total_liabilities',
        'total current liabilities': 'current_liabilities',
        'current liabilities': 'current_liabilities',
        'total long-term debt': 'long_term_debt',
        'long-term debt': 'long_term_debt',
        'long term debt': 'long_term_debt',
        'total debt': 'total_debt',
        'current portion of long-term debt': 'short_term_debt',
        'short-term debt': 'short_term_debt',
        'notes payable': 'short_term_debt',
        'accounts payable': 'accounts_payable',
        'trade payables': 'accounts_payable',
        # Leases — split by type
        'operating lease right-of-use': 'operating_lease',
        'operating lease liability': 'operating_lease',
        'operating lease': 'operating_lease',
        'finance lease right-of-use': 'finance_lease',
        'finance lease liability': 'finance_lease',
        'finance lease': 'finance_lease',
        'right-of-use asset': 'operating_lease',
        'total lease obligation': 'lease_obligation',
        'lease obligation': 'lease_obligation',
        'rental expense': 'lease_obligation',
        # Cash flow
        'net cash used in operating': 'cash_flow_operations',
        'net cash provided by operating': 'cash_flow_operations',
        'free cash flow': 'free_cash_flow',
        'dividends paid': 'dividends_paid',
        'dividends': 'dividends_paid',
    }

    def parse_table_year_columns(self, text: str) -> list:
        """
        Parse markdown tables where column headers contain years.
        Extracts (year, metric_name, value) for every year column found.
        This gives multi-year coverage from a single document.
        """
        results = []
        lines = [l for l in text.split('\n') if '|' in l]

        if len(lines) < 2:
            return results

        # Find the first header row that contains at least one year
        year_cols = {}   # col_index -> year
        header_row_idx = None

        for row_idx, line in enumerate(lines):
            cells = [c.strip() for c in line.split('|')]
            for col_idx, cell in enumerate(cells):
                m = re.search(r'\b(19[9]\d|20[0-3]\d)\b', cell)
                if m:
                    year_cols[col_idx] = int(m.group())
            if year_cols:
                header_row_idx = row_idx
                break

        if not year_cols:
            return results

        # Parse data rows that follow the header
        for line in lines[header_row_idx + 1:]:
            # Skip separator rows (---|---|---)
            if re.match(r'\|\s*[-:]+\s*\|', line):
                continue

            cells = [c.strip() for c in line.split('|')]

            # First non-empty, non-separator cell is the metric label
            metric_label = ''
            for cell in cells:
                if cell and not re.match(r'^[-:]+$', cell):
                    metric_label = cell.lower()
                    break

            if not metric_label:
                continue

            # Map to canonical metric name (longest keyword match wins)
            metric_name = None
            best_len = 0
            for keyword, canonical in self._TABLE_METRIC_MAP.items():
                if keyword in metric_label and len(keyword) > best_len:
                    metric_name = canonical
                    best_len = len(keyword)

            if not metric_name:
                continue

            # Extract value for each year column
            for col_idx, year in year_cols.items():
                if col_idx >= len(cells):
                    continue
                val_str = cells[col_idx].replace(',', '').replace('$', '').replace('(', '-').replace(')', '').strip()
                try:
                    val = float(val_str)
                    if abs(val) < 0.01:
                        continue
                    results.append((year, metric_name, round(abs(val), 2)))
                except ValueError:
                    continue

        return results

    @staticmethod
    def _extract_full_context(example: dict) -> str:
        """
        Build full document context from an example, handling all three dataset formats:
        - FinQA / ConvFinQA: pre_text + table + post_text
        - TAT-DQA: context field only
        """
        parts = []
        if example.get('pre_text'):
            parts.append(str(example['pre_text']).strip())
        if example.get('table'):
            parts.append(str(example['table']).strip())
        if example.get('post_text'):
            parts.append(str(example['post_text']).strip())
        if not parts and example.get('context'):
            parts.append(str(example['context']).strip())
        return "\n\n".join(parts)

    @staticmethod
    def _extract_year(example: dict) -> int:
        """Extract year from report_year field or fall back to scanning the question."""
        year_raw = example.get('report_year', 0)
        try:
            year = int(str(year_raw))
            if year > 0:
                return year
        except (ValueError, TypeError):
            pass
        # Fall back: scan question for a 4-digit year
        m = re.search(r'\b(19[9]\d|20[0-3]\d)\b', example.get('question', ''))
        return int(m.group()) if m else 0

    def build_from_dataset(
        self,
        dataset_path: str,
        dataset_name: str = 'dataset',
        max_examples: int = 50,
        use_llm: bool = False
    ):
        print(f"\n{'='*60}")
        print("Building Knowledge Graph")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_path} ({dataset_name})")
        print(f"Max examples: {max_examples}")
        print(f"Using LLM extraction: {use_llm}")

        self.neo4j.create_indexes()

        dataset = load_from_disk(dataset_path)
        data = concatenate_datasets(list(dataset.values()))

        examples = data.select(range(min(max_examples, len(data))))

        companies_created = set()
        metrics_created = 0
        documents_created = 0

        for i, example in enumerate(tqdm(examples, desc="Building graph")):
            try:
                # Works for FinQA/ConvFinQA (pre_text+table+post_text) and TAT-DQA (context)
                context = self._extract_full_context(example)

                if not context or len(context) < 50:
                    continue

                company = example.get('company_name', 'Unknown')
                year = self._extract_year(example)
                question = example.get('question', '')

                # Prefix doc_id with dataset_name to avoid collisions across datasets
                raw_id = example.get('id', f'doc_{i}')
                doc_id = f"{dataset_name}_{raw_id}"

                if company not in companies_created:
                    self.neo4j.create_company(company, {
                        'sector': example.get('company_sector', 'Unknown'),
                        'industry': example.get('company_industry', 'Unknown'),
                        'headquarters': example.get('company_headquarters', 'Unknown'),
                        'symbol': example.get('company_symbol', 'Unknown')
                    })
                    companies_created.add(company)

                self.neo4j.create_document(
                    doc_id=doc_id,
                    text=context[:8000],
                    properties={
                        'company': company,
                        'year': year,
                        'question': question[:500],
                    }
                )
                documents_created += 1

                if year > 0:
                    self.neo4j.link_document_to_company(doc_id, company, year)

                metadata = {'company': company, 'year': year}

                if use_llm:
                    entities = self.extract_entities_with_llm(context, metadata)
                else:
                    entities = self.extract_entities_rule_based(context, metadata)

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

                # Multi-year extraction: parse year columns from table field (or context)
                table_text = example.get('table', '') or context
                for (metric_year, metric_name, metric_value) in self.parse_table_year_columns(table_text):
                    try:
                        self.neo4j.create_metric(
                            name=metric_name,
                            value=metric_value,
                            year=metric_year,
                            company=company,
                            properties={'unit': 'millions', 'source_doc': doc_id}
                        )
                        metrics_created += 1
                    except Exception:
                        continue

            except Exception:
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
    print("="*60)
    print("Testing Graph Builder")
    print("="*60)

    neo4j = Neo4jClient()
    neo4j.clear_database()

    builder = GraphBuilder(neo4j, llm_model="llama3.1:8b")

    builder.build_from_dataset(
        dataset_path="data/benchmarks/t2-ragbench-FinQA",
        max_examples=50,
        use_llm=False
    )

    print("\n" + "="*60)
    print("Testing Graph Queries")
    print("="*60)

    companies = neo4j.query_graph(
        "MATCH (c:Company) RETURN c.name as company LIMIT 10"
    )
    print(f"\nSample companies:")
    for c in companies:
        print(f"   {c['company']}")

    metrics = neo4j.query_graph("""
        MATCH (c:Company)-[:HAS_METRIC]->(m:Metric)-[:FOR_YEAR]->(y:Year)
        RETURN c.name as company, m.name as metric, m.value as value, y.value as year
        ORDER BY c.name, y.value DESC
        LIMIT 20
    """)

    print(f"\nSample metrics:")
    for m in metrics:
        print(f"   {m['company']} ({m['year']}): {m['metric']} = ${m['value']}M")

    neo4j.close()


if __name__ == "__main__":
    main()