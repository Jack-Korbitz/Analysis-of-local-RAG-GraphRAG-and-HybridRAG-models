from neo4j import GraphDatabase
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


class Neo4jClient:
    """Client for interacting with Neo4j graph database"""
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize Neo4j client

        Reads from environment variables if not provided:
            NEO4J_URI
            NEO4J_USER
            NEO4J_PASSWORD

        Args:
            uri: Neo4j connection URI (overrides env var)
            user: Username (overrides env var)
            password: Password (overrides env var)
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")

        if not self.uri or not self.user or not self.password:
            raise ValueError(
                "Neo4j credentials missing. "
                "Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in your .env file."
            )

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        
        self._verify_connection()
        print(f"Neo4j client initialized: {self.uri}")
    
    def _verify_connection(self):
        """Verify connection to Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            print("Connected to Neo4j successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def close(self):
        """Close the connection"""
        self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        with self.driver.session() as session:
            session.run(
                "CREATE INDEX company_name IF NOT EXISTS "
                "FOR (c:Company) ON (c.name)"
            )
            session.run(
                "CREATE INDEX document_id IF NOT EXISTS "
                "FOR (d:Document) ON (d.id)"
            )
            session.run(
                "CREATE INDEX metric_name IF NOT EXISTS "
                "FOR (m:Metric) ON (m.name)"
            )
            session.run(
                "CREATE INDEX year_value IF NOT EXISTS "
                "FOR (y:Year) ON (y.value)"
            )
        print("Indexes created")
    
    def create_company(self, name: str, properties: Dict = None) -> str:
        """Create a Company node"""
        with self.driver.session() as session:
            query = """
            MERGE (c:Company {name: $name})
            SET c += $properties
            RETURN c.name as name
            """
            result = session.run(
                query,
                name=name,
                properties=properties or {}
            )
            return result.single()["name"]
    
    def create_document(self, doc_id: str, text: str, properties: Dict = None) -> str:
        """Create a Document node"""
        with self.driver.session() as session:
            query = """
            CREATE (d:Document {id: $doc_id, text: $text})
            SET d += $properties
            RETURN d.id as id
            """
            result = session.run(
                query,
                doc_id=doc_id,
                text=text,
                properties=properties or {}
            )
            return result.single()["id"]
    
    def create_metric(
        self,
        name: str,
        value: float,
        year: int,
        company: str,
        properties: Dict = None
    ):
        """Create a Metric and link it to Company and Year"""
        with self.driver.session() as session:
            query = """
            MATCH (c:Company {name: $company})
            MERGE (y:Year {value: $year})
            CREATE (m:Metric {name: $name, value: $value})
            SET m += $properties
            CREATE (c)-[:HAS_METRIC]->(m)
            CREATE (m)-[:FOR_YEAR]->(y)
            RETURN m.name as name
            """
            result = session.run(
                query,
                name=name,
                value=value,
                year=year,
                company=company,
                properties=properties or {}
            )
            return result.single()["name"]
    
    def link_document_to_company(
        self,
        doc_id: str,
        company_name: str,
        year: int
    ):
        """Link a Document to a Company and Year"""
        with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $doc_id})
            MATCH (c:Company {name: $company_name})
            MERGE (y:Year {value: $year})
            CREATE (d)-[:ABOUT]->(c)
            CREATE (d)-[:FROM_YEAR]->(y)
            """
            session.run(
                query,
                doc_id=doc_id,
                company_name=company_name,
                year=year
            )
    
    def query_graph(
        self,
        cypher_query: str,
        parameters: Dict = None
    ) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [dict(record) for record in result]
    
    def get_company_metrics(
        self,
        company: str,
        year: Optional[int] = None
    ) -> List[Dict]:
        """Get all metrics for a company, optionally filtered by year"""
        with self.driver.session() as session:
            if year:
                query = """
                MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)-[:FOR_YEAR]->(y:Year {value: $year})
                RETURN m.name as metric, m.value as value, y.value as year
                """
                result = session.run(query, company=company, year=year)
            else:
                query = """
                MATCH (c:Company {name: $company})-[:HAS_METRIC]->(m:Metric)-[:FOR_YEAR]->(y:Year)
                RETURN m.name as metric, m.value as value, y.value as year
                ORDER BY y.value DESC
                """
                result = session.run(query, company=company)
            
            return [dict(record) for record in result]
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph"""
        with self.driver.session() as session:
            query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            """
            result = session.run(query)
            stats = {record["label"]: record["count"] for record in result}
            
            rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
            rel_result = session.run(rel_query)
            stats["relationships"] = rel_result.single()["count"]
            
            return stats


def main():
    """Test Neo4j client"""
    print("="*60)
    print("Testing Neo4j Client")
    print("="*60)
    
    # Initialize client
    client = Neo4jClient()
    
    # Clear database
    print("\nClearing database...")
    client.clear_database()
    
    # Create indexes
    print("\nCreating indexes...")
    client.create_indexes()
    
    # Create test data
    print("\nCreating test data...")
    
    client.create_company("Analog Devices", {"sector": "Technology"})
    client.create_company("Intel", {"sector": "Technology"})
    
    client.create_metric(
        name="interest_expense",
        value=3.8,
        year=2009,
        company="Analog Devices"
    )
    
    client.create_metric(
        name="revenue",
        value=2500.0,
        year=2009,
        company="Analog Devices"
    )
    
    # Query the graph
    print("\nQuerying graph...")
    metrics = client.get_company_metrics("Analog Devices", year=2009)
    
    print("\nAnalog Devices metrics for 2009:")
    for metric in metrics:
        print(f"   {metric['metric']}: ${metric['value']}M")
    
    # Get stats
    print("\nGraph statistics:")
    stats = client.get_graph_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Close connection
    client.close()
    print("\nConnection closed")


if __name__ == "__main__":
    main()