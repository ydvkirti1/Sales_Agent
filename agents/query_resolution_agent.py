"""
Query Resolution Agent
Converts natural language queries into structured SQL queries using LLM
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional
import logging
import re
from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryResolutionAgent:
    """
    Agent that converts natural language to SQL queries.
    
    This agent uses LLM to understand user intent and generate
    appropriate SQL queries for DuckDB execution.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize Query Resolution Agent
        
        Args:
            llm: Optional pre-initialized LLM. If None, creates new one.
        """
        self.llm = llm if llm else self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize OpenAI LLM"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for query resolution"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query generator for retail sales data analysis.
Your task is to convert natural language questions into accurate SQL queries using DuckDB syntax.

Available datasets and their schemas will be provided. You must:
1. Identify which dataset(s) to query
2. Generate valid SQL that answers the question
3. Use appropriate aggregations (SUM, COUNT, AVG, etc.)
4. Handle date filtering correctly
5. Return ONLY the SQL query, no explanations

SQL Guidelines:
- Use lowercase for table names (they are registered in DuckDB)
- Use proper date functions for filtering
- Use aggregation functions for summary statistics
- Include GROUP BY when using aggregations
- Use LIMIT when appropriate for large result sets

Example transformations:
- "total sales" -> SUM(amount) or SUM(amount_column)
- "number of orders" -> COUNT(*) or COUNT(order_id)
- "average sales" -> AVG(amount)
- "by category" -> GROUP BY category
- "in Q3" -> WHERE date_column BETWEEN '2023-07-01' AND '2023-09-30'
- "YoY growth" -> Compare current period with previous year period

Return ONLY the SQL query without markdown formatting or explanations."""),
            ("human", """Schema Information:
{schema_info}

User Question: {user_query}

Generate the SQL query:""")
        ])
    
    def resolve_query(self, user_query: str, schema_info: str) -> Dict[str, Any]:
        """
        Convert natural language query to SQL
        
        Args:
            user_query: Natural language question
            schema_info: Schema information about available datasets
            
        Returns:
            Dictionary with SQL query and metadata:
            {
                "sql_query": str,
                "original_query": str,
                "status": "success" | "error",
                "error": str (if status is error)
            }
        """
        if not user_query or not user_query.strip():
            logger.warning("Empty user query provided")
            return {
                "sql_query": None,
                "original_query": user_query,
                "status": "error",
                "error": "Empty query provided"
            }
        
        try:
            logger.info(f"Resolving query: {user_query[:100]}...")
            
            prompt = self.prompt_template.format_messages(
                schema_info=schema_info or "No schema information available",
                user_query=user_query
            )
            
            response = self.llm.invoke(prompt)
            sql_query = response.content.strip()
            
            # Clean up SQL query (remove markdown code blocks if present)
            sql_query = self._clean_sql_query(sql_query)
            
            # Basic validation
            if not sql_query or len(sql_query) < 10:
                raise ValueError("Generated SQL query is too short or empty")
            
            logger.info(f"Successfully generated SQL query ({len(sql_query)} chars)")
            
            return {
                "sql_query": sql_query,
                "original_query": user_query,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Query resolution error: {str(e)}", exc_info=True)
            return {
                "sql_query": None,
                "original_query": user_query,
                "status": "error",
                "error": str(e)
            }
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean SQL query by removing markdown formatting"""
        # Remove markdown code blocks
        sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'^```\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'```\s*$', '', sql_query, flags=re.MULTILINE)
        
        # Remove leading/trailing whitespace
        sql_query = sql_query.strip()
        
        return sql_query

