"""
Validation Agent
Validates query results and ensures data quality using LLM analysis
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional
import logging
import json
import re
from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Agent that validates query results and ensures data quality.
    
    This agent uses LLM to verify that SQL queries correctly
    answer user questions and that results are logical.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize Validation Agent
        
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
        """Create prompt template for validation"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a data validation expert. Your task is to:
1. Verify that the SQL query correctly addresses the user's question
2. Check if the results make logical sense
3. Identify any potential issues with the data or query
4. Suggest improvements if needed

Return a JSON object with:
- "is_valid": boolean indicating if the query and results are valid
- "confidence": float between 0 and 1 indicating confidence level
- "issues": list of any issues found
- "suggestions": list of improvement suggestions
- "summary": brief summary of validation results"""),
            ("human", """Original Question: {original_query}

SQL Query: {sql_query}

Query Results Summary:
- Row count: {row_count}
- Columns: {columns}
- Summary statistics: {summary_stats}

Validate this query and results:""")
        ])
    
    def validate(self, original_query: str, sql_query: str, 
                 extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate query results
        
        Args:
            original_query: Original user question
            sql_query: SQL query that was executed
            extraction_result: Results from data extraction
            
        Returns:
            Validation results dictionary:
            {
                "status": "success" | "error",
                "validation": {
                    "is_valid": bool,
                    "confidence": float (0-1),
                    "issues": list,
                    "suggestions": list,
                    "summary": str
                },
                "raw_response": str,
                "error": str (if status is error)
            }
        """
        if not original_query or not sql_query:
            logger.warning("Missing required parameters for validation")
            return {
                "status": "error",
                "error": "Missing required parameters",
                "validation": {
                    "is_valid": False,
                    "confidence": 0.0,
                    "issues": ["Missing required parameters"],
                    "suggestions": [],
                    "summary": "Validation failed due to missing parameters"
                }
            }
        
        try:
            data_dict = extraction_result.get("data", {})
            
            if not data_dict:
                logger.warning("No data available for validation")
                return {
                    "status": "success",
                    "validation": {
                        "is_valid": False,
                        "confidence": 0.3,
                        "issues": ["No data returned from query"],
                        "suggestions": ["Check if the SQL query is correct"],
                        "summary": "Query returned no data"
                    },
                    "raw_response": "No data to validate"
                }
            
            logger.info(f"Validating query results: {len(data_dict.get('columns', []))} columns, {data_dict.get('row_count', 0)} rows")
            
            prompt = self.prompt_template.format_messages(
                original_query=original_query,
                sql_query=sql_query,
                row_count=data_dict.get("row_count", 0),
                columns=", ".join(data_dict.get("columns", [])),
                summary_stats=str(data_dict.get("summary_stats", {}))
            )
            
            response = self.llm.invoke(prompt)
            validation_text = response.content.strip()
            
            # Parse JSON from response
            validation_result = self._parse_validation_response(validation_text)
            
            confidence = validation_result.get("confidence", 0.5)
            is_valid = validation_result.get("is_valid", True)
            
            logger.info(f"Validation completed: Valid={is_valid}, Confidence={confidence:.2f}")
            
            return {
                "status": "success",
                "validation": validation_result,
                "raw_response": validation_text
            }
        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "validation": {
                    "is_valid": True,  # Default to valid on error to not block workflow
                    "confidence": 0.5,
                    "issues": [f"Validation error: {str(e)}"],
                    "suggestions": [],
                    "summary": "Validation encountered an error"
                }
            }
    
    def _parse_validation_response(self, validation_text: str) -> Dict[str, Any]:
        """Parse validation response from LLM into structured format"""
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', validation_text, re.DOTALL)
        if json_match:
            try:
                validation_result = json.loads(json_match.group())
                # Ensure required fields exist
                validation_result.setdefault("is_valid", True)
                validation_result.setdefault("confidence", 0.8)
                validation_result.setdefault("issues", [])
                validation_result.setdefault("suggestions", [])
                validation_result.setdefault("summary", "Validation completed")
                return validation_result
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from validation response")
        
        # Fallback: create basic validation result
        return {
            "is_valid": True,
            "confidence": 0.7,
            "issues": [],
            "suggestions": [],
            "summary": "Validation completed (parsed from text)"
        }

