"""
Data Extraction Agent
Executes SQL queries and extracts relevant data from datasets using DuckDB
"""
from data_processor import DataProcessor
from typing import Dict, Any, Optional
import pandas as pd
import logging
from config import MAX_ROWS_FOR_ANALYSIS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractionAgent:
    """
    Agent that extracts data based on SQL queries.
    
    This agent executes SQL queries on the data processor and
    returns structured data with summary statistics.
    """
    
    def __init__(self, data_processor: DataProcessor):
        """
        Initialize Data Extraction Agent
        
        Args:
            data_processor: DataProcessor instance for executing queries
        """
        if not data_processor:
            raise ValueError("DataProcessor is required")
        self.data_processor = data_processor
    
    def extract_data(self, sql_query: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute SQL query and extract data
        
        Args:
            sql_query: SQL query to execute
            max_rows: Maximum number of rows to return (defaults to config value)
            
        Returns:
            Dictionary with extracted data and metadata:
            {
                "status": "success" | "error",
                "data": {
                    "data": list of records,
                    "columns": list of column names,
                    "row_count": int,
                    "summary_stats": dict
                },
                "sql_query": str,
                "error": str (if status is error)
            }
        """
        if not sql_query or not sql_query.strip():
            logger.warning("Empty SQL query provided")
            return {
                "status": "error",
                "error": "Empty SQL query provided",
                "sql_query": sql_query,
                "data": None
            }
        
        max_rows = max_rows or MAX_ROWS_FOR_ANALYSIS
        
        try:
            logger.info(f"Executing SQL query: {sql_query[:100]}...")
            
            # Execute query
            result_df = self.data_processor.execute_query(sql_query)
            
            if result_df is None or result_df.empty:
                logger.warning("Query returned empty result")
                return {
                    "status": "success",
                    "data": {
                        "data": [],
                        "columns": [],
                        "row_count": 0,
                        "summary_stats": {}
                    },
                    "sql_query": sql_query
                }
            
            original_row_count = len(result_df)
            
            # Limit rows if necessary
            if original_row_count > max_rows:
                result_df = result_df.head(max_rows)
                logger.warning(f"Result truncated from {original_row_count} to {max_rows} rows")
            
            # Convert to dictionary format
            data_dict = {
                "data": result_df.to_dict('records'),
                "columns": list(result_df.columns),
                "row_count": len(result_df),
                "summary_stats": self._calculate_summary_stats(result_df),
                "original_row_count": original_row_count,
                "truncated": original_row_count > max_rows
            }
            
            logger.info(f"Successfully extracted {len(result_df)} rows with {len(result_df.columns)} columns")
            
            return {
                "status": "success",
                "data": data_dict,
                "sql_query": sql_query
            }
        except Exception as e:
            logger.error(f"Data extraction error: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "sql_query": sql_query,
                "data": None
            }
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for numeric columns"""
        stats = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            stats[col] = {
                "sum": float(df[col].sum()) if len(df) > 0 else 0,
                "mean": float(df[col].mean()) if len(df) > 0 else 0,
                "min": float(df[col].min()) if len(df) > 0 else 0,
                "max": float(df[col].max()) if len(df) > 0 else 0,
                "count": int(df[col].count())
            }
        
        return stats

