"""
Data Processing Layer for Retail Insights Assistant
Handles CSV, Excel, and JSON file parsing and querying
"""
import pandas as pd
import duckdb
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data ingestion, cleaning, and querying"""
    
    def __init__(self, data_dir: str = "Sales Dataset/Sales Dataset"):
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect()
        self.loaded_datasets = {}
        self.schema_info = {}
        
    def load_dataset(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load dataset from file or directory
        Returns metadata about loaded datasets
        """
        if file_path:
            return self._load_single_file(file_path)
        else:
            return self._load_all_files()
    
    def _load_single_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single data file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        dataset_name = path.stem.replace(" ", "_").lower()
        
        try:
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path, low_memory=False)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                df = pd.json_normalize(data)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
            
            # Register in DuckDB
            self.conn.register(dataset_name, df)
            self.loaded_datasets[dataset_name] = df
            self.schema_info[dataset_name] = self._get_schema_info(df)
            
            logger.info(f"Loaded {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
            
            return {
                "dataset_name": dataset_name,
                "rows": len(df),
                "columns": list(df.columns),
                "schema": self.schema_info[dataset_name],
                "sample_data": df.head(3).to_dict('records')
            }
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def _load_all_files(self) -> Dict[str, Any]:
        """Load all data files from the data directory"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        results = {}
        for file_path in self.data_dir.glob("*.*"):
            if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.json']:
                try:
                    info = self._load_single_file(str(file_path))
                    results[info["dataset_name"]] = info
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {str(e)}")
        
        return results
    
    def _get_schema_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract schema information from dataframe"""
        return {
            "columns": {
                col: {
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": df[col].dropna().head(5).tolist() if len(df[col].dropna()) > 0 else []
                }
                for col in df.columns
            },
            "total_rows": len(df),
            "date_columns": [col for col in df.columns if 'date' in col.lower()],
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
        }
    
    def get_schema_summary(self) -> str:
        """Get a human-readable summary of all loaded datasets"""
        summary = []
        for name, info in self.schema_info.items():
            summary.append(f"\nDataset: {name}")
            summary.append(f"  Rows: {info['total_rows']:,}")
            summary.append(f"  Columns: {', '.join(info['columns'].keys())}")
            if info['date_columns']:
                summary.append(f"  Date columns: {', '.join(info['date_columns'])}")
            if info['numeric_columns']:
                summary.append(f"  Numeric columns: {', '.join(info['numeric_columns'])}")
        return "\n".join(summary)
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query on loaded datasets"""
        try:
            result = self.conn.execute(sql_query).df()
            return result
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            raise
    
    def get_data_summary(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistical summary of the data"""
        summaries = {}
        
        if dataset_name:
            datasets_to_summarize = [dataset_name] if dataset_name in self.loaded_datasets else []
        else:
            datasets_to_summarize = list(self.loaded_datasets.keys())
        
        for name in datasets_to_summarize:
            df = self.loaded_datasets[name]
            summary = {
                "shape": df.shape,
                "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            summaries[name] = summary
        
        return summaries
    
    def close(self):
        """Close database connection"""
        self.conn.close()

