"""
FastAPI Backend API for Retail Insights Assistant
Provides REST API endpoints for the multi-agent system
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import logging
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from langgraph_agent import RetailInsightsOrchestrator
from config import DATA_DIR, OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Retail Insights Assistant API",
    description="GenAI-powered API for analyzing retail sales data",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
data_processor: Optional[DataProcessor] = None
orchestrator: Optional[RetailInsightsOrchestrator] = None
system_initialized = False


# Pydantic models for request/response
class InitializeRequest(BaseModel):
    """Request model for system initialization"""
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key (if not in .env)")


class InitializeResponse(BaseModel):
    """Response model for system initialization"""
    status: str
    message: str
    datasets_loaded: int
    schema_info: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for Q&A queries"""
    query: str = Field(..., description="Natural language question about sales data")
    mode: str = Field("qa", description="Mode: 'qa' for Q&A or 'summarization' for summary")


class QueryResponse(BaseModel):
    """Response model for queries"""
    status: str
    answer: Optional[str] = None
    summary: Optional[str] = None
    sql_query: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    system_initialized: bool
    message: str


class DatasetInfo(BaseModel):
    """Dataset information model"""
    name: str
    rows: int
    columns: List[str]
    shape: tuple


class DataExplorerResponse(BaseModel):
    """Response model for data explorer"""
    datasets: List[DatasetInfo]
    selected_dataset: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global data_processor, orchestrator, system_initialized
    
    try:
        if OPENAI_API_KEY:
            logger.info("Initializing system on startup...")
            data_processor = DataProcessor(DATA_DIR)
            datasets = data_processor.load_dataset()
            orchestrator = RetailInsightsOrchestrator(data_processor)
            system_initialized = True
            logger.info(f"System initialized with {len(datasets)} dataset(s)")
        else:
            logger.warning("OpenAI API key not found. System will need manual initialization.")
    except Exception as e:
        logger.error(f"Failed to initialize on startup: {str(e)}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "system_initialized": system_initialized,
        "message": "Retail Insights Assistant API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if system_initialized else "not_initialized",
        "system_initialized": system_initialized,
        "message": "System is ready" if system_initialized else "System needs initialization"
    }


@app.post("/initialize", response_model=InitializeResponse)
async def initialize_system(request: InitializeRequest):
    """
    Initialize the system with data processor and orchestrator
    
    Args:
        request: InitializeRequest with optional API key
        
    Returns:
        InitializeResponse with initialization status
    """
    global data_processor, orchestrator, system_initialized
    
    try:
        # Set API key if provided
        if request.openai_api_key:
            os.environ["OPENAI_API_KEY"] = request.openai_api_key
            logger.info("OpenAI API key set from request")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key is required. Provide it in request or set OPENAI_API_KEY environment variable."
            )
        
        # Initialize components
        data_processor = DataProcessor(DATA_DIR)
        datasets = data_processor.load_dataset()
        orchestrator = RetailInsightsOrchestrator(data_processor)
        system_initialized = True
        
        schema_info = data_processor.get_schema_summary()
        
        logger.info(f"System initialized successfully with {len(datasets)} dataset(s)")
        
        return {
            "status": "success",
            "message": f"System initialized with {len(datasets)} dataset(s)",
            "datasets_loaded": len(datasets),
            "schema_info": schema_info
        }
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        system_initialized = False
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query through the multi-agent system
    
    Args:
        request: QueryRequest with query and mode
        
    Returns:
        QueryResponse with answer, data, and metadata
    """
    global orchestrator, system_initialized
    
    if not system_initialized or not orchestrator:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please call /initialize first."
        )
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get schema info
        schema_info = data_processor.get_schema_summary()
        
        # Process query
        result = orchestrator.process_query(
            user_query=request.query,
            mode=request.mode,
            schema_info=schema_info
        )
        
        # Format response
        response_data = {
            "status": "success" if not result.get("error") else "error",
            "error": result.get("error")
        }
        
        if request.mode == "summarization":
            response_data["summary"] = result.get("summary", "")
        else:
            response_data["answer"] = result.get("final_answer", "")
            response_data["sql_query"] = result.get("sql_query", "")
            
            # Include data if available
            extraction_result = result.get("extraction_result", {})
            if extraction_result.get("status") == "success":
                response_data["data"] = extraction_result.get("data", {})
            
            # Include validation if available
            validation_result = result.get("validation_result", {})
            if validation_result.get("status") == "success":
                response_data["validation"] = validation_result.get("validation", {})
        
        return QueryResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/datasets", response_model=DataExplorerResponse)
async def get_datasets():
    """
    Get list of available datasets
    
    Returns:
        DataExplorerResponse with dataset information
    """
    global data_processor, system_initialized
    
    if not system_initialized or not data_processor:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please call /initialize first."
        )
    
    try:
        datasets_info = []
        for name, df in data_processor.loaded_datasets.items():
            datasets_info.append({
                "name": name,
                "rows": len(df),
                "columns": list(df.columns),
                "shape": df.shape
            })
        
        return {
            "datasets": datasets_info
        }
    except Exception as e:
        logger.error(f"Error getting datasets: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get datasets: {str(e)}")


@app.get("/datasets/{dataset_name}")
async def get_dataset_data(dataset_name: str, limit: int = 100):
    """
    Get data from a specific dataset
    
    Args:
        dataset_name: Name of the dataset
        limit: Maximum number of rows to return
        
    Returns:
        Dataset data as JSON
    """
    global data_processor, system_initialized
    
    if not system_initialized or not data_processor:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please call /initialize first."
        )
    
    if dataset_name not in data_processor.loaded_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    
    try:
        df = data_processor.loaded_datasets[dataset_name]
        limited_df = df.head(limit)
        
        return {
            "dataset_name": dataset_name,
            "total_rows": len(df),
            "returned_rows": len(limited_df),
            "columns": list(df.columns),
            "data": limited_df.to_dict('records'),
            "statistics": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {}
        }
    except Exception as e:
        logger.error(f"Error getting dataset data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get dataset data: {str(e)}")


@app.get("/schema")
async def get_schema():
    """
    Get schema information for all datasets
    
    Returns:
        Schema information as JSON
    """
    global data_processor, system_initialized
    
    if not system_initialized or not data_processor:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please call /initialize first."
        )
    
    try:
        schema_info = data_processor.get_schema_summary()
        return {
            "schema_info": schema_info,
            "datasets": list(data_processor.loaded_datasets.keys())
        }
    except Exception as e:
        logger.error(f"Error getting schema: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

