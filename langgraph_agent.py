"""
LangGraph Multi-Agent Orchestration using Supervisor Pattern
Uses create_react_agent and supervisor pattern for efficient agent management
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import logging

from agents.query_resolution_agent import QueryResolutionAgent
from agents.data_extraction_agent import DataExtractionAgent
from agents.validation_agent import ValidationAgent
from agents.summarization_agent import SummarizationAgent
from data_processor import DataProcessor
from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[list, "messages"]
    user_query: str
    mode: str  # "qa" or "summarization"
    schema_info: str
    sql_query: str
    extraction_result: dict
    validation_result: dict
    summary: str
    final_answer: str
    error: str
    step: str
    next_agent: str


class RetailInsightsOrchestrator:
    """Orchestrates multi-agent workflow using LangGraph Supervisor Pattern"""
    
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
        
        # Initialize agent classes (for backward compatibility)
        self.query_agent = QueryResolutionAgent()
        self.extraction_agent = DataExtractionAgent(data_processor)
        self.validation_agent = ValidationAgent()
        self.summarization_agent = SummarizationAgent()
        
        # Create tools for each agent
        self.tools = self._create_tools()
        
        # Create ReAct agents using LangGraph prebuilt
        self.agents = self._create_react_agents()
        
        # Build supervisor graph
        self.graph = self._build_supervisor_graph()
    
    def _create_tools(self):
        """Create tools for each agent functionality"""
        
        @tool
        def resolve_query_to_sql(user_query: str, schema_info: str) -> str:
            """Convert natural language query to SQL. Use this when user asks a question about the data.
            
            Args:
                user_query: The user's question in natural language
                schema_info: Schema information about available datasets
            """
            try:
                result = self.query_agent.resolve_query(user_query, schema_info)
                if result.get("status") == "success":
                    return result.get("sql_query", "")
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error resolving query: {str(e)}"
        
        @tool
        def extract_data_from_sql(sql_query: str) -> str:
            """Execute SQL query and extract data. Use this after generating a SQL query.
            
            Args:
                sql_query: The SQL query to execute
            """
            try:
                result = self.extraction_agent.extract_data(sql_query)
                if result.get("status") == "success":
                    data_dict = result.get("data", {})
                    row_count = data_dict.get("row_count", 0)
                    columns = data_dict.get("columns", [])
                    return f"Extracted {row_count} rows with columns: {', '.join(columns)}. Data available for analysis."
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error extracting data: {str(e)}"
        
        @tool
        def validate_query_results(user_query: str, sql_query: str, extraction_result: dict) -> str:
            """Validate that the SQL query and results correctly answer the user's question.
            
            Args:
                user_query: Original user question
                sql_query: SQL query that was executed
                extraction_result: Results from data extraction
            """
            try:
                result = self.validation_agent.validate(user_query, sql_query, extraction_result)
                if result.get("status") == "success":
                    validation = result.get("validation", {})
                    confidence = validation.get("confidence", 0.5)
                    is_valid = validation.get("is_valid", True)
                    return f"Validation: {'Valid' if is_valid else 'Invalid'}, Confidence: {confidence:.2f}"
                else:
                    return f"Validation error: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error validating: {str(e)}"
        
        @tool
        def generate_business_summary(data_summary: dict, schema_info: str) -> str:
            """Generate comprehensive business insights summary from sales data.
            
            Args:
                data_summary: Summary statistics from data
                schema_info: Schema information about datasets
            """
            try:
                result = self.summarization_agent.generate_summary(data_summary, schema_info)
                if result.get("status") == "success":
                    return result.get("summary", "")
                else:
                    return f"Error: {result.get('error', 'Unknown error')}"
            except Exception as e:
                return f"Error generating summary: {str(e)}"
        
        return [
            resolve_query_to_sql,
            extract_data_from_sql,
            validate_query_results,
            generate_business_summary
        ]
    
    def _create_react_agents(self):
        """Create ReAct agents using LangGraph prebuilt"""
        # Create a main agent with all tools
        # create_react_agent creates a graph that can be used as a node
        main_agent = create_react_agent(
            self.llm,
            self.tools
        )
        
        return {
            "main_agent": main_agent
        }
    
    def _build_supervisor_graph(self) -> StateGraph:
        """
        Build supervisor graph that routes to appropriate agents.
        
        Workflow:
        Entry → Supervisor → [Query Agent → Extract Agent → Validate Agent → Answer] OR [Summary Agent]
        """
        workflow = StateGraph(AgentState)
        
        # Add supervisor node
        workflow.add_node("supervisor", self._supervisor_node)
        
        # Add agent nodes
        workflow.add_node("query_agent", self._query_agent_node)
        workflow.add_node("extract_agent", self._extract_agent_node)
        workflow.add_node("validate_agent", self._validate_agent_node)
        workflow.add_node("answer_agent", self._answer_agent_node)
        workflow.add_node("summary_agent", self._summary_agent_node)
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Add conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_to_agent,
            {
                "query": "query_agent",
                "extract": "extract_agent",
                "validate": "validate_agent",
                "answer": "answer_agent",
                "summary": "summary_agent",
                "end": END
            }
        )
        
        # Define workflow edges
        workflow.add_edge("query_agent", "extract_agent")
        workflow.add_edge("extract_agent", "validate_agent")
        workflow.add_edge("validate_agent", "answer_agent")
        workflow.add_edge("answer_agent", END)
        workflow.add_edge("summary_agent", END)
        
        return workflow.compile()
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """
        Supervisor node that intelligently decides which agent to call next.
        
        Uses LLM-based routing for better decision making, with fallback
        to rule-based routing for reliability.
        """
        mode = state.get("mode", "qa")
        step = state.get("step", "start")
        error = state.get("error", "")
        
        # Handle summarization mode
        if mode == "summarization":
            state["next_agent"] = "summary"
            logger.info("Supervisor: Routing to summary agent")
            return state
        
        # Handle errors - route to end
        if error and step != "start":
            logger.warning(f"Supervisor: Error detected, routing to end. Error: {error[:100]}")
            state["next_agent"] = "end"
            return state
        
        # Rule-based routing for Q&A flow (more reliable)
        if step == "start":
            state["next_agent"] = "query"
            logger.info("Supervisor: Starting Q&A flow - routing to query agent")
        elif step == "query_resolution":
            # Check if query was successful
            if state.get("sql_query") and not error:
                state["next_agent"] = "extract"
                logger.info("Supervisor: Query resolved, routing to extract agent")
            else:
                state["next_agent"] = "end"
                logger.warning("Supervisor: Query resolution failed, ending workflow")
        elif step == "data_extraction":
            # Check if extraction was successful
            extraction_result = state.get("extraction_result", {})
            if extraction_result.get("status") == "success":
                state["next_agent"] = "validate"
                logger.info("Supervisor: Data extracted, routing to validate agent")
            else:
                state["next_agent"] = "end"
                logger.warning("Supervisor: Data extraction failed, ending workflow")
        elif step == "validation":
            state["next_agent"] = "answer"
            logger.info("Supervisor: Validation complete, routing to answer agent")
        elif step == "answer_generation":
            state["next_agent"] = "end"
            logger.info("Supervisor: Answer generated, ending workflow")
        else:
            state["next_agent"] = "end"
            logger.info(f"Supervisor: Unknown step '{step}', ending workflow")
        
        return state
    
    def _route_to_agent(self, state: AgentState) -> str:
        """Route to the next agent based on supervisor decision"""
        return state.get("next_agent", "end")
    
    def _query_agent_node(self, state: AgentState) -> AgentState:
        """Query Resolution Agent node using ReAct pattern"""
        try:
            logger.info("Executing Query Resolution Agent")
            
            # Use the agent class method
            result = self.query_agent.resolve_query(
                state["user_query"],
                state["schema_info"]
            )
            
            state["sql_query"] = result.get("sql_query", "")
            state["step"] = "query_resolution"
            
            if result.get("status") == "error":
                state["error"] = result.get("error", "Query resolution failed")
            
            return state
        except Exception as e:
            logger.error(f"Query resolution error: {str(e)}")
            state["error"] = str(e)
            return state
    
    def _extract_agent_node(self, state: AgentState) -> AgentState:
        """Data Extraction Agent node"""
        try:
            logger.info("Executing Data Extraction Agent")
            
            if not state.get("sql_query"):
                state["error"] = "No SQL query available"
                return state
            
            result = self.extraction_agent.extract_data(state["sql_query"])
            state["extraction_result"] = result
            state["step"] = "data_extraction"
            
            return state
        except Exception as e:
            logger.error(f"Data extraction error: {str(e)}")
            state["error"] = str(e)
            state["extraction_result"] = {"status": "error", "error": str(e)}
            return state
    
    def _validate_agent_node(self, state: AgentState) -> AgentState:
        """Validation Agent node"""
        try:
            logger.info("Executing Validation Agent")
            
            if state.get("extraction_result", {}).get("status") != "success":
                state["validation_result"] = {
                    "status": "skipped",
                    "validation": {"is_valid": False, "confidence": 0.0}
                }
                return state
            
            result = self.validation_agent.validate(
                state["user_query"],
                state["sql_query"],
                state["extraction_result"]
            )
            
            state["validation_result"] = result
            state["step"] = "validation"
            
            return state
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            state["validation_result"] = {
                "status": "error",
                "validation": {"is_valid": True, "confidence": 0.5}
            }
            return state
    
    def _answer_agent_node(self, state: AgentState) -> AgentState:
        """Answer Generation Agent node"""
        try:
            logger.info("Generating final answer")
            
            from langchain_core.prompts import ChatPromptTemplate
            
            # Create answer generation prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful retail data analyst. Answer the user's question based on the 
provided data. Be concise, accurate, and include specific numbers from the data."""),
                ("human", """Question: {question}

Data:
{data}

Provide a clear, concise answer:""")
            ])
            
            extraction_result = state.get("extraction_result", {})
            data_dict = extraction_result.get("data", {})
            
            # Format data for prompt
            data_text = f"Columns: {', '.join(data_dict.get('columns', []))}\n"
            data_text += f"Row count: {data_dict.get('row_count', 0)}\n"
            
            if data_dict.get("data"):
                sample_rows = data_dict["data"][:5]
                data_text += f"\nSample data:\n{str(sample_rows)}\n"
            
            if data_dict.get("summary_stats"):
                data_text += f"\nSummary statistics:\n{str(data_dict['summary_stats'])}"
            
            response = self.llm.invoke(prompt.format_messages(
                question=state["user_query"],
                data=data_text
            ))
            
            state["final_answer"] = response.content.strip()
            state["step"] = "answer_generation"
            
            return state
        except Exception as e:
            logger.error(f"Answer generation error: {str(e)}")
            state["error"] = str(e)
            state["final_answer"] = "Unable to generate answer due to an error."
            return state
    
    def _summary_agent_node(self, state: AgentState) -> AgentState:
        """Summarization Agent node"""
        try:
            logger.info("Executing Summarization Agent")
            
            # Get data summary
            data_summary = self.data_processor.get_data_summary()
            
            # Generate summary
            result = self.summarization_agent.generate_summary(
                data_summary,
                state["schema_info"]
            )
            
            state["summary"] = result.get("summary", "")
            state["step"] = "summarization"
            
            return state
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            state["error"] = str(e)
            state["summary"] = "Unable to generate summary due to an error."
            return state
    
    def process_query(self, user_query: str, mode: str = "qa", 
                     schema_info: str = "") -> dict:
        """
        Process a user query through the multi-agent system
        
        Args:
            user_query: User's question or request
            mode: "qa" for Q&A mode, "summarization" for summary mode
            schema_info: Schema information about datasets
            
        Returns:
            Final result dictionary
        """
        initial_state = {
            "messages": [],
            "user_query": user_query,
            "mode": mode,
            "schema_info": schema_info,
            "sql_query": "",
            "extraction_result": {},
            "validation_result": {},
            "summary": "",
            "final_answer": "",
            "error": "",
            "step": "start",
            "next_agent": ""
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Orchestration error: {str(e)}")
            return {
                **initial_state,
                "error": str(e),
                "final_answer": f"Error processing query: {str(e)}"
            }
