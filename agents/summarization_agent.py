"""
Summarization Agent
Generates human-readable business insights summaries from sales data
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional
import logging
from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationAgent:
    """
    Agent that generates business insights summaries.
    
    This agent analyzes sales data and creates executive-friendly
    summaries with key trends, patterns, and actionable insights.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize Summarization Agent
        
        Args:
            llm: Optional pre-initialized LLM. If None, creates new one.
        """
        self.llm = llm if llm else self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize OpenAI LLM with slightly higher temperature for creativity"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE + 0.2,  # Slightly higher for more creative summaries
            api_key=OPENAI_API_KEY
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for summarization"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert retail business analyst. Your task is to analyze sales data and generate 
concise, actionable business insights in natural language.

Guidelines:
- Write in clear, executive-friendly language
- Highlight key trends, patterns, and anomalies
- Include specific numbers and percentages
- Identify top performers and underperformers
- Mention time periods and regions when relevant
- Keep the summary concise (2-4 paragraphs)
- Use bullet points for key metrics
- Focus on actionable insights

Format your response as a well-structured business summary."""),
            ("human", """Analyze the following sales data and generate a comprehensive business summary:

Data Summary:
{data_summary}

Schema Information:
{schema_info}

Generate a business insights summary:""")
        ])
    
    def generate_summary(self, data_summary: Dict[str, Any], 
                        schema_info: str) -> Dict[str, Any]:
        """
        Generate business insights summary
        
        Args:
            data_summary: Summary statistics from data
            schema_info: Schema information about datasets
            
        Returns:
            Dictionary with generated summary:
            {
                "status": "success" | "error",
                "summary": str,
                "data_summary": dict,
                "error": str (if status is error)
            }
        """
        if not data_summary:
            logger.warning("No data summary provided")
            return {
                "status": "error",
                "error": "No data summary provided",
                "summary": "Unable to generate summary: no data available."
            }
        
        try:
            logger.info("Generating business insights summary...")
            
            # Format data summary for prompt
            summary_text = self._format_data_summary(data_summary)
            
            if not summary_text or len(summary_text.strip()) < 10:
                logger.warning("Formatted data summary is too short")
                return {
                    "status": "error",
                    "error": "Insufficient data for summary generation",
                    "summary": "Unable to generate summary: insufficient data."
                }
            
            prompt = self.prompt_template.format_messages(
                data_summary=summary_text,
                schema_info=schema_info or "No schema information available"
            )
            
            response = self.llm.invoke(prompt)
            summary = response.content.strip()
            
            if not summary or len(summary) < 50:
                logger.warning("Generated summary is too short")
                summary = "Summary generated but appears incomplete. Please check the data."
            
            logger.info(f"Successfully generated summary ({len(summary)} chars)")
            
            return {
                "status": "success",
                "summary": summary,
                "data_summary": data_summary
            }
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "summary": "Unable to generate summary due to an error."
            }
    
    def _format_data_summary(self, data_summary: Dict[str, Any]) -> str:
        """Format data summary into readable text"""
        lines = []
        for dataset_name, summary in data_summary.items():
            lines.append(f"\nDataset: {dataset_name}")
            lines.append(f"  Shape: {summary.get('shape', 'N/A')}")
            
            if summary.get('numeric_summary'):
                lines.append("  Numeric Summary:")
                for col, stats in summary['numeric_summary'].items():
                    if isinstance(stats, dict):
                        lines.append(f"    {col}: {stats}")
            
            if summary.get('null_counts'):
                null_info = {k: v for k, v in summary['null_counts'].items() if v > 0}
                if null_info:
                    lines.append(f"  Missing Values: {null_info}")
        
        return "\n".join(lines)

