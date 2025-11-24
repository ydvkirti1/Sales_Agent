"""
Configuration file for Retail Insights Assistant
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration - OpenAI Only
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.1  # Lower temperature for more consistent results

# Data Configuration
DATA_DIR = "Sales Dataset/Sales Dataset"
CACHE_DIR = ".cache"
MAX_ROWS_FOR_ANALYSIS = 100000  # Limit for in-memory processing

# Agent Configuration
ENABLE_VALIDATION = True
MAX_QUERY_RETRIES = 3

# UI Configuration
PAGE_TITLE = "Retail Insights Assistant"
PAGE_ICON = "📊"

