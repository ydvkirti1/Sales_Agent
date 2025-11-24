# Retail Insights Assistant

A GenAI-powered solution for analyzing large-scale retail sales data, generating automated business insights, and answering ad-hoc analytical questions in natural language.

## 🎯 Features

- **Multi-Agent System**: Powered by LangGraph with specialized agents for query resolution, data extraction, and validation
- **Natural Language Q&A**: Ask questions about your sales data in plain English
- **Automated Summarization**: Generate comprehensive business insights from your data

## 🏗️ Architecture

The system uses a multi-agent architecture with the following components:

1. **Query Resolution Agent**: Converts natural language queries to SQL
2. **Data Extraction Agent**: Executes queries and extracts relevant data
3. **Validation Agent**: Validates query results and ensures data quality
4. **Summarization Agent**: Generates business insights and summaries


## 📋 Prerequisites

- Python 3.10 or higher
- OpenAI API key (get from https://platform.openai.com/api-keys)
- Sales data in CSV, Excel, or JSON format

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd your path

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Get your API key from:** https://platform.openai.com/api-keys

### 3. Prepare Your Data

Place your sales data files in the `Sales Dataset/Sales Dataset/` directory. Supported formats:
- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)
- JSON files (`.json`)



The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Q&A Mode

1. Click on the **💬 Q&A Mode** tab
2. Enter your question in natural language, for example:
   - "Which category saw the highest sales in Q3?"
   - "What was the total revenue last month?"
   - "Show me top 10 products by sales"
3. Click **🔍 Ask** to process your query
4. View the answer, generated SQL query, and data visualizations

### Summarization Mode

1. Click on the **📝 Summarization Mode** tab
2. Click **📊 Generate Summary**
3. Review the comprehensive business insights generated from your data

### Data Explorer

1. Click on the **📊 Data Explorer** tab
2. Select a dataset from the dropdown
3. Explore the data with interactive tables and statistics

## 🔧 Configuration

Edit `config.py` to customize:

- OpenAI model selection (default: gpt-4o-mini)
- Data directory path
- Maximum rows for analysis
- Agent behavior settings

## 📁 Project Structure

```
testing/
├── agents/
│   ├── __init__.py
│   ├── query_resolution_agent.py    # Converts NL to SQL
│   ├── data_extraction_agent.py     # Executes queries
│   ├── validation_agent.py           # Validates results
│   └── summarization_agent.py       # Generates summaries
├── Sales Dataset/
│   └── Sales Dataset/                # Your data files here
├── langgraph_agent.py                # LangGraph orchestration
├── data_processor.py                 # Data processing layer
├── config.py                         # Configuration
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🧪 Example Queries

### Sales Analysis
- "What is the total sales by category?"
- "Which region had the highest growth in Q4?"
- "Show me products with sales above $10,000"

### Time-based Queries
- "Compare sales between Q3 and Q4"
- "What was the revenue trend over the last 6 months?"
- "Which month had the highest sales?"

### Product Analysis
- "List top 10 best-selling products"
- "Which products are underperforming?"
- "What is the average order value by product category?"

## 🔍 How It Works

1. **User Input**: You ask a question in natural language
2. **Query Resolution**: The Query Resolution Agent converts your question to SQL
3. **Data Extraction**: The Data Extraction Agent executes the SQL and retrieves data
4. **Validation**: The Validation Agent checks the results for accuracy
5. **Answer Generation**: The system generates a natural language answer with visualizations

## 🐛 Troubleshooting

### API Key Issues
- Ensure your API key is set in the `.env` file
- Verify API key has sufficient credits/quota in your OpenAI account
- Check that the API key is correct and active

### Data Loading Issues
- Ensure data files are in the correct directory
- Check file formats are supported (CSV, Excel, JSON)
- Verify file permissions

### Query Errors
- Check that your question references valid columns
- Ensure date formats are correct
- Review the generated SQL query in the expandable section

## 📊 Performance Considerations

### Current Limitations
- Best suited for datasets up to ~10GB in-memory
- Sequential query processing
- Single-machine processing

### Optimization Tips
- Use Parquet format for better compression
- Partition large datasets by date/region
- Enable query result caching
- Use appropriate data types

## 🔐 Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Implement authentication for production deployments
- Validate and sanitize user inputs

## 📝 Assumptions & Limitations

### Assumptions
- Data files are in a structured format (CSV, Excel, JSON)
- Date columns follow standard formats
- Numeric columns are properly formatted
- Dataset schemas are relatively stable

### Limitations
- Requires structured data (not free-form text)
- Best performance with tabular data
- LLM API costs scale with usage
- Some complex queries may require refinement

## 🛠️ Future Improvements

- [ ] Support for real-time data streaming
- [ ] Advanced visualization options
- [ ] Query history and favorites
- [ ] Export results to PDF/Excel
- [ ] Multi-user support with authentication
- [ ] Integration with cloud data warehouses
- [ ] Advanced caching strategies
- [ ] Cost tracking and optimization

## 📄 License

This project is provided as-is for evaluation purposes.

## 🤝 Contributing

This is an assignment project. For questions or improvements, please refer to the project requirements.

## 📧 Support

For issues or questions, please review the troubleshooting section or check the architecture documentation.

---

**Built with ❤️ using LangGraph, LangChain, and Streamlit**

