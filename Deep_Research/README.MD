# 🔬 Deep Research

An AI-powered intelligent research system that generates comprehensive, structured reports using multiple specialized agents.

## 📋 Overview

Deep Research is a powerful tool for automated deep research that leverages:
- **Intelligent Agents** (Planner, Search, Writer)
- **Web Search** (with SerpAPI support and web scraping)
- **Professional Report Generation** in Markdown format
- **Automated Email Delivery** (optional with SendGrid)

## 🚀 Key Features

- **Multi-Agent Architecture**: Each agent has a specific role
- **Smart Search**: Decomposes queries into multiple optimized searches
- **Concurrent Processing**: Parallel search execution for faster results
- **Gradio UI**: Simple and user-friendly web interface
- **Structured Reports**: With table of contents, logical sections, and references
- **Internal Caching**: Prevents duplicate searches

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd deep-research
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:

```env
# OpenAI API (Required)
OPENAI_API_KEY=sk-your-openai-api-key

# SerpAPI (Optional - for better search)
SERPAPI_API_KEY=your-serpapi-key

# SendGrid (Optional - for email delivery)
SENDGRID_API_KEY=your-sendgrid-api-key
SENDGRID_FROM_EMAIL=sender@example.com
SENDGRID_TO_EMAIL=recipient@example.com

# Advanced Settings
RESEARCH_MAX_SEARCHES=5
RESEARCH_CONCURRENCY=3
LOG_LEVEL=INFO
```

## 🎯 Usage

### Method 1: Gradio Web Interface
```bash
python -m deep_research.app
```
Open your browser and navigate to the displayed URL (typically `http://localhost:7860`)

### Method 2: Programmatic Usage
```python
import asyncio
from deep_research.research_manager import ResearchManager

async def main():
    manager = ResearchManager()
    query = "Latest developments in quantum computing"
    
    async for chunk in manager.run(query):
        print(chunk, end='', flush=True)

asyncio.run(main())
```

## 🏗️ System Architecture

### Agents

1. **Planner Agent** (`planner_agent.py`)
   - Input: User query
   - Output: 3-6 optimized search queries
   - Role: Decompose topic into different aspects

2. **Search Agent** (`search_agent.py`)
   - Input: Single search query
   - Output: 300-word summary of results
   - Role: Web search and key information extraction

3. **Writer Agent** (`writer_agent.py`)
   - Input: Search summaries
   - Output: Comprehensive 1000+ word report
   - Role: Generate structured report with intro, sections, and conclusion

4. **Email Agent** (`email_agent.py`)
   - Input: Final report
   - Output: Professional email
   - Role: Format and send report via email

### Workflow

```
User → Query
    ↓
Planner Agent → [Search1, Search2, ..., SearchN]
    ↓
Search Agents → [Summary1, Summary2, ..., SummaryN] (Parallel)
    ↓
Writer Agent → Final Markdown Report
    ↓
Email Agent → Send Email (Optional)
```

## 📁 Project Structure

```
deep_research/
├── __init__.py          # Loads .env configuration
├── agents.py            # Base Agent classes and tools
├── app.py              # Gradio UI interface
├── research_manager.py  # Research pipeline management
├── planner_agent.py    # Planning agent
├── search_agent.py     # Search agent
├── writer_agent.py     # Writer agent
└── email_agent.py      # Email agent
```

## ⚙️ Advanced Configuration

### Concurrency Control
```env
RESEARCH_CONCURRENCY=3  # Number of concurrent searches
RESEARCH_MAX_SEARCHES=5 # Maximum number of searches
```

### Model Settings
Each agent uses independent model settings:
```python
ModelSettings(
    model="gpt-4o-mini",      # or gpt-4, gpt-3.5-turbo
    temperature=0.7,          # Creativity (0-1)
    max_tokens=2000,          # Maximum output tokens
    top_p=0.9                 # Nucleus sampling
)
```

### Search Sources
- **With SerpAPI**: More accurate Google results
- **Without SerpAPI**: Direct web scraping (slower)

## 🐛 Troubleshooting

### API Key Error
```
Error: OpenAI API key not found
```
**Solution**: Check your `.env` file and ensure `OPENAI_API_KEY` is set.

### Rate Limit Error
```
Rate limit exceeded
```
**Solution**: 
- Reduce `RESEARCH_CONCURRENCY`
- Add delays between requests
- Use cheaper models

### Web Search Issues
If searches are failing:
1. Check internet connection
2. Set SerpAPI key for better results
3. Verify User-Agent settings

## 📊 Sample Output

A typical report includes:
- **Title**: Report on [Topic]
- **Table of Contents**: Numbered list of sections
- **Introduction**: 150-200 word overview
- **Main Sections**: ~200 words each
- **Conclusion**: 120-150 word summary
- **References**: Numbered list of sources used
- **Follow-up Questions**: 3 questions for further research

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Gradio for the excellent UI framework
- Python community for amazing libraries

---

Built with SinaMohammadHosseinzadeh