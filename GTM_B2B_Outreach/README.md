# GTM B2B Outreach Agent

A multi-agent system for automating B2B outreach using GPT-5. This application finds target companies, identifies key contacts, researches insights, and generates personalized emails.

## 🏗️ Architecture

**Path #1: React (Vite + MUI) frontend + FastAPI backend**

```
project/
├─ backend/          # FastAPI + Agno agents
│  ├─ app.py         # FastAPI server with SSE
│  ├─ service.py     # Pipeline orchestration
│  ├─ agents.py      # Agent definitions
│  ├─ models.py      # Pydantic schemas
│  └─ requirements.txt
├─ frontend/         # React + TypeScript + MUI
│  ├─ src/
│  │  ├─ App.tsx     # Main application
│  │  ├─ api.ts      # API client
│  │  ├─ types.ts    # TypeScript types
│  │  └─ components/ # React components
│  ├─ package.json
│  └─ vite.config.ts
└─ docker-compose.yml
```

## 🚀 Quick Start

### Option 1: Local Development

#### Backend Setup

```bash
cd backend

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="sk-..."
export EXA_API_KEY="exa_..."
export CORS_ORIGINS="http://localhost:5173"

# Run the server
uvicorn app:app --reload --port 8000
```

Backend will be available at `http://localhost:8000`

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will be available at `http://localhost:5173`

### Option 2: Docker

```bash
# Set environment variables in your shell
export OPENAI_API_KEY="sk-..."
export EXA_API_KEY="exa_..."

# Run with docker-compose
docker-compose up
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

## 🔑 Environment Variables

Create a `.env` file in the `backend/` directory:

```
OPENAI_API_KEY=sk-...
EXA_API_KEY=exa_...
CORS_ORIGINS=http://localhost:5173
```

## 📋 Features

### Multi-Agent Pipeline

1. **Company Finder Agent** (GPT-5 + Exa)
   - Finds companies matching your target criteria
   - Returns company name, website, and fit reasoning

2. **Contact Finder Agent** (GPT-4o + Exa)
   - Identifies 2-3 decision makers per company
   - Prioritizes GTM, Sales, Partnerships roles
   - Infers emails using common patterns

3. **Research Agent** (GPT-5 + Exa)
   - Gathers insights from company websites
   - Searches Reddit discussions
   - Provides 2-4 personalization points

4. **Email Writer Agent** (GPT-5)
   - Generates personalized outreach emails
   - 4 style options: Professional, Casual, Cold, Consultative
   - 120-160 words with strong personalization

### Real-time Progress

- Server-Sent Events (SSE) for live progress updates
- Background task processing
- Progress bar with percentage tracking

### Modern UI

- Material UI components
- Responsive design
- Copy-to-clipboard for emails
- Clean, intuitive interface

## 🎯 Usage

1. **Fill in the form:**
   - Target companies (industry, size, region, tech stack)
   - Your product/service offering
   - Sender information (name, company)
   - Optional calendar link
   - Number of companies to target (1-10)
   - Email style preference

2. **Click "Start Outreach"**
   - Watch real-time progress
   - Pipeline runs through all 4 stages

3. **Review Results:**
   - Target companies with fit reasoning
   - Contact information (with inferred emails marked)
   - Research insights from web + Reddit
   - Personalized email drafts

4. **Copy emails** and use them in your outreach campaigns!

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Agno** - Agent framework with OpenAI integration
- **Exa** - Web search API
- **Pydantic** - Data validation
- **sse-starlette** - Server-Sent Events

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Material UI (MUI)** - Component library
- **EventSource API** - SSE client

## 📝 API Endpoints

### `POST /api/run`
Start a new outreach pipeline run

**Request:**
```json
{
  "target_desc": "B2B SaaS companies in fintech...",
  "offering_desc": "Marketing automation platform...",
  "sender_name": "John Doe",
  "sender_company": "Acme Corp",
  "calendar_link": "https://cal.com/...",
  "num_companies": 5,
  "email_style": "Professional"
}
```

**Response:**
```json
{
  "task_id": "uuid-here"
}
```

### `GET /api/progress/{task_id}`
Stream progress updates (SSE)

**Response stream:**
```
event: progress
data: 30

event: progress
data: 55
...
```

### `GET /api/result/{task_id}`
Fetch final results

**Response:**
```json
{
  "companies": [...],
  "contacts": [...],
  "research": [...],
  "emails": [...]
}
```

## 🔧 Development

### Adding New Features

**Backend:**
- Add new endpoints in `app.py`
- Extend agents in `agents.py`
- Update models in `models.py`

**Frontend:**
- Add components in `src/components/`
- Update types in `src/types.ts`
- Extend API client in `src/api.ts`

### Future Enhancements

- [ ] Regenerate individual emails
- [ ] Export results to CSV
- [ ] Email template customization
- [ ] Retry failed agent calls
- [ ] Caching for repeated searches
- [ ] User authentication
- [ ] Save/load campaigns
- [ ] Email tracking integration
- [ ] A/B testing different styles

## 📄 License

MIT

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 💡 Tips

- **API Costs:** Be mindful of API usage, especially with GPT-5
- **Rate Limits:** Exa API has rate limits; adjust `num_companies` accordingly
- **Email Verification:** Always verify inferred emails before sending
- **Personalization:** The more specific your inputs, the better the results
- **Testing:** Start with `num_companies: 2` to test the pipeline

## 🐛 Troubleshooting

**Backend won't start:**
- Check API keys are set correctly
- Ensure all dependencies are installed
- Verify Python version (3.11+ recommended)

**Frontend build errors:**
- Delete `node_modules/` and reinstall: `npm install`
- Clear Vite cache: `npm run build --force`

**No results returned:**
- Check backend logs for errors
- Verify API keys are valid
- Ensure internet connection for Exa searches

**CORS errors:**
- Verify `CORS_ORIGINS` environment variable matches frontend URL
- Check browser console for specific errors

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

Built with ❤️ using GPT-5, Agno, and Exa
