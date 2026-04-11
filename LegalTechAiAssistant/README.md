# 🏛️ LegalTech AI Assistant

A comprehensive AI-powered legal document analysis system built with React, FastAPI, and OpenAI GPT-4. Upload legal documents and ask questions to receive intelligent responses with proper source citations and relevant legal references.

## 🚀 Features

- **📄 Document Upload & Processing**: Upload PDF and text legal documents
- **🤖 AI-Powered Analysis**: GPT-4 powered intelligent responses in English
- **📚 Source Citations**: Automatic citation of relevant document sections
- **⚖️ Legal Database Integration**: GDPR and other legal references
- **🌐 Modern Web Interface**: React-based responsive frontend
- **🔍 Vector Search**: Advanced semantic search through legal documents
- **🐳 Docker Deployment**: Complete containerized solution

## 🏃‍♂️ Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key

### Installation

1. **Clone and setup**
```bash
git clone https://github.com/sinahosseinzadeh97/GenAI.git
cd GenAI/LegalTechAiAssistant
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

2. **Start the application**
```bash
docker-compose up -d
```

3. **Access the application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📖 Usage

### 1. Upload Documents
- Navigate to http://localhost:5173
- Click "Choose File" and select your legal document
- Click "Upload Document" and wait for processing

### 2. Ask Questions
- Enter your legal question in English
- Click "Ask Question" to get AI analysis
- Review responses with source citations

### Example Questions
- "What are the confidentiality obligations?"
- "What data protection regulations apply?"
- "What is the termination notice period?"
- "What happens if the agreement is breached?"

## 🛠️ Technology Stack

- **Backend**: FastAPI + OpenAI GPT-4 + PostgreSQL + pgvector
- **Frontend**: React 18 + TypeScript + Tailwind CSS + Vite
- **Infrastructure**: Docker + Docker Compose

## 📁 API Endpoints

- `POST /documents` — Upload document (PDF/TXT) → `{document_id, workflow_id}`
- `POST /query` — Ask questions → RAG `answer`, `sources`, `laws`
- `GET /workflows/{id}` — Retrieve workflow status

## 🔧 Development

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend  
cd frontend && npm install && npm run dev
```

## 🧪 Testing

Use the included `comprehensive_employment_contract.txt` for testing:
- Upload the contract file
- Ask questions about confidentiality, compensation, termination, etc.
- Verify English responses with proper citations

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | ✅ Yes |
| `VITE_API_URL` | Frontend API endpoint | No (auto-configured) |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │   PostgreSQL    │
│   (Port 5173)   │◄──►│   (Port 8000)   │◄──►│   (Port 5433)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   OpenAI GPT-4  │
                       │   API Service   │
                       └─────────────────┘
```

## 📁 Project Structure

```
LegalTechAiAssistant/
├── backend/
│   ├── app/
│   │   ├── core/          # Configuration
│   │   ├── models/        # Database models
│   │   ├── routers/       # API endpoints
│   │   ├── services/      # Business logic
│   │   └── main.py        # FastAPI app
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── types/         # TypeScript definitions
│   │   └── App.tsx        # Main application
│   └── Dockerfile
├── docker-compose.yml
├── comprehensive_employment_contract.txt  # Test document
└── README.md
```

## 🚀 Production Deployment

```bash
# Production deployment
docker-compose up -d --build
```

### Production Checklist
- ✅ Set strong database passwords
- ✅ Configure proper CORS settings  
- ✅ Set up SSL certificates
- ✅ Configure backup strategies
- ✅ Monitor application logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- Create GitHub issues for bugs/features
- Check API docs at http://localhost:8000/docs
- Review logs: `docker-compose logs`

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- FastAPI team for the excellent framework
- React team for the frontend framework
- PostgreSQL and pgvector for vector search

---

**Status**: ✅ **Production Ready** — Complete legal document analysis with English AI responses and full citation support.
**https://www.linkedin.com/in/sina-hosseinzade20/**
**Built by SinaMohammadHosseinzadeh**
