# Smart Chatbot

Welcome to the Smart Chatbot project—a full-stack AI-powered chatbot application built with FastAPI, MongoDB, and a simple frontend interface. It leverages OpenAI to generate responses, stores chat history in MongoDB, and runs in Docker containers for easy deployment.

---

## 🚀 Features

* **AI Chat**: Conversational interface powered by OpenAI GPT models.
* **Session Management**: Optional session ID to maintain conversation context.
* **Chat History**: Persist chat logs in MongoDB and retrieve via REST API.
* **Frontend SPA**: Lightweight single-page application served by FastAPI.
* **Dockerized**: All services (API, database, frontend) run in containers.
* **CORS Enabled**: Configurable origins for cross-domain requests.
* **Health Check**: `/health` endpoint to verify service status.

---

## 📁 Project Structure

```
smart-chatbot/
├── app/
│   ├── api/                 # FastAPI route definitions
│   │   └── chat.py          # Chat endpoints
│   ├── core/                # Configuration and utilities
│   │   ├── config.py        # Environment settings
│   │   └── openai_client.py # OpenAI client wrapper
│   ├── db/                  # Database connection
│   │   └── mongodb.py       # MongoDB startup/shutdown
│   ├── models/              # Pydantic schemas
│   │   └── schemas.py       # Request/response models
│   ├── services/            # Business logic
│   │   └── chat_service.py  # Chat processing & history
│   └── main.py              # App factory & routing
├── static/                  # Frontend single-page application
│   └── index.html           # HTML + JS chat interface
├── Dockerfile               # API service container
├── docker-compose.yml       # Orchestration for API + MongoDB
├── requirements.txt         # Python dependencies
├── .env.example             # Sample environment variables
└── README.md                # Project documentation
```

---

## ⚙️ Prerequisites

* Docker & Docker Compose
* OpenAI API Key (set in `.env`)
* MongoDB (via Docker Compose)

---

## 🔧 Setup & Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/smart-chatbot.git
   cd smart-chatbot
   ```

2. **Copy environment variables**:

   ```bash
   cp .env.example .env
   ```

   Fill in your OpenAI key and other settings in `.env`:

   ```ini
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-3.5-turbo
   MONGODB_URI=mongodb://mongodb:27017
   ```

3. **Build and start services**:

   ```bash
   docker-compose up --build
   ```

   * The API will be available at `http://localhost:8000/`.
   * MongoDB will run on port `27017`.

---

## 📦 Usage

### API Endpoints

* **Root**: `GET /` — Serves the frontend chat interface.
* **Docs**: `GET /docs` — OpenAPI documentation (Swagger UI).
* **Health**: `GET /health` — Returns `{ "status": "healthy" }`.
* **Chat**: `POST /api/v1/chat` — Send `{ "message": "..." }`, receive AI response.
* **History**: `GET /api/v1/chat/history/{session_id}` — Retrieve past messages for a session.

### Frontend Chat

* Navigate to `http://localhost:8000/` in your browser.
* Type your message in the input box and press **Send**.
* View AI responses and scroll through the conversation.

---

## 🌐 Environment Variables

| Variable         | Description                       | Example               |
| ---------------- | --------------------------------- | --------------------- |
| `OPENAI_API_KEY` | Your OpenAI API secret key        | `sk-xxx`              |
| `OPENAI_MODEL`   | Model name (e.g. `gpt-3.5-turbo`) | `gpt-4`               |
| `MONGODB_URI`    | MongoDB connection string         | `mongodb://...:27017` |
| `API_VERSION`    | API version prefix (e.g. `v1`)    | `v1`                  |
| `CORS_ORIGINS`   | Allowed CORS origins (JSON list)  | `["*"]`               |

---

## 🐳 Docker

* **Build**: `docker-compose build`
* **Run**:   `docker-compose up`
* **Logs**:  `docker-compose logs -f api`
* **Stop**:  `docker-compose down`

---

## 🤝 Contributing

Feel free to open issues or submit pull requests for bug fixes and enhancements. Please follow the [Contributing Guide](CONTRIBUTING.md) if available.

---

## 👤 Author

**Sina Hosseinzade** — [LinkedIn](https://www.linkedin.com/in/sina-hosseinzade20/)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
