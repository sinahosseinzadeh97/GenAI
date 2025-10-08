# 🏠 AI Home Renovation Planner

> 🇮🇷 [نسخه فارسی / Persian Version](README_FA.md)

Powerful AI-driven home renovation planning, design, and visualization using Gemini 2.5 Flash

## 📦 Project Components

This project includes **two user interfaces**:

### 1️⃣ Web Interface (ADK Web UI)
- Browser-based access
- Graphical user interface
- Ideal for development and testing

### 2️⃣ Telegram Bot 🤖
- Access via Telegram
- Easy and quick to use
- Perfect for end users
- **Bilingual: English & Persian**
- **Automatic image generation**

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
Google AI API Key
```

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup API Key
# Edit .env file and add your GOOGLE_API_KEY
```

---

## 💻 Using Web Interface

```bash
# Run ADK server
adk web

# Open in browser
http://127.0.0.1:8000/dev-ui/
```

Your agent **HomeRenovationPlanner** will appear in the dropdown menu.

---

## 🤖 Using Telegram Bot

### Quick Setup

```bash
# 1. Get token from @BotFather on Telegram

# 2. Add token to .env
echo "TELEGRAM_BOT_TOKEN=your_token_here" >> .env

# 3. Run bot
python3 telegram_bot.py
```

📖 **Full Guide:** [QUICKSTART_TELEGRAM.md](QUICKSTART_TELEGRAM.md)

---

## 🌟 Features

### 🎨 Analysis & Design
- AI-powered image analysis of current space
- Use inspiration photos
- Professional renovation design

### 💰 Estimation & Planning
- Accurate cost estimation
- Project timeline
- Materials and specifications list

### 🖼️ Visualization
- Generate photorealistic 8K images
- Edit and improve images
- Multiple design versions

### 🌐 Multilingual
- Full support for English & Persian
- Switch languages anytime with `/lang`

---

## 📁 Project Structure

```
AI_Home_Renovation/
├── home_renovation_agent/     # Main ADK agent
│   ├── agent.py              # Agent definitions
│   ├── tools.py              # Image generation tools
│   ├── agent.yaml            # Agent config
│   └── __init__.py
│
├── telegram_bot.py           # Telegram bot ⭐
├── run_bot.sh               # Run script
│
├── requirements.txt         # Dependencies
├── .env                    # API tokens & keys
│
└── README files:
    ├── README.md           # This file
    ├── QUICKSTART_TELEGRAM.md   # Quick start guide
    ├── README_TELEGRAM.md       # Complete Telegram guide
    └── TELEGRAM_SETUP.md        # Setup instructions
```

---

## 🎯 Usage Examples

### Example 1: Modern Kitchen Design

**Input:**
```
Kitchen photo + 
"I want to renovate this 150 sqft kitchen in modern style with white cabinets and $15k budget"
```

**Output:**
- Current space analysis
- Modern design plan
- Cost estimate: $14-16k
- Timeline: 2-3 months
- Photorealistic 8K image

### Example 2: Design with Inspiration

**Input:**
```
Current room photo + 
Inspiration photo + 
"I want exactly this style"
```

**Output:**
- Style matching with current space
- Custom design plan
- Final rendered image

---

## ⚙️ Advanced Settings

### Running Telegram Bot Permanently

#### With screen:
```bash
screen -S renovation_bot
python3 telegram_bot.py
# Ctrl+A then D to detach
```

#### With systemd:
```bash
# Create service file
sudo nano /etc/systemd/system/renovation-bot.service

# Content:
[Unit]
Description=AI Renovation Telegram Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/AI_Home_Renovation
ExecStart=/usr/bin/python3 telegram_bot.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable
sudo systemctl enable renovation-bot
sudo systemctl start renovation-bot
```

---

## 🔧 Troubleshooting

### ❌ "No agents found"

**Solution:** Ensure you're in the correct directory and `agent.yaml` exists.

```bash
cd /Users/apple/Desktop/AI_Home_Renovation
adk web
```

### ❌ "TELEGRAM_BOT_TOKEN not found"

**Solution:** Add token to `.env` file:

```bash
TELEGRAM_BOT_TOKEN=your_token_here
```

### ❌ "Unauthorized" (Telegram)

**Solution:** Token is incorrect. Get a new one from @BotFather.

---

## 📚 Documentation

- [Telegram Quick Start](QUICKSTART_TELEGRAM.md) - Start in 3 minutes
- [Complete Telegram Guide](README_TELEGRAM.md) - All details
- [Setup Instructions](TELEGRAM_SETUP.md) - Step by step
- [Main README](README.md) - This file

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   End User      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│  Web  │ │Telegram│
│  UI   │ │  Bot  │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
    ┌────▼────────────────┐
    │  HomeRenovation     │
    │  Planner (Root)     │
    └────┬────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────────┐
│ Info  │ │  Planning   │
│ Agent │ │  Pipeline   │
└───────┘ └──┬──────────┘
             │
        ┌────┴────┐
        │         │
    ┌───▼───┐ ┌──▼────┐
    │Visual │ │Design │
    │Assess │ │Planner│
    └───────┘ └───────┘
```

---

## 🔑 Security Notes

- ✅ Keep tokens in `.env` file
- ✅ Don't commit `.env` (already in .gitignore)
- ✅ Use personal API keys
- ✅ Each user gets isolated session

---

## 📊 Performance

- ⚡ Response in under 5 seconds (text analysis)
- 🎨 Image generation: 30-60 seconds
- 💾 Support images up to 20MB
- 🔄 Unlimited editing capability

---

## 🤝 Built With

This project uses:
- Google ADK (Agent Development Kit)
- Gemini 2.5 Flash (Multimodal AI)
- Python Telegram Bot

---

## 📞 Support

If you encounter issues:

1. **Check logs:**
   ```bash
   python3 telegram_bot.py  # Shows errors
   ```

2. **Test agent:**
   ```bash
   adk web  # Does agent work?
   ```

3. **Check .env:**
   ```bash
   cat .env  # Are tokens correct?
   ```

---

## 📝 License

This project is free for personal use.

---

## 🎉 Enjoy!

**Built with ❤️ using Google ADK + Gemini 2.5 Flash**

⭐ If you like this project, share it with friends!

---

### 🔗 Useful Links

- [Google AI Studio](https://aistudio.google.com/) - Get API Key
- [BotFather](https://t.me/botfather) - Create Telegram bot
- [ADK Docs](https://google.github.io/adk-docs/) - ADK Documentation