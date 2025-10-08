#!/bin/bash

# Script to run the Telegram bot

echo "🤖 Starting AI Home Renovation Telegram Bot..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your API keys."
    echo "See .env.example for reference."
    exit 1
fi

# Check if TELEGRAM_BOT_TOKEN is set
if ! grep -q "TELEGRAM_BOT_TOKEN=" .env || grep -q "TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE" .env || grep -q "TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here" .env; then
    echo "❌ Error: TELEGRAM_BOT_TOKEN not configured!"
    echo ""
    echo "Please:"
    echo "1. Go to @BotFather on Telegram"
    echo "2. Create a new bot with /newbot"
    echo "3. Copy the token"
    echo "4. Add it to your .env file as:"
    echo "   TELEGRAM_BOT_TOKEN=your_token_here"
    echo ""
    exit 1
fi

# Run the bot
echo "✅ Configuration OK"
echo "🚀 Launching bot..."
echo ""
python3 telegram_bot.py
