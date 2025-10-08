# 🚀 Telegram Bot Quick Start

## Step 1: Get Telegram Token (2 minutes)

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot`
3. Bot name: `My Renovation Bot` (or any name)
4. Username: `my_renovation_bot` (must end with `bot`)
5. Copy the token

## Step 2: Setup Token (30 seconds)

Edit `.env` file:

```bash
nano .env
```

Add this line (replace with your actual token):

```
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz-1234567
```

`Ctrl+O` → Enter → `Ctrl+X`

## Step 3: Run (10 seconds)

```bash
python3 telegram_bot.py
```

Or:

```bash
./run_bot.sh
```

## Step 4: Test

1. Find your bot on Telegram
2. Send `/start`
3. Enjoy! 🎉

## Quick Example

```
You: Hello
Bot: Hello! How can I help?

You: I want to design a modern 150 sqft kitchen
Bot: [Complete renovation plan + cost estimate + photorealistic image]
```

## Important Notes

✅ **Make sure:**
- GOOGLE_API_KEY exists in .env
- TELEGRAM_BOT_TOKEN is correct in .env
- Internet connection is active

❌ **Errors:**
- "Token not found" → Add token to .env
- "Unauthorized" → Token is wrong, get new one from BotFather

## Running Permanently

```bash
screen -S bot
python3 telegram_bot.py
# Ctrl+A then D to detach
```

---

**Ready! You can now chat with your bot on Telegram.** 🚀