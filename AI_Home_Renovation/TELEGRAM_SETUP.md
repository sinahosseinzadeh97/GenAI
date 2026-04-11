# Telegram Bot Setup Guide 🤖

## Step 1️⃣: Create Telegram Bot

1. Open Telegram and message [@BotFather](https://t.me/botfather)
2. Send the command `/newbot`
3. Choose a name for your bot (e.g., `AI Home Renovation Bot`)
4. Choose a username ending with `bot` (e.g., `ai_home_renovation_bot`)
5. BotFather will give you a token like this:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```
6. Copy this token

## Step 2️⃣: Configure .env File

1. Open the `.env` file in project folder
2. Find the line `TELEGRAM_BOT_TOKEN`:
   ```
   TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE
   ```
3. Replace `YOUR_TELEGRAM_BOT_TOKEN_HERE` with your token:
   ```
   TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
   ```
4. Save the file

## Step 3️⃣: Run the Bot

In terminal, run:

```bash
cd /Users/apple/Desktop/AI_Home_Renovation
python3 telegram_bot.py
```

If everything is correct, you'll see:
```
INFO - Starting Telegram bot...
INFO - Application started
```

## Step 4️⃣: Use the Bot

1. Find your bot on Telegram (using the username you created)
2. Send `/start` command
3. Bot will greet you!

### How to Use:

**Scenario 1: Renovation with Photos**
1. Send photo of current room
2. (Optional) Send inspiration photo
3. Describe what you want: "I want to renovate my kitchen in modern style"

**Scenario 2: Renovation without Photos**
Just describe: "I want a 150 sqft minimalist kitchen with $10k budget"

### Bot Commands:

- `/start` - Start new conversation
- `/help` - Show help
- `/new` - Start new renovation project
- `/lang` - Change language (English/Persian)

## Bot Features 🎯

✅ **Image Analysis:** Upload and analyze current space photos  
✅ **Inspiration:** Use inspiration photos for design  
✅ **Planning:** Create comprehensive renovation plan  
✅ **Cost Estimate:** Calculate project costs  
✅ **Timeline:** Estimate project duration  
✅ **Visualization:** Generate photorealistic 8K images  
✅ **Bilingual:** Full support for English & Persian

## Important Notes ⚠️

1. **Image Quality:** Send high-quality photos
2. **Clear Descriptions:** The more detailed, the better results
3. **Budget:** Always mention your budget
4. **Response Time:** Image generation takes 30-60 seconds

## Troubleshooting 🔧

### Error: "TELEGRAM_BOT_TOKEN not found"
✅ **Solution:** Check `.env` file and ensure token is added

### Error: "Unauthorized"
✅ **Solution:** Token is incorrect. Get a new one from BotFather

### Bot doesn't respond
✅ **Solution:** 
1. Make sure script is running
2. Check terminal logs
3. Use `/new` command

### Image not generating
✅ **Solution:** 
1. Check GOOGLE_API_KEY in .env
2. Make sure API is active
3. Provide more detailed request

## Running Bot Permanently 🔄

To run bot continuously even after closing terminal:

### Method 1: Using screen
```bash
screen -S renovation_bot
cd /Users/apple/Desktop/AI_Home_Renovation
python3 telegram_bot.py
# Ctrl+A then D to detach
```

### Method 2: Using nohup
```bash
cd /Users/apple/Desktop/AI_Home_Renovation
nohup python3 telegram_bot.py > bot.log 2>&1 &
```

### Method 3: Using systemd (Linux)
Create file `/etc/systemd/system/renovation-bot.service`:
```ini
[Unit]
Description=AI Home Renovation Telegram Bot
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/Users/apple/Desktop/AI_Home_Renovation
ExecStart=/usr/bin/python3 telegram_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable renovation-bot
sudo systemctl start renovation-bot
```

## Support 💬

If you have issues:
1. Check terminal logs
2. Verify .env file
3. Ensure all libraries are installed: `pip install -r requirements.txt`

Good luck! 🚀