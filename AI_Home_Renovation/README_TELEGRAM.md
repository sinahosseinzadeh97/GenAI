# 🤖 AI Home Renovation Telegram Bot

A Telegram bot powered by Gemini 2.5 Flash AI to help you plan and design home renovations!

## 🌟 Features

- 📸 **Smart Image Analysis:** Upload photos of your current space and get detailed analysis
- 🎨 **Design with Inspiration:** Use inspiration photos to create your plan
- 💰 **Cost Estimation:** Get accurate renovation cost estimates
- ⏱️ **Timeline Planning:** Know how long your project will take
- 🖼️ **Photorealistic Images:** Generate 8K quality images of renovated space
- 🔄 **Image Editing:** Iteratively improve generated images
- 🌐 **Bilingual:** Full support for English and Persian

## 📋 Prerequisites

- Python 3.10+
- Google AI Studio account (for API Key)
- Telegram account

## 🚀 Installation & Setup

### Step 1: Install Libraries

```bash
cd /Users/apple/Desktop/AI_Home_Renovation
pip install -r requirements.txt
```

### Step 2: Create Telegram Bot

1. Go to [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` command
3. Choose name: `My Renovation Bot`
4. Choose username ending with `bot`: `my_renovation_bot`
5. Copy the token BotFather gives you (looks like: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

### Step 3: Setup Tokens

Edit `.env` file in project folder and add your Telegram token:

```bash
GOOGLE_API_KEY=AIzaSyCgOTf5fybrMhv7zNrq1LJ9Nuc8qi-zXY4
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
```

**Note:** Replace with your actual token!

### Step 4: Run the Bot

#### Method 1: With Script (Easiest)

```bash
./run_bot.sh
```

#### Method 2: Direct

```bash
python3 telegram_bot.py
```

If everything is correct, you'll see:

```
INFO - Starting Telegram bot...
```

## 💬 Using the Bot

### Getting Started

1. Find your bot on Telegram (using username you created)
2. Click Start or send `/start`
3. You'll receive a welcome message!

### Commands

- `/start` - Start new conversation with bot
- `/help` - Show complete help
- `/new` - Start new renovation project
- `/lang` - Change language (English ⇄ Persian)

### Usage Scenarios

#### 🎯 Scenario 1: Renovation with Current Room Photo

```
User: [Send kitchen photo]
Bot: What is this photo?
      [Current room button] [Inspiration button]

User: [Click "Current room"]
Bot: 📸 Current room photo received!

User: I want to renovate this kitchen in modern style with $15k budget

Bot: 🤔 Analyzing...
     [Sends complete plan...]
     
     🎨 Generating photorealistic image...
     ⏱️ This will take 30-60 seconds...
     
     [Sends realistic image]
     ✅ Image generated!
```

#### 🎨 Scenario 2: Renovation with Inspiration

```
User: [Send current kitchen photo]
Bot: What is this photo?

User: [Click "Current room"]
Bot: 📸 Current room photo received!

User: [Send beautiful kitchen photo from internet]
Bot: What is this photo?

User: [Click "Inspiration"]
Bot: 🎨 Inspiration photo received!

User: I want exactly this style
Bot: [Design based on inspiration + image]
```

#### 💡 Scenario 3: Renovation without Photos

```
User: I want to design a 200 sqft modern minimalist kitchen
Bot: [Design + estimate + image]
```

#### ✏️ Scenario 4: Edit Generated Image

```
User: Make cabinets cream color
Bot: [Updated image + new version]
```

## 📸 Tips for Best Results

### Photography

✅ **Do:**
- Bright, well-lit photos
- Complete view of space
- High quality (at least 1080p)
- Multiple angles

❌ **Don't:**
- Dark or low-quality images
- Too close zoom
- Blurry photos

### Request Descriptions

✅ **Good:**
- "I want to renovate my 150 sqft kitchen in modern farmhouse style with white cabinets, black countertops, and $15k budget"

❌ **Weak:**
- "Change the kitchen"

## 🔧 Troubleshooting

### ❌ "TELEGRAM_BOT_TOKEN not found"

**Cause:** Token not in `.env` file  
**Solution:**
1. Open `.env` file
2. Add line:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   ```

### ❌ "Unauthorized"

**Cause:** Incorrect token  
**Solution:**
- Get new token from @BotFather
- Ensure you copied complete token

### ⏰ Bot is slow

**Normal!** Image generation takes 30-60 seconds.

### 🖼️ Image not generating

**Check:**
1. Is GOOGLE_API_KEY correct?
2. Internet connection active?
3. Was request detailed enough?

## 🔄 Running Bot Permanently

### Using screen (Recommended)

```bash
# Start new session
screen -S renovation_bot

# Run bot
cd /Users/apple/Desktop/AI_Home_Renovation
python3 telegram_bot.py

# Detach: Ctrl+A then D

# Reattach to session
screen -r renovation_bot

# End session
screen -X -S renovation_bot quit
```

### Using nohup

```bash
cd /Users/apple/Desktop/AI_Home_Renovation
nohup python3 telegram_bot.py > bot.log 2>&1 &

# View logs
tail -f bot.log

# Stop
pkill -f telegram_bot.py
```

## 📊 System Architecture

```
┌─────────────┐
│    User     │
│  Telegram   │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Telegram Bot    │
│  (telegram_bot.py)│
└──────┬───────────┘
       │
       ▼
┌─────────────────────┐
│   Gemini 2.5 Flash  │
│   - Text Analysis   │
│   - Image Generation│
└─────────────────────┘
```

## 🎓 Complete Examples

### Example 1: Complete Kitchen Renovation

```
👤 User: [Send old kitchen photo]
🤖 Bot: What is this photo?

👤 User: [Click: Current room]
🤖 Bot: 📸 Current room photo received!

👤 User: I want to completely renovate this 12m² kitchen. 
         Modern style with white cabinets, gray stone countertops, 
         and ceramic flooring. Budget: $15,000
         
🤖 Bot: 🤔 Analyzing...

🤖 Bot: 
## Current Space Analysis

Your kitchen:
- Dimensions: ~3×4 meters (12m²)
- Issues: Outdated cabinets, poor lighting, old flooring
...

## Design Plan

**Style:** Modern Minimalist
**Colors:**
- Cabinets: Matte white (Shaker style)
- Counters: Quartz gray
- Flooring: 60×60 light gray ceramic
...

💰 **Cost Estimate:** $14,000-$16,000
⏱️ **Timeline:** 2-3 months

🤖 Bot: [Sends photorealistic 8K image]
🎨 Image of your renovated space
```

### Example 2: Design with Inspiration

```
👤 User: [Send current room photo + inspiration photo]
🤖 Bot: [Categorizes photos]
👤 User: I want exactly this style
🤖 Bot: [Analysis + custom plan + image]
```

## 🔐 Security

- Never commit API tokens to git
- `.env` file is in `.gitignore`
- Use personal API keys
- Each user has isolated session
- Images processed in memory

## 🆘 Support

If you encounter problems:

1. **Check logs:**
   ```bash
   python3 telegram_bot.py
   ```
   Errors will display in terminal

2. **Test agent:**
   ```bash
   adk web
   ```
   Does agent work in Web UI?

3. **Check .env:**
   ```bash
   cat .env
   ```
   Are tokens correct?

## 📝 License

This project is free for personal use.

---

**Built with ❤️ and powered by Google ADK + Gemini 2.5 Flash**

🌟 Enjoy your bot and build your dream home!