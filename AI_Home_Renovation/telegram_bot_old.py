"""
Telegram Bot for AI Home Renovation Planner
This bot integrates with the ADK agent to provide renovation planning via Telegram.
"""

import os
import logging
import asyncio
from io import BytesIO
from typing import Optional
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import ADK components
import sys
sys.path.insert(0, os.path.dirname(__file__))

from google import genai
from google.genai import types


class RenovationBot:
    """Main bot class that handles Telegram interactions and ADK agent communication."""
    
    # Multi-language messages
    MESSAGES = {
        'fa': {
            'welcome': """
سلام {name}! 👋

من ربات **برنامه‌ریز بازسازی خانه هوشمند** هستم! 🏠✨

می‌توانم به شما در موارد زیر کمک کنم:
📸 تحلیل عکس‌های فضای فعلی شما
🎨 ایجاد طرح بازسازی بر اساس سبک مورد نظرتان
💰 برآورد هزینه و زمان‌بندی پروژه
🖼️ تولید تصاویر واقع‌گرایانه از فضای بازسازی شده

**چگونه استفاده کنم؟**
1️⃣ عکسی از فضای فعلی خود بفرستید
2️⃣ (اختیاری) عکس الهام‌بخش از سبک مورد نظرتان بفرستید
3️⃣ بگویید چه نوع بازسازی می‌خواهید

برای تغییر زبان: /lang
برای شروع، عکس بفرستید یا درخواست خود را بنویسید! 📷
""",
            'analyzing': "🤔 در حال تحلیل و برنامه‌ریزی...",
            'generating_image': "🎨 در حال تولید تصویر واقع‌گرایانه...\nاین کار 30-60 ثانیه طول می‌کشد...",
            'new_session': "✅ جلسه جدید شروع شد! برای بازسازی جدید آماده‌ام. عکس یا درخواست خود را ارسال کنید.",
            'photo_received': "{emoji} عکس {label} دریافت شد!\n\nحالا بگویید چه نوع بازسازی می‌خواهید یا عکس دیگری بفرستید.",
            'photo_question': "این عکس چیست؟",
            'photo_current': "📸 عکس اتاق فعلی",
            'photo_inspiration': "🎨 عکس الهام‌بخش",
            'image_generated': "✅ تصویر با موفقیت تولید شد!",
        },
        'en': {
            'welcome': """
Hello {name}! 👋

I'm your **AI Home Renovation Planner** bot! 🏠✨

I can help you with:
📸 Analyzing photos of your current space
🎨 Creating renovation plans based on your desired style
💰 Estimating costs and project timelines
🖼️ Generating photorealistic images of your renovated space

**How to use:**
1️⃣ Send a photo of your current space
2️⃣ (Optional) Send inspiration photos of your desired style
3️⃣ Tell me what kind of renovation you want

Change language: /lang
To get started, send a photo or describe what you need! 📷
""",
            'analyzing': "🤔 Analyzing and planning...",
            'generating_image': "🎨 Generating photorealistic image...\nThis will take 30-60 seconds...",
            'new_session': "✅ New session started! Ready for a new renovation project. Send a photo or your request.",
            'photo_received': "{emoji} {label} photo received!\n\nNow tell me what kind of renovation you want, or send another photo.",
            'photo_question': "What is this photo?",
            'photo_current': "📸 Current room photo",
            'photo_inspiration': "🎨 Inspiration photo",
            'image_generated': "✅ Image successfully generated!",
        }
    }
    
    def __init__(self, telegram_token: str):
        self.telegram_token = telegram_token
        self.app = Application.builder().token(telegram_token).build()
        
        # User session storage (user_id -> session data)
        self.user_sessions = {}
        
        # Gemini client for image generation
        self.genai_client = genai.Client()
        
        # Setup handlers
        self._setup_handlers()
    
    def _get_message(self, user_id: int, key: str, **kwargs) -> str:
        """Get message in user's language."""
        session = self.user_sessions.get(user_id, {})
        lang = session.get('language', 'fa')  # Default to Persian
        message = self.MESSAGES[lang].get(key, self.MESSAGES['fa'][key])
        return message.format(**kwargs) if kwargs else message
    
    def _setup_handlers(self):
        """Setup all command and message handlers."""
        # Command handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("new", self.new_session_command))
        self.app.add_handler(CommandHandler("lang", self.language_command))
        
        # Message handlers
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        welcome_message = f"""
سلام {user.first_name}! 👋

من ربات **برنامه‌ریز بازسازی خانه هوشمند** هستم! 🏠✨

می‌توانم به شما در موارد زیر کمک کنم:
📸 تحلیل عکس‌های فضای فعلی شما
🎨 ایجاد طرح بازسازی بر اساس سبک مورد نظرتان
💰 برآورد هزینه و زمان‌بندی پروژه
🖼️ تولید تصاویر واقع‌گرایانه از فضای بازسازی شده

**چگونه استفاده کنم؟**
1️⃣ عکسی از فضای فعلی خود بفرستید
2️⃣ (اختیاری) عکس الهام‌بخش از سبک مورد نظرتان بفرستید
3️⃣ بگویید چه نوع بازسازی می‌خواهید (مثلاً "می‌خوام آشپزخونه‌م رو مدرن کنم")

برای شروع، به من بگویید کدام اتاق را می‌خواهید بازسازی کنید یا عکس بفرستید! 📷
"""
        await update.message.reply_text(welcome_message)
        
        # Initialize user session
        user_id = update.effective_user.id
        self.user_sessions[user_id] = {
            "messages": [],
            "images": [],
            "current_room_photo": None,
            "inspiration_photo": None,
        }
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """
📖 **راهنمای استفاده:**

**دستورات:**
/start - شروع مکالمه جدید
/new - شروع پروژه جدید
/help - نمایش این راهنما

**نحوه استفاده:**
1. عکس اتاق فعلی خود را ارسال کنید
2. (اختیاری) عکس الهام‌بخش ارسال کنید
3. درخواست خود را بنویسید: "می‌خوام آشپزخونه‌م رو به سبک مدرن بازسازی کنم با بودجه 20 میلیون تومان"

**قابلیت‌ها:**
✅ تحلیل تصویر با هوش مصنوعی
✅ طراحی بازسازی حرفه‌ای
✅ برآورد هزینه و زمان‌بندی
✅ تولید تصویر واقع‌گرایانه از نتیجه نهایی

سوال دیگری دارید؟ فقط بپرسید! 🎯
"""
        await update.message.reply_text(help_text)
    
    async def new_session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new command to start a new session."""
        user_id = update.effective_user.id
        self.user_sessions[user_id] = {
            "messages": [],
            "images": [],
            "current_room_photo": None,
            "inspiration_photo": None,
        }
        await update.message.reply_text(
            "✅ جلسه جدید شروع شد! برای بازسازی جدید آماده‌ام. عکس یا درخواست خود را ارسال کنید."
        )
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages."""
        user_id = update.effective_user.id
        
        # Initialize session if not exists
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "messages": [],
                "images": [],
                "current_room_photo": None,
                "inspiration_photo": None,
            }
        
        # Download photo
        photo = update.message.photo[-1]  # Get highest resolution
        photo_file = await photo.get_file()
        photo_bytes = BytesIO()
        await photo_file.download_to_memory(photo_bytes)
        photo_bytes.seek(0)
        
        # Store photo data
        session = self.user_sessions[user_id]
        
        # Ask user what type of photo this is
        if not session["current_room_photo"] and not session["inspiration_photo"]:
            keyboard = [
                [
                    InlineKeyboardButton("📸 عکس اتاق فعلی", callback_data=f"photo_current_{len(session['images'])}"),
                    InlineKeyboardButton("🎨 عکس الهام‌بخش", callback_data=f"photo_inspiration_{len(session['images'])}"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Store photo temporarily
            session['images'].append({
                'bytes': photo_bytes.read(),
                'type': None,
                'caption': update.message.caption or ""
            })
            
            await update.message.reply_text(
                "این عکس چیست؟",
                reply_markup=reply_markup
            )
        else:
            # Determine type automatically
            if not session["current_room_photo"]:
                photo_type = "current_room"
                emoji = "📸"
                label = "اتاق فعلی"
            else:
                photo_type = "inspiration"
                emoji = "🎨"
                label = "الهام‌بخش"
            
            session['images'].append({
                'bytes': photo_bytes.read(),
                'type': photo_type,
                'caption': update.message.caption or ""
            })
            
            if photo_type == "current_room":
                session["current_room_photo"] = len(session['images']) - 1
            else:
                session["inspiration_photo"] = len(session['images']) - 1
            
            await update.message.reply_text(
                f"{emoji} عکس {label} دریافت شد!\n\n"
                f"حالا بگویید چه نوع بازسازی می‌خواهید یا عکس دیگری بفرستید."
            )
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages."""
        user_id = update.effective_user.id
        user_text = update.message.text
        
        # Initialize session if not exists
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "messages": [],
                "images": [],
                "current_room_photo": None,
                "inspiration_photo": None,
            }
        
        session = self.user_sessions[user_id]
        
        # Show typing indicator
        await update.message.chat.send_action("typing")
        
        try:
            # Prepare message content for ADK agent
            content_parts = []
            
            # Add images if available
            if session['images']:
                for img_data in session['images']:
                    if img_data['type']:
                        # Create inline data part for image
                        content_parts.append(types.Part.from_bytes(
                            data=img_data['bytes'],
                            mime_type="image/jpeg"
                        ))
            
            # Add text
            content_parts.append(types.Part.from_text(text=user_text))
            
            # Create message content
            user_content = types.Content(
                role="user",
                parts=content_parts
            )
            
            # Add to session history
            session['messages'].append(user_content)
            
            # Send to agent and get response
            await update.message.reply_text("🤔 در حال تحلیل و برنامه‌ریزی...")
            
            # Use the agent
            response_text = await self._call_agent(session['messages'])
            
            # Send response back to user
            if response_text:
                # Split long messages
                await self._send_long_message(update, response_text)
            else:
                await update.message.reply_text(
                    "❌ متأسفانه مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
                )
        
        except Exception as e:
            logger.error(f"Error handling text message: {e}", exc_info=True)
            await update.message.reply_text(
                f"❌ خطا: {str(e)}\n\nلطفاً دوباره تلاش کنید یا از /new برای شروع جلسه جدید استفاده کنید."
            )
    
    async def _call_agent(self, messages: list) -> Optional[str]:
        """Call the ADK agent with messages and return response."""
        try:
            # Get the last user message
            last_message = messages[-1] if messages else None
            
            if not last_message:
                return None
            
            # Extract text from last message
            user_text = ""
            for part in last_message.parts:
                if hasattr(part, 'text') and part.text:
                    user_text = part.text
                    break
            
            # Create a simple text-based prompt for the agent
            # Since ADK's synchronous interface is complex, we'll use direct Gemini API
            client = genai.Client()
            
            # Build prompt with context
            prompt_parts = []
            
            # Add images if available (they're already in Part format)
            for part in last_message.parts:
                # Check if it's not a text part (i.e., it's an image)
                if not (hasattr(part, 'text') and part.text):
                    prompt_parts.append(part)
            
            # Add text
            if user_text:
                prompt_parts.append(types.Part.from_text(text=user_text))
            
            # Use the agent's instruction as system context
            system_instruction = """
شما یک دستیار هوشمند برنامه‌ریزی بازسازی خانه هستید.

وظایف شما:
1. تحلیل عکس‌های فضای فعلی و الهام‌بخش
2. ارائه طرح جامع بازسازی
3. برآورد هزینه (بر اساس قیمت‌های ایران)
4. زمان‌بندی پروژه
5. توضیح دقیق مصالح و متریال

به فارسی پاسخ دهید و خیلی دقیق و حرفه‌ای باشید.
"""
            
            # Call Gemini
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(role="user", parts=prompt_parts)
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling agent: {e}", exc_info=True)
            return None
    
    async def _send_long_message(self, update: Update, text: str, max_length: int = 4000):
        """Send long messages by splitting them."""
        # Telegram has a 4096 character limit per message
        # Use 4000 to be safe
        if len(text) <= max_length:
            await update.message.reply_text(text)
        else:
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= max_length:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        await update.message.reply_text(current_chunk.strip())
                        await asyncio.sleep(0.5)
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                await update.message.reply_text(current_chunk.strip())
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button callbacks."""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        session = self.user_sessions.get(user_id)
        
        if not session:
            await query.edit_message_text("❌ جلسه منقضی شده. لطفاً از /start استفاده کنید.")
            return
        
        # Parse callback data
        data_parts = query.data.split('_')
        action = data_parts[1]  # current or inspiration
        img_index = int(data_parts[2])
        
        # Update image type
        if img_index < len(session['images']):
            session['images'][img_index]['type'] = action
            
            if action == "current":
                session["current_room_photo"] = img_index
                emoji = "📸"
                label = "اتاق فعلی"
            else:
                session["inspiration_photo"] = img_index
                emoji = "🎨"
                label = "الهام‌بخش"
            
            await query.edit_message_text(
                f"{emoji} عکس {label} ذخیره شد!\n\n"
                f"حالا بگویید چه نوع بازسازی می‌خواهید."
            )
    
    def run(self):
        """Start the bot."""
        self.app.add_handler(CallbackQueryHandler(self.handle_callback_query))
        logger.info("Starting Telegram bot...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point."""
    # Get Telegram token from environment
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables!")
        logger.error("Please add TELEGRAM_BOT_TOKEN to your .env file")
        return
    
    # Create and run bot
    bot = RenovationBot(telegram_token)
    bot.run()


if __name__ == "__main__":
    main()
