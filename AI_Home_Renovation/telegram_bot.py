"""
Telegram Bot for AI Home Renovation Planner - Version 2
Features: Image Generation + Bilingual (Persian/English)
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

# Import Gemini
import sys
sys.path.insert(0, os.path.dirname(__file__))

from google import genai
from google.genai import types


class RenovationBot:
    """Main bot class - Bilingual with image generation."""
    
    # Multi-language messages
    MESSAGES = {
        'fa': {
            'welcome': """سلام {name}! 👋

من ربات **برنامه‌ریز بازسازی خانه هوشمند** هستم! 🏠✨

قابلیت‌ها:
📸 تحلیل عکس فضای فعلی
🎨 ایجاد طرح بازسازی
💰 برآورد هزینه و زمان‌بندی  
🖼️ تولید تصویر واقع‌گرایانه 8K

دستورات:
/start - شروع  
/new - پروژه جدید
/lang - تغییر زبان
/help - راهنما

عکس بفرستید یا درخواستتان را بنویسید! 📷""",
            'analyzing': "🤔 در حال تحلیل...",
            'generating_image': "🎨 در حال تولید تصویر واقع‌گرایانه...\n⏱️ این کار 30-60 ثانیه طول می‌کشد...",
            'new_session': "✅ جلسه جدید! آماده بازسازی جدید.",
            'photo_received': "{emoji} عکس {label} دریافت شد!\n\nدرخواست خود را بنویسید.",
            'photo_question': "این عکس چیست؟",
            'photo_current': "📸 اتاق فعلی",
            'photo_inspiration': "🎨 الهام‌بخش",
            'image_generated': "✅ تصویر تولید شد!",
            'lang_changed': "✅ زبان به {lang} تغییر کرد.",
            'error': "❌ خطا: {error}",
        },
        'en': {
            'welcome': """Hello {name}! 👋

I'm your **AI Home Renovation Planner**! 🏠✨

Features:
📸 Analyze current space photos
🎨 Create renovation plans
💰 Estimate costs & timeline
🖼️ Generate photorealistic 8K images

Commands:
/start - Start
/new - New project
/lang - Change language
/help - Help

Send a photo or describe your needs! 📷""",
            'analyzing': "🤔 Analyzing...",
            'generating_image': "🎨 Generating photorealistic image...\n⏱️ This will take 30-60 seconds...",
            'new_session': "✅ New session! Ready for new renovation.",
            'photo_received': "{emoji} {label} photo received!\n\nDescribe what you want.",
            'photo_question': "What is this photo?",
            'photo_current': "📸 Current room",
            'photo_inspiration': "🎨 Inspiration",
            'image_generated': "✅ Image generated!",
            'lang_changed': "✅ Language changed to {lang}.",
            'error': "❌ Error: {error}",
        }
    }
    
    def __init__(self, telegram_token: str):
        self.telegram_token = telegram_token
        self.app = Application.builder().token(telegram_token).build()
        self.user_sessions = {}
        self.genai_client = genai.Client()
        self._setup_handlers()
    
    def _get_message(self, user_id: int, key: str, **kwargs) -> str:
        """Get message in user's language."""
        session = self.user_sessions.get(user_id, {})
        lang = session.get('language', 'fa')
        message = self.MESSAGES[lang].get(key, self.MESSAGES['fa'][key])
        return message.format(**kwargs) if kwargs else message
    
    def _setup_handlers(self):
        """Setup handlers."""
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("new", self.new_session_command))
        self.app.add_handler(CommandHandler("lang", self.language_command))
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback_query))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        user_id = user.id
        
        self.user_sessions[user_id] = {
            "messages": [],
            "images": [],
            "current_room_photo": None,
            "inspiration_photo": None,
            "language": "fa",
            "last_design_plan": None,
        }
        
        welcome = self._get_message(user_id, 'welcome', name=user.first_name)
        await update.message.reply_text(welcome)
    
    async def language_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /lang command."""
        user_id = update.effective_user.id
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "messages": [],
                "images": [],
                "current_room_photo": None,
                "inspiration_photo": None,
                "language": "fa",
                "last_design_plan": None,
            }
        
        keyboard = [
            [
                InlineKeyboardButton("🇮🇷 فارسی", callback_data="lang_fa"),
                InlineKeyboardButton("🇬🇧 English", callback_data="lang_en"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🌐 Choose / انتخاب:",
            reply_markup=reply_markup
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        user_id = update.effective_user.id
        lang = self.user_sessions.get(user_id, {}).get('language', 'fa')
        
        if lang == 'fa':
            help_text = """📖 **راهنما:**

**دستورات:**
/start - شروع
/new - پروژه جدید  
/lang - تغییر زبان
/help - راهنما

**نحوه استفاده:**
1. عکس اتاق فعلی بفرستید
2. (اختیاری) عکس الهام‌بخش بفرستید
3. درخواست بنویسید

**مثال:**
"می‌خوام آشپزخونه 15 متری رو به سبک مدرن با بودجه 30 میلیون بازسازی کنم"

✨ ربات طرح + برآورد + تصویر واقع‌گرایانه می‌دهد!"""
        else:
            help_text = """📖 **Help:**

**Commands:**
/start - Start
/new - New project
/lang - Change language
/help - Help

**How to use:**
1. Send current room photo
2. (Optional) Send inspiration photo
3. Describe what you want

**Example:**
"I want to renovate my 150 sqft kitchen in modern style with $10k budget"

✨ Bot provides plan + estimate + photorealistic image!"""
        
        await update.message.reply_text(help_text)
    
    async def new_session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new command."""
        user_id = update.effective_user.id
        lang = self.user_sessions.get(user_id, {}).get('language', 'fa')
        
        self.user_sessions[user_id] = {
            "messages": [],
            "images": [],
            "current_room_photo": None,
            "inspiration_photo": None,
            "language": lang,
            "last_design_plan": None,
        }
        
        message = self._get_message(user_id, 'new_session')
        await update.message.reply_text(message)
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages."""
        user_id = update.effective_user.id
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "messages": [],
                "images": [],
                "current_room_photo": None,
                "inspiration_photo": None,
                "language": "fa",
                "last_design_plan": None,
            }
        
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = BytesIO()
        await photo_file.download_to_memory(photo_bytes)
        photo_bytes.seek(0)
        
        session = self.user_sessions[user_id]
        
        if not session["current_room_photo"] and not session["inspiration_photo"]:
            photo_current = self._get_message(user_id, 'photo_current')
            photo_inspiration = self._get_message(user_id, 'photo_inspiration')
            
            keyboard = [
                [
                    InlineKeyboardButton(photo_current, callback_data=f"photo_current_{len(session['images'])}"),
                    InlineKeyboardButton(photo_inspiration, callback_data=f"photo_inspiration_{len(session['images'])}"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            session['images'].append({
                'bytes': photo_bytes.read(),
                'type': None,
                'caption': update.message.caption or ""
            })
            
            question = self._get_message(user_id, 'photo_question')
            await update.message.reply_text(question, reply_markup=reply_markup)
        else:
            if not session["current_room_photo"]:
                photo_type = "current_room"
                label = "current room" if session.get('language') == 'en' else "اتاق فعلی"
                emoji = "📸"
            else:
                photo_type = "inspiration"
                label = "inspiration" if session.get('language') == 'en' else "الهام‌بخش"
                emoji = "🎨"
            
            session['images'].append({
                'bytes': photo_bytes.read(),
                'type': photo_type,
                'caption': update.message.caption or ""
            })
            
            if photo_type == "current_room":
                session["current_room_photo"] = len(session['images']) - 1
            else:
                session["inspiration_photo"] = len(session['images']) - 1
            
            message = self._get_message(user_id, 'photo_received', emoji=emoji, label=label)
            await update.message.reply_text(message)
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages."""
        user_id = update.effective_user.id
        user_text = update.message.text
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "messages": [],
                "images": [],
                "current_room_photo": None,
                "inspiration_photo": None,
                "language": "fa",
                "last_design_plan": None,
            }
        
        session = self.user_sessions[user_id]
        lang = session.get('language', 'fa')
        
        await update.message.chat.send_action("typing")
        
        try:
            # Prepare content
            content_parts = []
            
            # Add images
            if session['images']:
                for img_data in session['images']:
                    if img_data['type']:
                        content_parts.append(types.Part.from_bytes(
                            data=img_data['bytes'],
                            mime_type="image/jpeg"
                        ))
            
            # Add text
            content_parts.append(types.Part.from_text(text=user_text))
            
            user_content = types.Content(role="user", parts=content_parts)
            session['messages'].append(user_content)
            
            # Analyze request
            analyzing_msg = self._get_message(user_id, 'analyzing')
            await update.message.reply_text(analyzing_msg)
            
            response_text = await self._call_agent(session['messages'], lang)
            
            if response_text:
                # Save design plan for later image generation
                session['last_design_plan'] = response_text
                
                # Send text response
                await self._send_long_message(update, response_text)
                
                # Generate image if plan contains enough details
                if len(response_text) > 300:  # Has substantial content
                    generating_msg = self._get_message(user_id, 'generating_image')
                    status_message = await update.message.reply_text(generating_msg)
                    
                    image_bytes = await self._generate_renovation_image(response_text, session, lang)
                    
                    if image_bytes:
                        success_msg = self._get_message(user_id, 'image_generated')
                        await update.message.reply_photo(
                            photo=image_bytes,
                            caption=success_msg
                        )
                        await status_message.delete()
                    else:
                        await status_message.edit_text("⚠️ Image generation failed, but you have the plan above.")
            else:
                error_msg = self._get_message(user_id, 'error', error="Could not process request")
                await update.message.reply_text(error_msg)
        
        except Exception as e:
            logger.error(f"Error handling text: {e}", exc_info=True)
            error_msg = self._get_message(user_id, 'error', error=str(e))
            await update.message.reply_text(error_msg)
    
    async def _call_agent(self, messages: list, lang: str) -> Optional[str]:
        """Call Gemini for renovation planning."""
        try:
            last_message = messages[-1] if messages else None
            if not last_message:
                return None
            
            user_text = ""
            for part in last_message.parts:
                if hasattr(part, 'text') and part.text:
                    user_text = part.text
                    break
            
            prompt_parts = []
            
            # Add images
            for part in last_message.parts:
                if not (hasattr(part, 'text') and part.text):
                    prompt_parts.append(part)
            
            # Add text
            if user_text:
                prompt_parts.append(types.Part.from_text(text=user_text))
            
            # System instruction based on language
            if lang == 'fa':
                system_instruction = """شما یک دستیار هوشمند برنامه‌ریزی بازسازی خانه هستید.

وظایف:
1. تحلیل عکس‌های فضای فعلی و الهام‌بخش
2. ارائه طرح جامع بازسازی با جزئیات کامل
3. برآورد هزینه (قیمت‌های ایران - تومان)
4. زمان‌بندی پروژه
5. توضیح دقیق مصالح، رنگ‌ها، و طراحی

مهم: پاسخ را به فارسی و خیلی دقیق بنویسید."""
            else:
                system_instruction = """You are an AI home renovation planning assistant.

Tasks:
1. Analyze current space and inspiration photos
2. Provide comprehensive renovation plan with full details
3. Estimate costs (in USD)
4. Project timeline
5. Detailed materials, colors, and design specifications

Important: Be very detailed and professional."""
            
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(role="user", parts=prompt_parts)],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling agent: {e}", exc_info=True)
            return None
    
    async def _generate_renovation_image(self, design_plan: str, session: dict, lang: str) -> Optional[BytesIO]:
        """Generate photorealistic renovation image."""
        try:
            # Extract key design elements
            if lang == 'fa':
                prompt_base = f"""تصویر واقع‌گرایانه 8K از یک فضای بازسازی شده بر اساس این طرح:

{design_plan[:1000]}

سبک: مدرن و حرفه‌ای
کیفیت: عکاسی حرفه‌ای، روشنایی عالی، زاویه دید مناسب
"""
            else:
                prompt_base = f"""Photorealistic 8K image of a renovated space based on this plan:

{design_plan[:1000]}

Style: Modern and professional
Quality: Professional photography, excellent lighting, proper angle
"""
            
            # Create enhanced prompt
            rewrite_response = self.genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"""Create a detailed image generation prompt based on this renovation plan. 
Make it very specific with colors, materials, layout, lighting. Output as a single detailed paragraph.

{prompt_base}"""
            )
            
            enhanced_prompt = rewrite_response.text
            logger.info(f"Enhanced prompt: {enhanced_prompt[:200]}...")
            
            # Generate image
            image_content = [types.Content(
                role="user",
                parts=[types.Part.from_text(text=enhanced_prompt)]
            )]
            
            config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
            
            for chunk in self.genai_client.models.generate_content_stream(
                model="gemini-2.5-flash-image",
                contents=image_content,
                config=config
            ):
                if (chunk.candidates and chunk.candidates[0].content and 
                    chunk.candidates[0].content.parts):
                    part = chunk.candidates[0].content.parts[0]
                    if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                        image_bytes = BytesIO(part.inline_data.data)
                        logger.info("Image generated successfully")
                        return image_bytes
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating image: {e}", exc_info=True)
            return None
    
    async def _send_long_message(self, update: Update, text: str, max_length: int = 4000):
        """Send long messages by splitting."""
        if len(text) <= max_length:
            await update.message.reply_text(text)
        else:
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
        """Handle button callbacks."""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        session = self.user_sessions.get(user_id)
        
        if not session:
            await query.edit_message_text("Session expired. Use /start")
            return
        
        data = query.data
        
        # Language change
        if data.startswith('lang_'):
            new_lang = data.split('_')[1]
            session['language'] = new_lang
            lang_name = "فارسی" if new_lang == 'fa' else "English"
            message = self._get_message(user_id, 'lang_changed', lang=lang_name)
            await query.edit_message_text(message)
            return
        
        # Photo type
        if data.startswith('photo_'):
            parts = data.split('_')
            action = parts[1]
            img_index = int(parts[2])
            
            if img_index < len(session['images']):
                session['images'][img_index]['type'] = action
                
                if action == "current":
                    session["current_room_photo"] = img_index
                    label = "current room" if session.get('language') == 'en' else "اتاق فعلی"
                    emoji = "📸"
                else:
                    session["inspiration_photo"] = img_index
                    label = "inspiration" if session.get('language') == 'en' else "الهام‌بخش"
                    emoji = "🎨"
                
                message = self._get_message(user_id, 'photo_received', emoji=emoji, label=label)
                await query.edit_message_text(message)
    
    def run(self):
        """Start the bot."""
        logger.info("Starting Telegram bot...")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point."""
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN not found!")
        return
    
    bot = RenovationBot(telegram_token)
    bot.run()


if __name__ == "__main__":
    main()
