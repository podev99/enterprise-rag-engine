import os
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAG_API_URL = os.getenv("RAG_API_URL")
API_KEY = os.getenv("API_KEY", "techcorp-secret-key-2026")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command in Telegram."""
    welcome_text = (
        "👋 Welcome to TechCorp AI Support Bot!\n\n"
        "Ask me any question about company policies, products, or services."
    )
    await update.message.reply_text(welcome_text)


async def handle_user_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forwards incoming Telegram messages to Cloud RAG API and returns formatted response."""
    user_query = update.message.text
    await update.message.reply_text("🔎 Querying Enterprise Knowledge Base...")

    headers = {"X-API-Key": API_KEY}
    payload = {"query": user_query, "top_k": 2}

    try:
        response = requests.post(RAG_API_URL, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer received.")
            sources = data.get("sources", [])

            formatted_reply = f"🤖 *ANSWER:*\n{answer}\n\n"
            
            if sources and "do not have enough information" not in answer.lower():
                formatted_reply += "📌 *SOURCES USED:*\n"
                for src in sources:
                    formatted_reply += f"• File: `{src['source']}` | Chunk ID: `{src['chunk_id']}` (Distance: {src['distance']:.4f})\n"
            else:
                formatted_reply += "📌 *SOURCES:* None (Out-of-Scope / Refused)"

            await update.message.reply_text(formatted_reply, parse_mode="Markdown")
        else:
            await update.message.reply_text(
                f"⚠️ API Error ({response.status_code}): {response.text}"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to reach RAG Cloud API: {str(e)}")


def main():
    """Starts the Telegram Bot client."""
    if not TELEGRAM_TOKEN or not RAG_API_URL:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or RAG_API_URL in environment variables.")

    print("🚀 TechCorp Telegram Bot is running and waiting for messages...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_query)
    )

    app.run_polling()


if __name__ == "__main__":
    main()