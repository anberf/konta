# Telegram channel adapter. This is the only module that knows Telegram exists.
#
# Its whole job: turn a Telegram update into (channel, user_id, text-or-audio), hand it to core, and send back
# whatever strings core returns. No business logic, no Spanish text, no Supabase.

import logging  # standard library module used to log bot activity and errors
import os  # standard library module used to read environment variables
from collections import deque  # fixed-size queue used to remember recently processed message IDs

from telegram import Update  # Telegram type representing an incoming update (e.g. a message)
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters  # framework building blocks

import core  # the channel-agnostic state machine and reply generator

logger = logging.getLogger(__name__)  # module-level logger, configured by the entry point

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")  # token used to authenticate with the Telegram API

CHANNEL = "telegram"  # the channel name recorded on every transaction this adapter creates

PROCESSED_MESSAGE_LIMIT = 50  # how many recent messages to remember for duplicate-delivery protection
processed_message_ids: deque = deque(maxlen=PROCESSED_MESSAGE_LIMIT)  # (chat_id, message_id) pairs, oldest first
processed_message_id_set: set = set()  # mirrors the deque above for O(1) duplicate lookups


def mark_message_processed(key: tuple[int, int]) -> bool:  # record a message; return False if it was a duplicate
    if key in processed_message_id_set:  # this exact (chat, message) was already handled
        return False  # tell the caller this message is a duplicate and must be skipped
    if len(processed_message_ids) == processed_message_ids.maxlen:  # the deque is already at capacity
        oldest = processed_message_ids[0]  # the entry that append() is about to evict
        processed_message_id_set.discard(oldest)  # remove it from the lookup set too, keeping both in sync
    processed_message_ids.append(key)  # remember this message, evicting the oldest one if at capacity
    processed_message_id_set.add(key)  # add it to the lookup set
    return True  # this message is new, safe to process


async def send_all(message, replies: list[str]) -> None:  # send each reply core produced, in order
    for reply in replies:  # a flow may legitimately produce more than one message
        await message.reply_text(reply)  # each one is its own Telegram bubble, exactly as before


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # entry point for text messages
    message = update.message  # the incoming Telegram message object
    if message is None or message.text is None:  # ignore updates that are not plain text messages
        return  # nothing to do

    dedup_key = (message.chat_id, message.message_id)  # unique identifier for this exact Telegram message
    if not mark_message_processed(dedup_key):  # this message was already processed (duplicate delivery)
        logger.info("Skipping duplicate message_id %s in chat %s", message.message_id, message.chat_id)  # log it
        return  # skip silently: no reply, no Supabase writes, no state change

    replies = core.handle_incoming_message(CHANNEL, str(update.effective_user.id), message.text)  # business logic
    await send_all(message, replies)  # send whatever core decided to say


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # entry point for voice notes
    message = update.message  # the incoming Telegram message object
    if message is None or message.voice is None:  # ignore updates that are not voice messages
        return  # nothing to do

    dedup_key = (message.chat_id, message.message_id)  # unique identifier for this exact Telegram message
    if not mark_message_processed(dedup_key):  # this message was already processed (duplicate delivery)
        logger.info("Skipping duplicate voice message_id %s in chat %s", message.message_id, message.chat_id)
        return  # skip silently: no reply, no Supabase writes, no state change

    try:  # downloading from Telegram can fail independently of transcription
        voice_file = await context.bot.get_file(message.voice.file_id)  # resolve the file ID to a downloadable file
        audio_buffer = await voice_file.download_as_bytearray()  # pull the .ogg bytes into memory, no temp file needed
    except Exception:  # network error, expired file ID, Telegram outage, ...
        logger.exception("Failed to download voice note")  # log the full traceback for debugging
        await message.reply_text(core.VOICE_TRANSCRIPTION_ERROR)  # ask the user to type it instead
        return  # stop here, leaving the user's state untouched (State A stays State A)

    replies = core.handle_incoming_voice(  # transcribe, then run the same dispatch a typed message takes
        CHANNEL, str(update.effective_user.id), bytes(audio_buffer), "audio/ogg"
    )
    await send_all(message, replies)  # send whatever core decided to say


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # sent when a user opens the bot
    await send_all(update.message, core.welcome_messages())  # welcome message, then the usage examples


def main() -> None:  # build and run the Telegram bot
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()  # construct the bot application
    application.add_handler(CommandHandler("start", handle_start))  # welcome message for new users
    application.add_handler(  # register the handler for incoming text messages
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)  # match any text message that isn't a command
    )
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))  # transcribe and handle voice notes
    application.run_polling()  # start polling Telegram for updates until interrupted
