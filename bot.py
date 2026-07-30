# Entry point. Railway's Procfile runs `python bot.py`, so this filename must not change.
#
# The code lives in three layers:
#   db.py               Supabase reads/writes, scoped by (channel, user_id). No Telegram, no Spanish text.
#   core.py             State machine, Claude classification, Whisper, all Spanish replies. No Telegram.
#   channel_telegram.py Telegram setup/handlers/polling. No business logic.
#
# Railway deployment: set these five variables under the service's "Variables" tab (Railway injects them as
# environment variables at runtime, so no .env file is deployed — .env is local-development only and gitignored):
#   TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
# OPENAI_API_KEY is used only to transcribe voice notes via the Whisper API.

import logging  # standard library module used to configure logging for the whole process
import os  # standard library module used to read environment variables
import sys  # standard library module used to exit cleanly on a missing configuration

from dotenv import load_dotenv  # loads variables from the local .env file into the environment

load_dotenv()  # populate os.environ from .env when running locally; on Railway the variables are already set

REQUIRED_ENV_VARS = [  # must all be set, or the bot exits at startup with a clear message
    "TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY",
]
missing_env_vars = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]  # which ones, if any, are missing
if missing_env_vars:  # fail fast with a clear message rather than a KeyError traceback deep in some other call
    # Checked before importing the layers below, because each constructs an API client at import time.
    print(f"Missing required environment variable(s): {', '.join(missing_env_vars)}")
    sys.exit(1)

logging.basicConfig(  # configure the root logger for the whole process
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # timestamp, logger name, level, message
    level=logging.INFO,  # log INFO and above (INFO, WARNING, ERROR, CRITICAL)
)

import channel_telegram  # noqa: E402 — imported after the env check, since it builds on core/db which need the keys


if __name__ == "__main__":  # only run the bot when this file is executed directly
    channel_telegram.main()  # start the Telegram adapter's polling loop
