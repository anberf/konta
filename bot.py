import difflib  # standard library module used for typo-tolerant fuzzy text matching
import json  # standard library module used to parse and serialize JSON data
import logging  # standard library module used to log bot activity and errors
import os  # standard library module used to read environment variables
import re  # standard library module used for regular expression matching
import sys  # standard library module used to exit cleanly on a missing configuration
from collections import deque  # fixed-size queue used to remember recently processed message IDs
from datetime import date, datetime, timedelta, timezone  # date/time utilities for date-range parsing and timestamps
from zoneinfo import ZoneInfo  # IANA timezone database lookup, used to resolve "today" in the vendor's local time

import anthropic  # official Anthropic SDK used to call the Claude API
from dotenv import load_dotenv  # loads variables from the local .env file into the environment
from supabase import create_client, Client  # Supabase client used to read and write the transactions table
from telegram import Update  # Telegram type representing an incoming update (e.g. a message)
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters  # Telegram framework blocks

load_dotenv()  # populate os.environ with the values defined in .env, if one exists (e.g. local dev, not on Railway)

REQUIRED_ENV_VARS = ["TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY"]  # must all be set
missing_env_vars = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]  # which ones, if any, are missing
if missing_env_vars:  # fail fast with a clear message rather than a KeyError traceback deep in some other call
    print(f"Missing required environment variable(s): {', '.join(missing_env_vars)}")
    sys.exit(1)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")  # Telegram bot token used to authenticate with the Telegram API
SUPABASE_URL = os.environ.get("SUPABASE_URL")  # Supabase project REST URL
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")  # Supabase anon API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # Anthropic API key used to call Claude

logging.basicConfig(  # configure the root logger for the whole process
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # log line format: timestamp, logger name, level, message
    level=logging.INFO,  # log INFO and above (INFO, WARNING, ERROR, CRITICAL)
)
logger = logging.getLogger(__name__)  # module-level logger used throughout this file

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)  # Supabase client instance used for all database ops
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)  # Anthropic client instance used for all Claude API calls

CLAUDE_MODEL = "claude-sonnet-4-6"  # Claude model used to classify every incoming message

LOCAL_TZ = ZoneInfo("Europe/Amsterdam")  # the vendor's local timezone, used to resolve "today" for reports


def today_local() -> date:  # today's calendar date in the vendor's local timezone (created_at is stored as UTC)
    return datetime.now(LOCAL_TZ).date()

# Claude Sonnet 4.6 does not support structured outputs (output_config.format), so classification
# relies on the system prompt's "return ONLY a JSON object" instruction plus defensive parsing below.

SYSTEM_PROMPT = """You are a financial assistant for Spanish-speaking street food vendors. Your only job is to classify the user's message — the application decides whether to save, query, or correct anything based on your classification.

Classify the message into one of five intents:

* new_transaction — the user is registering a new income, expense, loan, debt, collection, or debt repayment (e.g. "vendí 500 pesos de tacos", "le presté 200 a Bryan", "wilmer me prestó 100000", "bryan me pagó 50000", "le pagué 100000 a wilmer").
* correction — the user wants to fix a past transaction (e.g. "no son 10000, son 20000", "borra lo último", "me equivoqué", "la luz debería ser recurrente, no única vez").
* query — the user is asking for a balance or report (e.g. "¿cuánto llevo hoy?", "¿cuánto me debe Bryan?").
* confirmation — the user is confirming or rejecting a proposed correction (e.g. "sí", "no", "ese no").
* cancellation — the user wants to cancel what's happening, undo the last transaction, or delete a specific past transaction (e.g. "olvídalo", "no importa", "cancela", "borra eso", "quiero eliminar la transacción de la luz", "borra el pago de gas").

Return ONLY a JSON object with these fields:

* intent: one of the five values above
* amount: number extracted or calculated, or null if not applicable
* type: "ingreso", "gasto", "préstamo", "deuda", "cobro", or "pago_deuda", or null if not applicable
* category: "negocio" or "personal" ("préstamo", "deuda", and "pago_deuda" can each be either — ask if unclear; "cobro" is always "negocio"), or null if not applicable
* recurrence: "única vez", "recurrente", or "variable", or null if not applicable
* description: clean 1-sentence summary in Spanish, or null if not applicable
* debtor_name: for "préstamo" (money the business lends OUTWARD to a customer) or "cobro" (money collected back from a debtor), name of the person who owes or owed the business money, or null
* creditor_name: for "deuda" (money someone lends TO the business) or "pago_deuda" (money repaid to a creditor), name of the person who lent the money, or null
* needs_clarification: true or false
* clarification_question: simple Spanish question if needs_clarification is true, otherwise null
* correction_hint: for a "correction" intent, this is ALWAYS the OLD value of the specific field being changed — the value that was originally registered and is now wrong. It is NEVER the new value the user just stated. Example: in "ah no fueron 24000", 24000 is the NEW value (goes in new_value below); correction_hint is whatever amount was previously registered, which you must infer from the conversation context. If you cannot determine the old value from context, set needs_clarification to true and set clarification_question to ask the user what the original value was. Null for all intents other than "correction".
* transaction_keyword: for a "correction" or "cancellation" intent, a short keyword or phrase naming WHICH past transaction the user means, taken from what they're describing rather than the field they're changing — e.g. in "la luz debería ser recurrente" this is "luz" (not "recurrente", which is the new value, and not "única vez", which would be the old recurrence value); in "quiero eliminar la transacción de la luz" this is also "luz". Use a word that would plausibly appear in that transaction's description (e.g. "luz", "arepas", "Bryan"). Set to null if the user doesn't reference a specific transaction by description (e.g. "borra lo último", "no son 10000, son 20000" with no other context — these rely on the most recent transaction or the amount instead).
* new_value: for a "correction" intent, the corrected value the user wants to apply — a new amount (e.g. "24000"), a new description (e.g. "gas para el negocio"), or a new recurrence category ("recurrente", "variable", or "única vez") if the user is correcting how often the transaction repeats (e.g. "la luz debería ser recurrente"). Null for all other intents.
* query_period: for a "query" intent, one of "hoy", "semana", "mes", "deudores", "acreedores", "custom", or null if unclear. "hoy" is asking about today/current status (e.g. "cómo voy", "cómo voy hoy", "cuánto tengo", "cuánto llevo", "resumen"). "semana" is asking about this week (e.g. "cómo voy esta semana"). "mes" is asking about this month (e.g. "cómo voy este mes"). "deudores" is asking who owes the vendor money (e.g. "quién me debe", "lista de deudores", "cuánto me deben"). "acreedores" is asking who the vendor owes money to (e.g. "a quién le debo", "lista de acreedores", "cuánto debo"). "custom" is when the user names a specific starting point (e.g. "desde el 10 de julio", "desde otra fecha") — also set query_date_hint. Null for all non-"query" intents.
* query_date_hint: for a "query" intent with query_period "custom", the raw date expression the user gave (e.g. "el 10 de julio", "hace 2 semanas"). Null otherwise.

Recurrence classification rules:
* "recurrente": bills and services paid regularly — luz, gas, agua, arriendo, teléfono, internet, sueldo de empleado, cualquier pago que se repite cada semana o mes.
* "variable": sales income and supply purchases that change day to day — ventas, compras de ingredientes o mercancía, gastos que dependen del volumen del negocio.
* "única vez": one-off purchases or events — compra de un electrodoméstico, una olla, un mueble, algo que no se repite.

Recurrence examples:
* "pagué la luz 50000" → recurrente
* "pagué el arriendo" → recurrente
* "compré harina para las arepas" → variable
* "vendí 10 empanadas" → variable
* "compré una estufa nueva" → única vez
* "le pagué a mi empleada" → recurrente

Loan vs. debt:
* "préstamo" is money the business or the vendor personally lends OUTWARD to someone else — extract debtor_name (the person who now owes it back). Category can be either "negocio" or "personal"; follow the category-ambiguity rule below like any other type.
* "deuda" is money someone lends TO the business or to the vendor personally (e.g. "wilmer me prestó", "me fiaron", "me adelantaron") — extract creditor_name (the person who lent the money). Category can be either "negocio" or "personal"; follow the category-ambiguity rule below like any other type.

Loan/debt examples:
* "wilmer me prestó 100000 para comprar insumos" → type: deuda, creditor_name: Wilmer, category: negocio
* "wilmer me prestó 100000" (no context which one) → type: deuda, creditor_name: Wilmer, needs_clarification: true, clarification_question: "¿Ese préstamo fue para el negocio o para algo personal?\n1. negocio\n2. personal"
* "le presté 50000 a pedro para que comprara mercancía" → type: préstamo, debtor_name: Pedro, category: negocio
* "le presté 50000 a pedro" (no context which one) → type: préstamo, debtor_name: Pedro, needs_clarification: true, clarification_question: "¿Ese préstamo fue del negocio o algo personal?\n1. negocio\n2. personal"
* "me fiaron la harina" → type: deuda, needs_clarification: true (ask for the amount first, then category if still unclear)

Collection vs. debt repayment:
* "cobro" is money the vendor collected back from a debtor who owed the business (e.g. "bryan me pagó", "cobré de pedro", "me devolvió la plata jason") — extract debtor_name (the person who paid). Always category "negocio", never ask.
* "pago_deuda" is money the vendor repaid to a creditor who had lent money (e.g. "le pagué a wilmer", "aboné al préstamo de jason") — extract creditor_name (the person who was repaid). Category can be either "negocio" or "personal"; follow the category-ambiguity rule below like any other type.

Collection/repayment examples:
* "bryan me pagó 50000" → type: cobro, debtor_name: Bryan, category: negocio
* "cobré 30000 de pedro" → type: cobro, debtor_name: Pedro, category: negocio
* "me devolvió la plata jason" → type: cobro, debtor_name: Jason, needs_clarification: true (ask for the amount)
* "cobré lo que me debía pedro" → type: cobro, debtor_name: Pedro, needs_clarification: true (ask for the amount)
* "le pagué 100000 a wilmer" → type: pago_deuda, creditor_name: Wilmer, needs_clarification: true, clarification_question: "¿Ese pago fue del negocio o algo personal?\n1. negocio\n2. personal" (unless context already signals which one)
* "aboné 50000 al préstamo de jason, fue para el negocio" → type: pago_deuda, creditor_name: Jason, category: negocio
* "aboné al préstamo" → type: pago_deuda, needs_clarification: true (ask for the creditor's name and the amount)

Query examples:
* "como voy" → intent: query, query_period: hoy
* "como voy hoy" → intent: query, query_period: hoy
* "cuanto tengo" → intent: query, query_period: hoy
* "que tengo hoy" → intent: query, query_period: hoy
* "cuanto llevo" → intent: query, query_period: hoy
* "resumen" → intent: query, query_period: hoy
* "como voy esta semana" → intent: query, query_period: semana
* "como voy este mes" → intent: query, query_period: mes
* "quien me debe" → intent: query, query_period: deudores
* "a quien le debo" → intent: query, query_period: acreedores

Rules:
* A message asking about a balance, total, or who-owes-whom (e.g. "cómo voy", "cuánto tengo", "cuánto me deben") is always intent "query", never "new_transaction" — it names no amount/type to save, only a question about existing data.
* If the user is answering a previous clarification question, combine their answer with the original message and classify again.
* Whenever clarification_question offers a choice between a small fixed set of options, always format it as a numbered list the user can answer with either the number or the word, e.g. "¿Es un pago que se repite cada semana o mes, o fue algo de una sola vez?\n1. se repite\n2. una sola vez" — never leave such a question as plain prose.
* If the recurrence cannot be confidently determined from the message using the rules and examples above (it isn't a clearly recurring bill/service, a clearly variable sale/supply purchase, or a clearly one-off item), set needs_clarification to true and set clarification_question to a numbered question, e.g. "¿Es un pago que se repite cada semana o mes, o fue algo de una sola vez?\n1. se repite\n2. una sola vez". Do not ask this for transactions that already clearly match one of the recurrence categories or examples above — only ask when genuinely unclear, and only after amount and type are already known (ask about amount or type first if those are also missing).
* If category cannot be confidently determined from the message (no mention of "negocio", "personal", "casa", "cliente", the business itself, etc. — nothing signals which one), set needs_clarification to true and set clarification_question to "¿Este gasto fue para el negocio o personal?\n1. negocio\n2. personal" (adjust wording for ingreso, préstamo, deuda, or pago_deuda if needed, e.g. "¿Ese préstamo fue para el negocio o para algo personal?\n1. negocio\n2. personal" — always keep the numbered list). Do NOT ask when the message already signals it either way — e.g. "de la casa" or "para mí" means personal; naming the business, a sale, or a supply purchase means negocio. "cobro" is always "negocio", never ask for it. Ask about category only after amount, type, and recurrence are already resolved (one open question at a time)."""
# ^ System prompt text sent to Claude on every classification call. The lines inside the string above
# are prompt content, not Python code, so they are not individually annotated with trailing comments.

CANCELLATION_KEYWORDS = ["olvidalo", "olvídalo", "no importa", "cancela", "borra eso"]  # phrases meaning "cancel"
AFFIRMATIVE_REPLIES = {"si", "sí", "s", "yes", "correcto", "esa es", "esa"}  # normalized replies treated as "yes"
NEGATIVE_REPLIES = {"no", "ese no", "esa no", "no es"}  # normalized replies treated as "no"
AMOUNT_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")  # regex matching a plain numeric string (optional sign and decimals)

YES_NO_MENU = "Responde con el número o la palabra:\n1. sí\n2. no"  # shared numbered menu for binary confirmations

TIME_WINDOW_QUESTION = (  # shared numbered menu for the "when was this transaction" question
    "¿Cuándo fue esa transacción? Responde con el número o la palabra:\n"
    "1. hoy\n2. ayer\n3. esta semana\n4. la semana pasada"
)

TIME_WINDOW_RETRY = (  # shown when the reply to TIME_WINDOW_QUESTION didn't match any known option
    "No entendí el periodo. Responde con el número o la palabra:\n"
    "1. hoy\n2. ayer\n3. esta semana\n4. la semana pasada"
)

NEW_VALUE_PROMPT = (  # fully numbered menu of every field the user can correct
    "¿Qué quieres corregir? Responde con el número o la palabra:\n"
    "1. el monto\n2. la descripción\n"
    "3. frecuencia: recurrente\n4. frecuencia: variable\n5. frecuencia: única vez\n"
    "6. categoría: negocio\n7. categoría: personal"
)

FIELD_MENU = {  # numbered/word options that need a follow-up question before a value is known
    "1": "amount", "monto": "amount", "valor": "amount",
    "2": "description", "descripcion": "description", "descripción": "description",
}

VALUE_MENU = {  # numbered options that are themselves a complete, immediately-applicable answer
    "3": "recurrente", "4": "variable", "5": "única vez",
    "6": "negocio", "7": "personal",
}

FREQUENCY_QUESTION = (  # shown when the user references "frecuencia" without naming a specific value
    "¿Cuál quieres que sea la frecuencia? Responde con el número o la palabra:\n"
    "1. recurrente\n2. variable\n3. única vez"
)

FREQUENCY_RETRY = (  # shown when the reply to FREQUENCY_QUESTION still isn't a recognizable value
    "No entendí. Responde con el número o la palabra:\n1. recurrente\n2. variable\n3. única vez"
)

CATEGORY_QUESTION = (  # shown when the user references "categoría" without naming a specific value
    "¿Cuál quieres que sea la categoría? Responde con el número o la palabra:\n1. negocio\n2. personal"
)

CATEGORY_RETRY = (  # shown when the reply to CATEGORY_QUESTION still isn't a recognizable value
    "No entendí. Responde con el número o la palabra:\n1. negocio\n2. personal"
)

FREQUENCY_MENU = {"1": "recurrente", "2": "variable", "3": "única vez"}  # numbered shortcuts for FREQUENCY_QUESTION
CATEGORY_MENU = {"1": "negocio", "2": "personal"}  # numbered shortcuts for CATEGORY_QUESTION

DESCRIPTION_FOLLOWUP_QUESTION = (  # asked after an amount-only correction, so the description isn't silently guessed
    "¿También quieres actualizar la descripción? Responde con el número o la palabra:\n"
    "1. sí\n2. no\n(Es mejor para el control de cuentas)"
)

DESCRIPTION_FOLLOWUP_RETRY = (  # shown when the reply to DESCRIPTION_FOLLOWUP_QUESTION wasn't a recognizable yes/no
    "No entendí. Responde con el número o la palabra:\n1. sí\n2. no"
)

# In-memory conversation state per Telegram user ID; a missing entry means the user is in State A.
user_states: dict[int, dict] = {}

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


def classify_message(prompt_text: str) -> dict:  # call Claude to classify a message and return the parsed JSON result
    response = claude.messages.create(  # send the classification request to the Claude API
        model=CLAUDE_MODEL,  # use the configured Claude model
        max_tokens=1024,  # cap the response length
        system=SYSTEM_PROMPT,  # attach the classification instructions as the system prompt
        messages=[{"role": "user", "content": prompt_text}],  # send the user's message as the only turn
    )
    raw_text = next(block.text for block in response.content if block.type == "text")  # pull out the text block
    return parse_classification(raw_text)  # parse and return the JSON object Claude returned


def parse_classification(raw_text: str) -> dict:  # defensively parse Claude's JSON reply, tolerating code fences
    cleaned = raw_text.strip()  # remove leading/trailing whitespace
    if cleaned.startswith("```"):  # Claude sometimes wraps JSON in a markdown code fence despite instructions
        cleaned = cleaned.strip("`").strip()  # remove the backticks
        if cleaned.startswith("json"):  # the fence may be labeled ```json
            cleaned = cleaned[4:].strip()  # drop the "json" language tag
    return json.loads(cleaned)  # parse the cleaned text as JSON


def is_cancellation(text: str) -> bool:  # check whether a message contains a known cancellation phrase
    normalized = text.strip().lower()  # normalize the text for case-insensitive matching
    return any(keyword in normalized for keyword in CANCELLATION_KEYWORDS)  # true if any cancellation keyword appears


def try_parse_amount(text: str) -> float | None:  # try to interpret a free-text reply as a numeric amount
    cleaned = text.strip().replace("$", "").replace(" ", "").replace(",", "")  # strip symbols, spaces, thousands separators
    if not AMOUNT_PATTERN.match(cleaned):  # reject anything that isn't a plain number after cleaning
        return None  # not a parseable amount
    return float(cleaned)  # convert the cleaned string to a float


def extract_amount_from_text(text: str) -> float | None:  # find a numeric amount among the words of free text
    for token in text.split():  # check each whitespace-separated token, e.g. "15000 al bryan" -> ["15000", "al", "bryan"]
        parsed = try_parse_amount(token)  # try to parse this single token as a number
        if parsed is not None:  # this token is a number
            return parsed  # use the first numeric token found
    return None  # no token in the text looked like a number


FUZZY_MATCH_THRESHOLD = 0.75  # similarity ratio (0-1) above which two words are considered the same, typos included


def fuzzy_match(word: str, candidates: list[str], threshold: float = FUZZY_MATCH_THRESHOLD) -> str | None:  # closest match
    matches = difflib.get_close_matches(word, candidates, n=1, cutoff=threshold)  # best candidate above the threshold
    return matches[0] if matches else None  # the matched candidate string, or None if nothing was close enough


RECURRENCE_PHRASES = ["unica vez", "única vez", "una vez", "unico", "único"]  # one-off phrases, checked as substrings
FREQUENCY_CONCEPT_ROOTS = ["frecuenc", "recurrenc"]  # roots of "frecuencia"/"recurrencia", the CATEGORY noun, not a value
RECURRENCE_VALUE_WORDS = {"recurrente": "recurrente", "variable": "variable"}  # canonical value words, fuzzy-matched below


def extract_recurrence_from_text(text: str) -> str | None:  # find an explicit recurrence category among free text
    normalized = text.strip().lower()  # normalize for case-insensitive matching
    for token in normalized.split():  # check each whitespace-separated word
        cleaned = token.strip(".,;:!?")  # drop trailing punctuation, e.g. "recurrente." -> "recurrente"
        if "ncia" in cleaned:  # "recurrencia"/"frecuencia" are the CATEGORY noun, never a value — skip this token
            continue  # move on to the next token
        match = fuzzy_match(cleaned, list(RECURRENCE_VALUE_WORDS))  # typo-tolerant match against the value words
        if match:  # this token is close enough to a known value, misspelled or not
            return RECURRENCE_VALUE_WORDS[match]  # return the canonical value it matched
    if any(phrase in normalized for phrase in RECURRENCE_PHRASES):  # multi-word phrases can't be fuzzy-matched per-token
        return "única vez"  # e.g. "fue de una vez", "única vez", "unico pago"
    return None  # no recurrence category was named in the text


def references_frequency_without_value(text: str) -> bool:  # user named the concept ("frecuencia") but no specific value
    normalized = text.strip().lower()  # normalize for matching
    mentions_concept = any(root in normalized for root in FREQUENCY_CONCEPT_ROOTS)  # e.g. "frecuencia", "recurrencia"
    return mentions_concept and extract_recurrence_from_text(text) is None  # true only if no concrete value was also given


CATEGORY_CONCEPT_ROOTS = ["categor"]  # root of "categoría"/"categoria", the CATEGORY noun, not a value
CATEGORY_VALUE_WORDS = {"negocio": "negocio", "personal": "personal"}  # canonical value words, fuzzy-matched below


def extract_category_from_text(text: str) -> str | None:  # find an explicit category value among free text
    normalized = text.strip().lower()  # normalize for case-insensitive matching
    for token in normalized.split():  # check each whitespace-separated word
        cleaned = token.strip(".,;:!?")  # drop trailing punctuation, e.g. "negocio." -> "negocio"
        match = fuzzy_match(cleaned, list(CATEGORY_VALUE_WORDS))  # typo-tolerant match against the value words
        if match:  # this token is close enough to a known value, misspelled or not
            return CATEGORY_VALUE_WORDS[match]  # return the canonical value it matched
    return None  # no category was named in the text


def references_category_without_value(text: str) -> bool:  # user named the concept ("categoría") but no specific value
    normalized = text.strip().lower()  # normalize for matching
    mentions_concept = any(root in normalized for root in CATEGORY_CONCEPT_ROOTS)  # e.g. "categoría", "categoria"
    return mentions_concept and extract_category_from_text(text) is None  # true only if no concrete value was also given


def parse_date_range(text: str) -> tuple[datetime, datetime] | None:  # map a Spanish time-window phrase to a range
    normalized = text.strip().lower()  # normalize the text for matching
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # current time as naive UTC, matching the DB column type
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)  # midnight of the current day

    if normalized == "4" or "semana pasada" in normalized:  # option 4, or "la semana pasada" = 7 to 14 days ago
        return today_start - timedelta(days=14), today_start - timedelta(days=7)  # return that range
    if normalized == "3" or "esta semana" in normalized:  # option 3, or "esta semana" = the last 7 days
        return today_start - timedelta(days=7), now  # return that range
    if normalized == "2" or "ayer" in normalized:  # option 2, or "ayer" = yesterday
        return today_start - timedelta(days=1), today_start  # return that range
    if normalized == "1" or "hoy" in normalized:  # option 1, or "hoy" = today so far
        return today_start, now  # return that range
    return None  # the text didn't match any known time window


def parse_supabase_timestamp(value: str) -> datetime:  # parse a timestamp string returned by Supabase
    return datetime.fromisoformat(value.replace("Z", "+00:00"))  # normalize a trailing "Z" before parsing


def format_transaction(row: dict) -> str:  # build a human-readable one-line summary of a transaction row
    created_at = row.get("created_at")  # raw created_at timestamp string, if present
    formatted_date = (  # compute the display-formatted date
        parse_supabase_timestamp(created_at).strftime("%d/%m/%Y %H:%M")  # format as DD/MM/YYYY HH:MM
        if created_at  # only if a timestamp is actually present
        else "sin fecha"  # fallback text if no timestamp is available
    )
    return (  # assemble the final summary string
        f"{row.get('description') or '(sin descripción)'} | "  # description, or a placeholder if missing
        f"Monto: {row.get('amount')} | "  # amount
        f"Tipo: {row.get('type')} | "  # transaction type
        f"Fecha y hora: {formatted_date}"  # formatted date and time
    )


def now_iso() -> str:  # produce the current UTC time as a naive ISO 8601 string, matching the DB column type
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()  # strip the timezone before formatting


def format_amount_es(amount: float) -> str:  # format a number using Spanish-style thousands separators
    if amount == int(amount):  # whole number, nothing after the decimal point to show
        return f"{int(amount):,}".replace(",", ".")  # e.g. 10000 -> "10.000"
    return f"{amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")  # e.g. 10000.5 -> "10.000,50"


def build_corrected_description(  # build a plain-Spanish description reflecting a corrected amount
    transaction_type: str | None,
    debtor_name: str | None,
    amount: float,
    creditor_name: str | None = None,
    category: str | None = None,
) -> str:
    formatted_amount = format_amount_es(amount)  # the new amount, formatted with Spanish thousands separators
    if transaction_type == "préstamo":  # loans mention the debtor when one is on record
        if debtor_name:  # a debtor name exists for this transaction
            return f"Préstamo de {formatted_amount} pesos a {debtor_name}."  # e.g. "Préstamo de 10.000 pesos a Bryan."
        return f"Préstamo de {formatted_amount} pesos."  # no debtor name on record
    if transaction_type == "deuda":  # money lent TO the business or the vendor, mentions the creditor when known
        suffix = "" if category == "personal" else " al negocio"  # personal debts don't mention the business
        if creditor_name:  # a creditor name exists for this transaction
            return f"{creditor_name} le prestó {formatted_amount} pesos{suffix}."  # e.g. "Wilmer le prestó 100.000 pesos al negocio."
        return f"Deuda de {formatted_amount} pesos{' con el negocio' if category != 'personal' else ''}."  # no creditor name on record
    if transaction_type == "cobro":  # money collected back from a debtor, mentions the debtor when one is on record
        if debtor_name:  # a debtor name exists for this transaction
            return f"Cobraste {formatted_amount} pesos a {debtor_name}."  # e.g. "Cobraste 50.000 pesos a Bryan."
        return f"Cobraste {formatted_amount} pesos."  # no debtor name on record
    if transaction_type == "pago_deuda":  # money repaid to a creditor, mentions the creditor when one is on record
        if creditor_name:  # a creditor name exists for this transaction
            return f"Pagaste {formatted_amount} pesos a {creditor_name}."  # e.g. "Pagaste 100.000 pesos a Wilmer."
        return f"Pagaste {formatted_amount} pesos."  # no creditor name on record
    if transaction_type == "ingreso":  # income transaction
        return f"Ingreso de {formatted_amount} pesos."  # plain income description
    if transaction_type == "gasto":  # expense transaction
        return f"Gasto de {formatted_amount} pesos."  # plain expense description
    return f"{formatted_amount} pesos."  # fallback if the transaction type is unknown


def normalize_classification_value(value) -> str | None:  # coerce a classification field to a string, whatever JSON type Claude used
    if value is None:  # nothing to normalize
        return None  # keep None as None
    if isinstance(value, float) and value.is_integer():  # Claude may return a whole number as a JSON float
        return str(int(value))  # avoid a stray ".0" suffix, e.g. 400000.0 -> "400000"
    return str(value)  # convert any other type (int, or an already-a-string) to a plain string


def save_transaction(user_id: int, raw_message: str, classification: dict) -> None:  # insert a new transaction row
    supabase.table("transactions").insert(  # build an insert request against the transactions table
        {
            "user_id": str(user_id),  # Telegram user ID, stored as text
            "raw_message": raw_message,  # the original (or combined) message text that produced this transaction
            "amount": classification.get("amount"),  # transaction amount from Claude's classification
            "type": classification.get("type"),  # transaction type from Claude's classification
            "category": classification.get("category"),  # transaction category from Claude's classification
            "recurrence": classification.get("recurrence"),  # transaction recurrence from Claude's classification
            "description": classification.get("description"),  # transaction description from Claude's classification
            "debtor_name": classification.get("debtor_name"),  # debtor name from Claude's classification, if any
            "creditor_name": classification.get("creditor_name"),  # creditor name from Claude's classification, if any
            "is_test": True,  # every user is a tester for now — flips to real-user detection once there's a production vendor
        },
        returning="minimal",  # skip returning the inserted row, since we don't need it back
    ).execute()  # send the insert request to Supabase


def build_new_transaction_confirmation(classification: dict) -> str:  # build the "Anotado ✅ ..." reply text
    if classification.get("type") == "deuda":  # money lent TO the business or the vendor gets its own phrasing
        creditor = classification.get("creditor_name") or "Alguien"  # fall back if no name was extracted
        formatted_amount = format_amount_es(classification.get("amount") or 0)  # Spanish thousands separators
        suffix = "" if classification.get("category") == "personal" else " al negocio"  # personal debts omit this
        return f"Anotado ✅ {creditor} le prestó {formatted_amount} pesos{suffix}."  # deuda confirmation
    if classification.get("type") == "cobro":  # money collected back from a debtor gets its own phrasing
        debtor = classification.get("debtor_name") or "Alguien"  # fall back if no name was extracted
        formatted_amount = format_amount_es(classification.get("amount") or 0)  # Spanish thousands separators
        return f"Anotado ✅ Cobraste {formatted_amount} pesos a {debtor}."  # cobro confirmation
    if classification.get("type") == "pago_deuda":  # money repaid to a creditor gets its own phrasing
        creditor = classification.get("creditor_name") or "Alguien"  # fall back if no name was extracted
        formatted_amount = format_amount_es(classification.get("amount") or 0)  # Spanish thousands separators
        return f"Anotado ✅ Pagaste {formatted_amount} pesos a {creditor}."  # pago_deuda confirmation
    summary = classification.get("description") or ""  # one-line summary of the transaction
    recurrence = classification.get("recurrence")  # the classified recurrence category, if any
    suffix = f" ({recurrence})" if recurrence else ""  # show the recurrence in parentheses when it's known
    return f"Anotado ✅ {summary}{suffix}".strip()  # combine into the final confirmation text


def build_split_classification(  # build a classification dict for one half of an auto-split transaction
    transaction_type: str, amount: float, category: str | None, recurrence: str | None,
    debtor_name: str | None = None, creditor_name: str | None = None,
) -> dict:
    return {
        "type": transaction_type,
        "amount": amount,
        "category": category,
        "recurrence": recurrence,
        "description": build_corrected_description(transaction_type, debtor_name, amount, creditor_name, category),
        "debtor_name": debtor_name,
        "creditor_name": creditor_name,
    }


OVERGUARD_QUESTION_TEMPLATE = (  # asked when a pago_deuda/cobro names someone with no debt on record
    "⚠️ No tengo registrada ninguna deuda con {name}. ¿Quieres registrar este pago de todas formas?\n1. sí\n2. no"
)


async def save_new_transaction(message, user_id: int, raw_text: str, classification: dict) -> None:  # finalize a
    transaction_type = classification.get("type")  # new_transaction classification, splitting overpayments
    amount = classification.get("amount") or 0  # of a pago_deuda/cobro against the existing balance
    category = classification.get("category")
    recurrence = classification.get("recurrence")

    if transaction_type == "pago_deuda":  # repaying a creditor — check it against what's actually owed
        creditor_name = classification.get("creditor_name")
        balance = get_creditor_balance(user_id, creditor_name, category) if creditor_name else 0

        if balance <= 0:  # no debt on record for this creditor/category — confirm before saving anything
            user_states[user_id] = {"state": "OVERGUARD_CONFIRM", "classification": classification, "raw_text": raw_text}
            await message.reply_text(OVERGUARD_QUESTION_TEMPLATE.format(name=creditor_name or "esa persona"))
            return

        if amount > balance:  # paid more than what was owed — split into a debt-closing part and an excess loan
            excess = amount - balance
            save_transaction(user_id, raw_text, build_split_classification("pago_deuda", balance, category, recurrence, creditor_name=creditor_name))
            save_transaction(user_id, raw_text, build_split_classification("préstamo", excess, category, recurrence, debtor_name=creditor_name))
            await message.reply_text(
                f"Anotado ✅ Registré ${format_amount_es(balance)} como pago a {creditor_name}. "
                f"Los ${format_amount_es(excess)} restantes quedaron como un préstamo a tu favor — "
                f"ahora {creditor_name} te debe ese dinero."
            )
            return

        save_transaction(user_id, raw_text, classification)  # amount <= balance, save normally
        await message.reply_text(build_new_transaction_confirmation(classification))
        return

    if transaction_type == "cobro":  # collecting from a debtor — check it against what's actually owed
        debtor_name = classification.get("debtor_name")
        balance = get_debtor_balance(user_id, debtor_name) if debtor_name else 0

        if balance <= 0:  # no debt on record for this debtor — confirm before saving anything
            user_states[user_id] = {"state": "OVERGUARD_CONFIRM", "classification": classification, "raw_text": raw_text}
            await message.reply_text(OVERGUARD_QUESTION_TEMPLATE.format(name=debtor_name or "esa persona"))
            return

        if amount > balance:  # collected more than what was owed — split into a debt-closing part and an excess debt
            excess = amount - balance
            save_transaction(user_id, raw_text, build_split_classification("cobro", balance, category, recurrence, debtor_name=debtor_name))
            save_transaction(user_id, raw_text, build_split_classification("deuda", excess, category, recurrence, creditor_name=debtor_name))
            await message.reply_text(
                f"Anotado ✅ Registré ${format_amount_es(balance)} como cobro de {debtor_name}. "
                f"Los ${format_amount_es(excess)} restantes quedaron como una deuda a su favor — "
                f"ahora le debes ese dinero a {debtor_name}."
            )
            return

        save_transaction(user_id, raw_text, classification)  # amount <= balance, save normally
        await message.reply_text(build_new_transaction_confirmation(classification))
        return

    save_transaction(user_id, raw_text, classification)  # any other type, no balance to check against
    await message.reply_text(build_new_transaction_confirmation(classification))


async def handle_state_overguard_confirm(message, user_id: int, text: str, state_info: dict) -> None:  # handle the
    normalized = text.strip().lower()  # sí/no reply to the "no debt on record" warning
    classification = state_info.get("classification")  # the originally classified transaction, unmodified
    raw_text = state_info.get("raw_text")  # the message that produced it

    if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # vendor confirmed: save it anyway, as-is
        user_states.pop(user_id, None)  # return to State A
        save_transaction(user_id, raw_text, classification)  # no balance to split against, save the full amount
        await message.reply_text(build_new_transaction_confirmation(classification))
        return

    if normalized == "2" or normalized in NEGATIVE_REPLIES:  # vendor declined: discard it
        user_states.pop(user_id, None)  # return to State A
        await message.reply_text("OK ✅ No se registró nada.")
        return

    await message.reply_text("No entendí. Responde con el número o la palabra:\n1. sí\n2. no")  # unrecognized reply


def description_matches_hint(description: str | None, hint: str) -> bool:  # typo-tolerant description match
    if not description:  # nothing to compare against
        return False  # can't match an empty description
    normalized_description = description.lower()  # normalize for matching
    normalized_hint = hint.strip().lower()  # normalize the hint the same way
    if normalized_hint in normalized_description:  # fast path: exact substring, no typo (handles most real cases)
        return True  # already an exact match, no need for fuzzy comparison
    words = normalized_description.split()  # compare the hint against each individual word in the description
    return fuzzy_match(normalized_hint, words) is not None  # true if any word is close enough, typos included


def search_transactions(  # search for transactions matching a date range and an optional correction hint
    user_id: int, start: datetime, end: datetime, hint: str | None
) -> list[dict]:
    query = (  # start building the Supabase query
        supabase.table("transactions")  # target the transactions table
        .select("*")  # select every column
        .eq("user_id", str(user_id))  # restrict to this Telegram user
        .eq("status", "activa")  # exclude voided ("anulada") transactions
        .gte("created_at", start.isoformat())  # only transactions on or after the start of the range
        .lt("created_at", end.isoformat())  # only transactions strictly before the end of the range
    )

    parsed_amount = try_parse_amount(hint) if hint else None  # check whether the hint looks like a numeric amount
    if parsed_amount is not None:  # the hint is a number, filter by exact amount at the database level
        query = query.eq("amount", parsed_amount)  # amounts have no "spelling", so an exact match is correct here

    result = query.order("created_at", desc=True).execute()  # run the query, newest transactions first
    rows = result.data or []  # the matching rows, or an empty list if none

    if hint and parsed_amount is None:  # the hint is descriptive text, apply typo-tolerant filtering in Python
        rows = [row for row in rows if description_matches_hint(row.get("description"), hint)]  # keep close matches

    return rows  # return the matching rows, or an empty list if none


def get_most_recent_transaction(user_id: int) -> dict | None:  # fetch this user's single most recent active transaction
    result = (  # look up the most recent active transaction
        supabase.table("transactions")  # target the transactions table
        .select("*")  # select every column
        .eq("user_id", str(user_id))  # restrict to this Telegram user
        .eq("status", "activa")  # only consider active (not already voided) transactions
        .order("created_at", desc=True)  # newest first
        .limit(1)  # only need the single most recent row
        .execute()  # run the query
    )
    rows = result.data or []  # the matching rows, or an empty list if none
    return rows[0] if rows else None  # return the row, or None if this user has no active transaction


def find_transactions_by_keyword(user_id: int, keyword: str) -> list[dict]:  # typo-tolerant search, newest first
    result = (  # look up every active transaction for this user, regardless of date
        supabase.table("transactions")  # target the transactions table
        .select("*")  # select every column
        .eq("user_id", str(user_id))  # restrict to this Telegram user
        .eq("status", "activa")  # only consider active (not already voided) transactions
        .order("created_at", desc=True)  # newest first
        .execute()  # run the query
    )
    rows = result.data or []  # every active transaction for this user
    return [row for row in rows if description_matches_hint(row.get("description"), keyword)]  # keep close matches


def calculate_report_totals(rows: list[dict]) -> dict:  # aggregate a list of transaction rows into report totals
    def sum_by_type(transaction_type: str) -> float:  # sum the amount field across rows matching a given type
        return sum(row.get("amount") or 0 for row in rows if row.get("type") == transaction_type)  # total for that type

    ingresos = sum_by_type("ingreso")  # total income
    gastos = sum_by_type("gasto")  # total expenses
    prestamos_dados = sum_by_type("préstamo")  # total lent outward by the vendor
    deudas_recibidas = sum_by_type("deuda")  # total borrowed by the vendor
    cobros = sum_by_type("cobro")  # total collected back from debtors
    pagos_deuda = sum_by_type("pago_deuda")  # total repaid to creditors
    bolsillo = ingresos + deudas_recibidas + cobros - gastos - prestamos_dados - pagos_deuda  # net cash on hand

    return {  # every total the report needs, in one dictionary
        "ingresos": ingresos,
        "gastos": gastos,
        "prestamos_dados": prestamos_dados,
        "deudas_recibidas": deudas_recibidas,
        "cobros": cobros,
        "pagos_deuda": pagos_deuda,
        "bolsillo": bolsillo,
    }


def get_period_report(user_id: int, start_date, end_date) -> dict:  # income/expense summary over an inclusive date range
    # start_date/end_date are LOCAL (Europe/Amsterdam) calendar dates; created_at is stored as naive UTC, so local
    # midnight must be converted to UTC before comparing — a fixed UTC-midnight boundary would be off by 1-2 hours.
    start_local = datetime.combine(start_date, datetime.min.time(), tzinfo=LOCAL_TZ)  # local midnight, start_date
    end_local = datetime.combine(end_date, datetime.min.time(), tzinfo=LOCAL_TZ) + timedelta(days=1)  # local midnight, day after end_date
    start = start_local.astimezone(timezone.utc).replace(tzinfo=None)  # equivalent naive UTC instant
    end = end_local.astimezone(timezone.utc).replace(tzinfo=None)  # equivalent naive UTC instant

    logger.info(  # debug: the exact filter being sent to Supabase, in both UTC and local terms
        "get_period_report user_id=%s local_range=[%s 00:00, %s 00:00) Europe/Amsterdam -> "
        "SQL-equivalent: created_at >= '%s' AND created_at < '%s' (UTC)",
        user_id, start_date.isoformat(), (end_date + timedelta(days=1)).isoformat(), start.isoformat(), end.isoformat(),
    )

    result = (  # look up every active transaction created within the range
        supabase.table("transactions")  # target the transactions table
        .select("*")  # select every column
        .eq("user_id", str(user_id))  # restrict to this Telegram user
        .eq("status", "activa")  # exclude voided ("anulada") transactions
        .gte("created_at", start.isoformat())  # only transactions on or after the start of the range
        .lt("created_at", end.isoformat())  # only transactions strictly before the end of the range
        .execute()  # run the query
    )
    rows = result.data or []  # the matching rows, or an empty list if none
    return calculate_report_totals(rows)  # aggregate them into the report totals


def get_daily_report(user_id: int, date) -> dict:  # income/expense summary for a single calendar day
    return get_period_report(user_id, date, date)  # a single day is just a one-day-long period


def get_debtor_list(user_id: int) -> tuple[list[dict], list[dict]]:  # (who still owes you, who you overpaid)
    result = (  # look up every active préstamo/cobro transaction for this user
        supabase.table("transactions")  # target the transactions table
        .select("*")  # select every column
        .eq("user_id", str(user_id))  # restrict to this Telegram user
        .eq("status", "activa")  # exclude voided ("anulada") transactions
        .in_("type", ["préstamo", "cobro"])  # only the two types that affect a debtor's balance
        .execute()  # run the query
    )
    rows = result.data or []  # the matching rows, or an empty list if none

    balances: dict[str, float] = {}  # running balance per debtor name
    for row in rows:  # walk every matching transaction
        debtor_name = row.get("debtor_name")  # who this row is about
        if not debtor_name:  # no debtor recorded on this row, nothing to attribute it to
            continue  # skip it
        amount = row.get("amount") or 0  # the transaction amount, defaulting to 0 if missing
        if row.get("type") == "préstamo":  # money lent out increases what the debtor owes
            balances[debtor_name] = balances.get(debtor_name, 0) + amount  # add to their balance
        elif row.get("type") == "cobro":  # money collected back decreases what the debtor owes
            balances[debtor_name] = balances.get(debtor_name, 0) - amount  # subtract from their balance

    debtors = sorted(  # debtors who still owe a positive balance, highest first
        [{"debtor_name": name, "balance": balance} for name, balance in balances.items() if balance > 0],
        key=lambda debtor: debtor["balance"], reverse=True,
    )
    overpaid = sorted(  # debtors the vendor overcollected from (negative balance), highest excess first
        [{"debtor_name": name, "excess": -balance} for name, balance in balances.items() if balance < 0],
        key=lambda debtor: debtor["excess"], reverse=True,
    )
    return debtors, overpaid


def get_creditor_list(user_id: int) -> tuple[list[dict], list[dict]]:  # (who you still owe, who you overpaid)
    result = (  # look up every active deuda/pago_deuda transaction for this user
        supabase.table("transactions")  # target the transactions table
        .select("*")  # select every column
        .eq("user_id", str(user_id))  # restrict to this Telegram user
        .eq("status", "activa")  # exclude voided ("anulada") transactions
        .in_("type", ["deuda", "pago_deuda"])  # only the two types that affect a creditor's balance
        .execute()  # run the query
    )
    rows = result.data or []  # the matching rows, or an empty list if none

    balances: dict[str, dict[str, float]] = {}  # running {"negocio": x, "personal": y} balance per creditor name
    for row in rows:  # walk every matching transaction
        creditor_name = row.get("creditor_name")  # who this row is about
        category = row.get("category")  # "negocio" or "personal"
        if not creditor_name or category not in ("negocio", "personal"):  # can't attribute this row to a balance
            continue  # skip it
        amount = row.get("amount") or 0  # the transaction amount, defaulting to 0 if missing
        entry = balances.setdefault(creditor_name, {"negocio": 0, "personal": 0})  # this creditor's running balances
        if row.get("type") == "deuda":  # money borrowed increases what's owed to the creditor
            entry[category] += amount  # add to the matching category balance
        elif row.get("type") == "pago_deuda":  # money repaid decreases what's owed to the creditor
            entry[category] -= amount  # subtract from the matching category balance

    creditors = []  # creditors still owed a positive total balance
    overpaid = []  # creditors the vendor overpaid (negative total balance)
    for name, entry in balances.items():  # walk every creditor seen above
        business_balance = entry["negocio"]  # what's owed from business-category debts
        personal_balance = entry["personal"]  # what's owed from personal-category debts
        total_balance = business_balance + personal_balance  # combined balance across both categories
        if total_balance > 0:  # still owed to this creditor
            creditors.append({  # one entry per creditor, with the business/personal breakdown
                "creditor_name": name,
                "business_balance": business_balance,
                "personal_balance": personal_balance,
                "total_balance": total_balance,
            })
        elif total_balance < 0:  # overpaid — the creditor now owes the vendor the excess
            overpaid.append({"creditor_name": name, "excess": -total_balance})

    creditors.sort(key=lambda creditor: creditor["total_balance"], reverse=True)  # highest total owed first
    overpaid.sort(key=lambda creditor: creditor["excess"], reverse=True)  # highest excess first
    return creditors, overpaid


def get_creditor_balance(user_id: int, creditor_name: str, category: str | None) -> float:  # one creditor's balance
    result = (  # in one category — deuda minus pago_deuda, restricted to that creditor and category
        supabase.table("transactions")
        .select("type,amount")
        .eq("user_id", str(user_id))
        .eq("status", "activa")
        .eq("creditor_name", creditor_name)
        .eq("category", category)
        .in_("type", ["deuda", "pago_deuda"])
        .execute()
    )
    rows = result.data or []  # the matching rows, or an empty list if none
    return sum(  # deuda increases the balance owed, pago_deuda decreases it
        (row.get("amount") or 0) if row.get("type") == "deuda" else -(row.get("amount") or 0) for row in rows
    )


def get_debtor_balance(user_id: int, debtor_name: str) -> float:  # one debtor's balance across both categories —
    result = (  # préstamo minus cobro, restricted to that debtor (matches get_debtor_list's category-agnostic model)
        supabase.table("transactions")
        .select("type,amount")
        .eq("user_id", str(user_id))
        .eq("status", "activa")
        .eq("debtor_name", debtor_name)
        .in_("type", ["préstamo", "cobro"])
        .execute()
    )
    rows = result.data or []  # the matching rows, or an empty list if none
    return sum(  # préstamo increases the balance owed, cobro decreases it
        (row.get("amount") or 0) if row.get("type") == "préstamo" else -(row.get("amount") or 0) for row in rows
    )


def resolve_shortcut_candidates(user_id: int, keyword: str | None) -> list[dict]:  # candidates for the shortcut flow
    candidates = find_transactions_by_keyword(user_id, keyword) if keyword else []  # typo-tolerant search, if named
    if not candidates:  # no keyword was given, or nothing matched it even fuzzily
        most_recent = get_most_recent_transaction(user_id)  # fall back to the most recent active transaction
        candidates = [most_recent] if most_recent else []  # wrap it as a single-item list, or none if none exists
    return candidates  # zero, one, or many candidate transactions


def build_multi_match_prompt(candidates: list[dict], verb: str) -> str:  # numbered-list prompt for several candidates
    lines = [f"{i}. {format_transaction(row)}" for i, row in enumerate(candidates, start=1)]  # number each candidate
    return "\n".join(lines) + f"\n¿Cuál es la que quieres {verb}? Responde con el número"  # ask which one


async def cancel_last_transaction(message) -> None:  # void the most recent active transaction for this user
    telegram_user_id = message.from_user.id  # Telegram user ID of the sender
    result = (  # look up the most recent active transaction
        supabase.table("transactions")  # target the transactions table
        .select("*")  # select every column
        .eq("user_id", str(telegram_user_id))  # restrict to this Telegram user
        .eq("status", "activa")  # only consider active (not already voided) transactions
        .order("created_at", desc=True)  # newest first
        .limit(1)  # only need the single most recent row
        .execute()  # run the query
    )
    rows = result.data or []  # the matching rows, or an empty list if none
    if not rows:  # no active transaction exists for this user
        await message.reply_text("No encontré ninguna transacción activa para anular.")  # nothing to void
        return  # stop here

    row = rows[0]  # the most recent active transaction
    supabase.table("transactions").update(  # void that transaction
        {"status": "anulada", "updated_at": now_iso()}  # mark it voided and refresh the update timestamp
    ).eq("id", row["id"]).execute()  # apply the update to that specific row

    await message.reply_text(  # confirm the transaction was voided, including its details
        f"Listo, anulé la última transacción ✅ {format_transaction(row)}"
    )


def save_correction(match: dict, field_updates: dict, raw_message: str) -> tuple[dict, dict]:  # apply given fields
    updates = {  # fields to update on the transaction
        "updated_at": now_iso(),  # always refresh the update timestamp
        "raw_message": raw_message.strip(),  # always record the correction text as the new raw message
        **field_updates,  # the specific field(s) this correction changes (amount, description, recurrence, category)
    }
    supabase.table("transactions").update(updates).eq("id", match["id"]).execute()  # apply the correction
    updated_row = {**match, **updates}  # local copy of the row reflecting the applied update, for the reply message
    return updated_row, updates  # return both the full row and just the changed fields, for building the reply


def apply_correction(match: dict, new_value_text: str) -> tuple[dict, dict]:  # auto-detect which field changed, apply it
    parsed_amount = extract_amount_from_text(new_value_text)  # look for a numeric amount anywhere among the words
    parsed_recurrence = None if parsed_amount is not None else extract_recurrence_from_text(new_value_text)  # or a word
    parsed_category = (  # or a category word, only checked once amount and recurrence are both ruled out
        None if parsed_amount is not None or parsed_recurrence is not None else extract_category_from_text(new_value_text)
    )

    if parsed_amount is not None:  # a number was found among the words (e.g. "15000" in "15000 al bryan")
        field_updates = {  # rebuild the description to reflect the new amount
            "amount": parsed_amount,
            "description": build_corrected_description(
                match.get("type"), match.get("debtor_name"), parsed_amount,
                match.get("creditor_name"), match.get("category"),
            ),
        }
    elif parsed_recurrence is not None:  # no amount, but the user named a recurrence category (e.g. "es recurrente")
        field_updates = {"recurrence": parsed_recurrence}  # update only the recurrence
    elif parsed_category is not None:  # no amount or recurrence, but the user named a category (e.g. "es personal")
        field_updates = {"category": parsed_category}  # update only the category
    else:  # nothing recognizable found anywhere in the text
        field_updates = {"description": new_value_text.strip()}  # treat the whole text as a corrected description

    return save_correction(match, field_updates, new_value_text)  # apply it and return the updated row + changed fields


def build_correction_confirmation(updated_row: dict, updates: dict) -> str:  # build the "Corregido ✅ ..." reply text
    detail = format_transaction(updated_row)  # the standard transaction summary (description, amount, type, date)
    if "recurrence" in updates:  # this correction specifically changed the recurrence category
        detail += f" | Frecuencia: {updates['recurrence']}"  # surface the new recurrence so the change is visible
    if "category" in updates:  # this correction specifically changed the category
        detail += f" | Categoría: {updates['category']}"  # surface the new category so the change is visible
    return f"Corregido ✅ {detail}"  # combine into the final confirmation text


REPORT_LABELS = [  # (totals key, display label) pairs, in the order they should appear in a period report
    ("ingresos", "💵 Ingresos:"),
    ("gastos", "💸 Gastos:"),
    ("prestamos_dados", "📤 Préstamos dados:"),
    ("deudas_recibidas", "📥 Deudas recibidas:"),
    ("cobros", "📥 Cobros recibidos:"),
    ("pagos_deuda", "💳 Pagos de deuda:"),
]

REPORT_FOLLOWUP_MENU = (  # shown after every period report, offering to drill into another period
    "¿Quieres ver más? Responde con el número o la palabra:\n"
    "1. Esta semana\n2. Este mes\n3. Desde otra fecha\n4. No, gracias"
)

R_DATE_QUESTION = "¿Desde cuándo? Por ejemplo: 'el lunes', 'el 10 de julio', 'hace 2 semanas'"  # asked in state R_DATE

R_DATE_MANUAL_QUESTION = (  # asked when Claude couldn't parse the free-text date expression
    "No entendí la fecha. ¿Puedes escribirla así? DD/MM/YYYY (por ejemplo: 10/07/2026). Así es más rápido para los dos 😊"
)

R_DATE_MANUAL_RETRY = "Formato no válido. Escribe la fecha así: DD/MM/YYYY (por ejemplo: 10/07/2026)."  # retry text

DEBTORS_FOLLOWUP_QUESTION = (  # asked after showing the debtor list
    "¿Quieres ver la lista de acreedores también? Responde con el número o la palabra:\n1. sí\n2. no"
)


def format_period_report(totals: dict, header: str) -> str:  # build the numbered income/expense report message
    lines = [f"📊 {header}", ""]  # header line, then a blank line
    for key, label in REPORT_LABELS:  # walk every possible line item, in display order
        amount = totals.get(key) or 0  # this line item's total, defaulting to 0 if missing
        if amount > 0:  # only show line items with a positive total, to keep the report clean
            lines.append(f"{label.ljust(26)}${format_amount_es(amount)}")  # e.g. "💵 Ingresos:              $10.000"
    lines.append("")  # blank line before the bottom-line summary
    lines.append(f"💰 En tu bolsillo deberías tener: ${format_amount_es(totals.get('bolsillo') or 0)}")  # always shown
    lines.append("")  # blank line before the follow-up menu
    lines.append(REPORT_FOLLOWUP_MENU)  # offer to drill into another period
    return "\n".join(lines)  # combine into the final report text


def format_debtor_list(debtors: list[dict], overpaid: list[dict]) -> str:  # build the "who owes/is owed" report
    lines = ["📋 Personas que te deben:", ""]  # header, then a blank line
    if debtors:  # at least one debtor with a positive balance
        lines += [f"{i}. {d['debtor_name']}: ${format_amount_es(d['balance'])}" for i, d in enumerate(debtors, 1)]
    else:  # nobody currently owes the vendor
        lines.append("(ninguno)")  # explicit empty-state line

    if overpaid:  # at least one debtor the vendor overcollected from
        lines += ["", "💳 Pagaste de más a:"]  # section omitted entirely when there are no overpayments
        lines += [f"{i}. {o['debtor_name']}: ${format_amount_es(o['excess'])}" for i, o in enumerate(overpaid, 1)]

    total = sum(d["balance"] for d in debtors)  # combined balance across debtors only, excluding overpaid
    lines += ["", f"Total por cobrar: ${format_amount_es(total)}"]
    return "\n".join(lines) + f"\n\n{DEBTORS_FOLLOWUP_QUESTION}"  # always offer to also show the creditor list


def format_creditor_list(creditors: list[dict], overpaid: list[dict]) -> str:  # build the "what the vendor owes" report
    negocio = [c for c in creditors if c["business_balance"] > 0]  # creditors owed money from the business
    personal = [c for c in creditors if c["personal_balance"] > 0]  # creditors owed money personally

    lines = ["📋 Lo que debes:", "", "🏢 Negocio:"]  # header, then the negocio section
    if negocio:  # at least one business creditor
        lines += [f"{i}. {c['creditor_name']}: ${format_amount_es(c['business_balance'])}" for i, c in enumerate(negocio, 1)]
    else:  # nothing owed from the business side
        lines.append("(ninguno)")  # explicit empty-state line

    lines += ["", "👤 Personal:"]  # the personal section
    if personal:  # at least one personal creditor
        lines += [f"{i}. {c['creditor_name']}: ${format_amount_es(c['personal_balance'])}" for i, c in enumerate(personal, 1)]
    else:  # nothing owed personally
        lines.append("(ninguno)")  # explicit empty-state line

    if overpaid:  # at least one creditor the vendor overpaid
        lines += ["", "📤 Te deben por pagos en exceso:"]  # section omitted entirely when there are no overpayments
        lines += [f"{i}. {o['creditor_name']}: ${format_amount_es(o['excess'])}" for i, o in enumerate(overpaid, 1)]

    total_negocio = sum(c["business_balance"] for c in negocio)  # combined business-side balance
    total_personal = sum(c["personal_balance"] for c in personal)  # combined personal-side balance
    lines += [
        "",
        f"Total negocio: ${format_amount_es(total_negocio)}",
        f"Total personal: ${format_amount_es(total_personal)}",
        f"Total general: ${format_amount_es(total_negocio + total_personal)}",  # creditors only, excludes overpaid
    ]
    return "\n".join(lines)  # combine into the final report text


def week_start(today: date) -> date:  # the Monday of the week containing today
    return today - timedelta(days=today.weekday())  # weekday() is 0 for Monday


def parse_date_expression(text: str) -> date | None:  # ask Claude to resolve a free-text Spanish date expression
    today = today_local()  # reference point for relative expressions like "hace 2 semanas"
    response = claude.messages.create(  # send the parsing request to the Claude API
        model=CLAUDE_MODEL,  # use the configured Claude model
        max_tokens=100,  # this reply is just a short JSON object
        system=(  # instructions for this one-off parsing call
            f"Today is {today.isoformat()} (YYYY-MM-DD). Parse the user's Spanish date expression into the "
            'start date they mean. Return ONLY a JSON object: {"date": "YYYY-MM-DD"}, or {"date": null} if '
            "you cannot confidently determine a date."
        ),
        messages=[{"role": "user", "content": text}],  # send the user's date expression as the only turn
    )
    raw_text = next(block.text for block in response.content if block.type == "text")  # pull out the text block
    date_str = parse_classification(raw_text).get("date")  # reuse the same defensive JSON parser as classification
    if not date_str:  # Claude could not confidently parse a date
        return None  # signal failure to the caller
    try:  # the date string should be ISO format, but validate defensively
        return date.fromisoformat(date_str)  # parsed date
    except ValueError:  # Claude returned something that isn't a valid ISO date
        return None  # signal failure to the caller


async def send_period_report(message, user_id: int, start: date, end: date, header: str) -> None:  # show a report,
    totals = get_period_report(user_id, start, end)  # then enter the follow-up state
    await message.reply_text(format_period_report(totals, header))  # send the formatted report
    user_states[user_id] = {"state": "R"}  # await a follow-up choice (another period, or "no, gracias")


async def send_debtor_list(message, user_id: int) -> None:  # show the debtor list, then offer the creditor list
    debtors, overpaid = get_debtor_list(user_id)  # who owes the vendor, and who the vendor overcollected from
    await message.reply_text(format_debtor_list(debtors, overpaid))  # send the formatted list
    user_states[user_id] = {"state": "R_DEBTORS_FOLLOWUP"}  # await sí/no on also showing the creditor list


async def send_creditor_list(message, user_id: int) -> None:  # show the creditor list; nothing to follow up with
    creditors, overpaid = get_creditor_list(user_id)  # who the vendor owes, and who the vendor overpaid
    await message.reply_text(format_creditor_list(creditors, overpaid))  # send the formatted list
    user_states.pop(user_id, None)  # this report has no further follow-up, return to State A


async def handle_custom_date_query(message, user_id: int, date_hint: str | None) -> None:  # "custom" query_period
    today = today_local()  # end of the report range is always today
    parsed = parse_date_expression(date_hint) if date_hint else None  # try to resolve the hint to a start date
    if parsed is not None:  # Claude confidently parsed a start date
        header = f"Resumen desde {parsed:%d/%m/%Y} al {today:%d/%m/%Y}"  # date-range header
        await send_period_report(message, user_id, parsed, today, header)  # show the report
    else:  # Claude couldn't parse the hint, fall back to asking for a manual date
        user_states[user_id] = {"state": "R_DATE_MANUAL"}  # await a DD/MM/YYYY reply
        await message.reply_text(R_DATE_MANUAL_QUESTION)  # ask for the manual format


async def handle_state_a(message, user_id: int, text: str) -> None:  # handle a message with no flow in progress
    classification = classify_message(text)  # ask Claude to classify the message
    logger.info("Classification for %r: %s", text, json.dumps(classification, ensure_ascii=False))  # debug visibility
    intent = classification.get("intent")  # extract the classified intent

    if intent == "new_transaction":  # the user is registering a new transaction
        if classification.get("needs_clarification"):  # Claude needs more information before saving
            user_states[user_id] = {"state": "B", "original_text": text}  # remember the text, move to State B
            question = (  # choose the clarification question to ask
                classification.get("clarification_question")  # Claude's suggested question
                or "¿Puedes dar más detalles?"  # generic fallback if none was provided
            )
            await message.reply_text(question)  # ask the clarification question
        else:  # the transaction is fully specified
            await save_new_transaction(message, user_id, text, classification)  # save it, splitting overpayments

    elif intent == "correction":  # the user wants to fix a past transaction
        hint = normalize_classification_value(classification.get("correction_hint"))  # the OLD field value, if named
        new_value = normalize_classification_value(classification.get("new_value"))  # the NEW value, coerced to a string
        keyword = normalize_classification_value(  # a description keyword identifying WHICH transaction, if named
            classification.get("transaction_keyword")
        )
        candidates = resolve_shortcut_candidates(user_id, keyword)  # typo-tolerant search, falling back to most recent

        if len(candidates) == 1 and new_value:  # exactly one candidate, and Claude already knows the new value too
            candidate = candidates[0]  # the single candidate transaction
            user_states[user_id] = {  # go straight to State D with the candidate and the proposed new value
                "state": "D",  # State D: awaiting confirmation
                "matches": [candidate],  # only one candidate to confirm
                "date_range": None,  # no date range has been resolved yet
                "correction_hint": hint,  # keep the original hint in case the user rejects this candidate
                "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                "new_value": new_value,  # the new value to apply immediately if the user confirms both
                "action": "correct",  # this confirmation leads to a correction, not a deletion
            }
            await message.reply_text(  # show the transaction, the proposed new value, and ask for confirmation
                f"¿Te refieres a esta transacción? {format_transaction(candidate)}\n"
                f"¿Y el nuevo valor es {new_value}? Responde con el número o la palabra:\n"
                "1. sí\n2. otro valor\n3. otra"
            )
        elif len(candidates) == 1:  # exactly one candidate, but no new value was extracted yet
            candidate = candidates[0]  # the single candidate transaction
            user_states[user_id] = {  # go straight to State D with this transaction as the candidate
                "state": "D",  # State D: awaiting confirmation
                "matches": [candidate],  # only one candidate to confirm
                "date_range": None,  # no date range has been resolved yet
                "correction_hint": hint,  # keep the original hint in case the user rejects this candidate
                "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                "action": "correct",  # this confirmation leads to a correction, not a deletion
            }
            await message.reply_text(  # show the transaction and ask for confirmation
                f"¿Te refieres a esta transacción? {format_transaction(candidate)}\n{YES_NO_MENU}"
            )
        elif candidates:  # multiple candidates matched (typos included), ask which one instead of guessing
            user_states[user_id] = {  # go straight to State D with every candidate to choose from
                "state": "D",  # State D: awaiting confirmation
                "matches": candidates,  # every candidate the fuzzy search found
                "date_range": None,  # no date range has been resolved yet
                "correction_hint": hint,  # keep the original hint in case all candidates are rejected
                "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                "action": "correct",  # this confirmation leads to a correction, not a deletion
            }
            await message.reply_text(build_multi_match_prompt(candidates, "corregir"))  # show the list, ask which one
        else:  # this user has no transactions at all yet, fall back to the time-window question
            user_states[user_id] = {  # move to State C to determine which transaction to correct
                "state": "C",  # State C: awaiting the time window
                "correction_hint": hint,  # the OLD value/description to search for
                "date_range": None,  # no date range chosen yet
                "action": "correct",  # this search leads to a correction, not a deletion
            }
            await message.reply_text(TIME_WINDOW_QUESTION)  # ask the user when the transaction happened

    elif intent == "query":  # the user is asking for a balance or report
        query_period = classification.get("query_period")  # which report Claude classified this query as
        today = today_local()  # reference date for "hoy"/"semana"/"mes"

        if query_period == "semana":  # this week, Monday through today
            start = week_start(today)  # Monday of the current week
            await send_period_report(message, user_id, start, today, f"Resumen de esta semana — {start:%d/%m/%Y} al {today:%d/%m/%Y}")
        elif query_period == "mes":  # this month, the 1st through today
            start = today.replace(day=1)  # first day of the current month
            await send_period_report(message, user_id, start, today, f"Resumen de este mes — {start:%d/%m/%Y} al {today:%d/%m/%Y}")
        elif query_period == "deudores":  # who owes the vendor money
            await send_debtor_list(message, user_id)  # show the debtor list
        elif query_period == "acreedores":  # who the vendor owes money to
            await send_creditor_list(message, user_id)  # show the creditor list
        elif query_period == "custom":  # the user named a specific starting point
            await handle_custom_date_query(message, user_id, classification.get("query_date_hint"))  # resolve it
        elif query_period == "hoy":  # today's status
            await send_period_report(message, user_id, today, today, f"Resumen de hoy — {today:%d/%m/%Y}")
        else:  # query_period is null — ambiguous, ask the user to be specific instead of guessing
            await message.reply_text(
                "No entendí qué quieres consultar. ¿Hoy, esta semana, este mes, deudores, o acreedores?"
            )

    elif intent == "cancellation":  # the user wants to undo the last transaction, or delete a specific one
        keyword = normalize_classification_value(  # a description keyword identifying WHICH transaction, if named
            classification.get("transaction_keyword")
        )
        if not keyword:  # no specific transaction named, undo the most recent one (fast path, no confirmation)
            await cancel_last_transaction(message)  # void the most recent active transaction
        else:  # a specific transaction was named, find it (typos included) and confirm before deleting it
            candidates = resolve_shortcut_candidates(user_id, keyword)  # typo-tolerant search, falling back to most recent
            if len(candidates) == 1:  # exactly one candidate
                candidate = candidates[0]  # the single candidate transaction
                user_states[user_id] = {  # go straight to State D with this transaction as the candidate
                    "state": "D",  # State D: awaiting confirmation
                    "matches": [candidate],  # only one candidate to confirm
                    "date_range": None,  # no date range has been resolved yet
                    "correction_hint": keyword,  # keep the keyword in case the user rejects this candidate
                    "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                    "action": "delete",  # this confirmation leads to a deletion, not a correction
                }
                await message.reply_text(  # show the transaction and ask for confirmation before deleting it
                    f"¿Quieres eliminar esta transacción? {format_transaction(candidate)}\n{YES_NO_MENU}"
                )
            elif candidates:  # multiple candidates matched (typos included), ask which one instead of guessing
                user_states[user_id] = {  # go straight to State D with every candidate to choose from
                    "state": "D",  # State D: awaiting confirmation
                    "matches": candidates,  # every candidate the fuzzy search found
                    "date_range": None,  # no date range has been resolved yet
                    "correction_hint": keyword,  # keep the keyword in case all candidates are rejected
                    "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                    "action": "delete",  # this confirmation leads to a deletion, not a correction
                }
                await message.reply_text(build_multi_match_prompt(candidates, "eliminar"))  # show the list, ask which
            else:  # this user has no transactions at all
                await message.reply_text("No encontré ninguna transacción activa para eliminar.")

    else:  # intent is "confirmation" (or anything unexpected) with nothing pending to confirm
        await message.reply_text("No tengo ninguna corrección pendiente de confirmar.")  # nothing to confirm


async def handle_state_b(message, user_id: int, text: str, state_info: dict) -> None:  # handle a clarification reply
    original = state_info.get("original_text", "")  # the message that originally triggered the clarification
    combined = (  # build the combined message to re-classify
        f"Mensaje original: {original}\n"  # include the original message
        f"Respuesta del usuario a la pregunta de aclaración: {text}"  # include the user's clarifying reply
    )
    classification = classify_message(combined)  # re-classify using the combined context
    intent = classification.get("intent")  # extract the classified intent

    if intent == "cancellation":  # the user cancelled instead of clarifying
        user_states.pop(user_id, None)  # abandon the pending transaction, return this user to State A
        await message.reply_text("Listo, no anoto nada ✅")  # confirm nothing was saved
        return  # stop processing this message here

    if classification.get("needs_clarification"):  # Claude still needs more information
        user_states[user_id] = {"state": "B", "original_text": combined}  # stay in State B with updated context
        question = (  # choose the next clarification question to ask
            classification.get("clarification_question")  # Claude's suggested question
            or "¿Puedes dar más detalles?"  # generic fallback if none was provided
        )
        await message.reply_text(question)  # ask the clarification question
        return  # stop processing this message here

    user_states.pop(user_id, None)  # clarification flow is complete; save_new_transaction may set a fresh state below
    await save_new_transaction(message, user_id, combined, classification)  # save it, splitting overpayments


async def handle_state_c(message, user_id: int, text: str, state_info: dict) -> None:  # handle a time-window reply
    if is_cancellation(text):  # the user wants to abandon the correction entirely
        user_states.pop(user_id, None)  # return this user to State A
        await message.reply_text("Listo, no cambio nada ✅")  # confirm nothing was changed
        return  # stop processing this message here

    date_range = state_info.get("date_range")  # the previously resolved date range, if any
    hint = state_info.get("correction_hint")  # the current correction hint (the old value to search for)

    if date_range is None:  # we don't have a date range yet, so this reply should specify one
        parsed_range = parse_date_range(text)  # try to parse a time-window phrase from the reply
        if parsed_range is None:  # the reply didn't match a known time window
            await message.reply_text(TIME_WINDOW_RETRY)  # ask the user to reply with one of the supported options
            return  # stay in State C (state is left unchanged)
        date_range = parsed_range  # use the newly parsed range
    else:  # we already have a date range, so this reply is a new description of the transaction
        hint = text  # replace the hint with the user's new description

    start, end = date_range  # unpack the resolved date range
    matches = search_transactions(user_id, start, end, hint)  # search Supabase for matching transactions

    action = state_info.get("action", "correct")  # whether this search leads to a correction or a deletion
    verb = "corregir" if action == "correct" else "eliminar"  # the verb to use in the confirmation prompts below

    if not matches:  # no transactions matched the range and hint
        user_states[user_id] = {  # stay in State C, keeping the date range for the next attempt
            "state": "C",  # still State C
            "correction_hint": hint,  # remember the hint that produced no matches
            "date_range": date_range,  # keep the resolved date range
            "action": action,  # keep the original action (correct or delete)
        }
        await message.reply_text(  # ask the user to describe the transaction differently
            "No encontré ninguna transacción con esos datos. ¿Puedes describirla diferente?"
        )
        return  # stop processing this message here

    user_states[user_id] = {  # move to State D
        "state": "D",  # State D: awaiting confirmation
        "matches": matches,  # the candidate transactions found above
        "date_range": date_range,  # keep the resolved date range in case of rejection
        "action": action,  # keep the original action (correct or delete)
    }

    if len(matches) == 1:  # exactly one transaction matched
        await message.reply_text(  # show it and ask for confirmation
            f"Encontré esta transacción: {format_transaction(matches[0])}\n"
            f"¿Es esta la que quieres {verb}? {YES_NO_MENU}"
        )
    else:  # multiple transactions matched
        await message.reply_text(build_multi_match_prompt(matches, verb))  # show the list, ask which one


async def handle_state_d(message, user_id: int, text: str, state_info: dict) -> None:  # handle a match-selection reply
    if is_cancellation(text):  # the user wants to abandon the correction entirely
        user_states.pop(user_id, None)  # return this user to State A
        await message.reply_text("Listo, no cambio nada ✅")  # confirm nothing was changed
        return  # stop processing this message here

    new_value = state_info.get("new_value")  # a new value Claude already extracted alongside the candidate, if any
    if new_value is not None:  # this D state offers a three-way choice instead of a plain yes/no
        await handle_state_d_with_new_value(message, user_id, text, state_info, new_value)  # delegate to that branch
        return  # stop processing this message here

    matches = state_info.get("matches", [])  # the candidate transactions found in State C
    normalized = text.strip().lower()  # normalize the reply for matching
    selected = None  # the transaction the user picked, if any
    went_back_to_c = False  # whether the user rejected the candidate(s) and should redo the search

    if len(matches) == 1:  # only one candidate was shown, expecting a yes/no reply
        if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # user confirmed this is the right transaction
            selected = matches[0]  # select the single candidate
        elif normalized == "2" or normalized in NEGATIVE_REPLIES:  # user rejected the single candidate
            went_back_to_c = True  # go back and ask for a new description
    else:  # multiple candidates were shown, expecting a number (or a rejection)
        if normalized in NEGATIVE_REPLIES:  # user rejected all candidates
            went_back_to_c = True  # go back and ask for a new description
        elif normalized.isdigit():  # the reply looks like a number
            index = int(normalized)  # parse the chosen index
            if 1 <= index <= len(matches):  # the index is within range
                selected = matches[index - 1]  # select the corresponding candidate

    action = state_info.get("action", "correct")  # whether this confirmation leads to a correction or a deletion

    if went_back_to_c:  # the user rejected the candidate(s) and the search must be redone
        if state_info.get("from_shortcut"):  # this candidate came from the last-transaction shortcut, not a real search
            user_states[user_id] = {  # move to State C to ask for the time window instead
                "state": "C",  # State C: awaiting the time window
                "correction_hint": state_info.get("correction_hint"),  # keep the original hint from classification
                "date_range": None,  # no date range chosen yet
                "action": action,  # keep the original action (correct or delete)
            }
            await message.reply_text(TIME_WINDOW_QUESTION)  # ask the user when the transaction happened
        else:  # this candidate came from an actual State C search, ask for a new description in the same window
            user_states[user_id] = {  # move back to State C, keeping the date range but clearing the hint
                "state": "C",  # back to State C
                "correction_hint": None,  # await a new description from the user
                "date_range": state_info.get("date_range"),  # keep the previously resolved date range
                "action": action,  # keep the original action (correct or delete)
            }
            await message.reply_text("¿Puedes describirla de otra forma?")  # ask for a new description
        return  # stop processing this message here

    if selected is None:  # the reply didn't resolve to a valid selection
        await message.reply_text(  # ask the user to reply more clearly
            "No entendí tu respuesta. Responde sí, no, o el número de la transacción."
        )
        return  # stay in State D (state is left unchanged)

    if action == "delete":  # this confirmation was for a deletion, not a correction — void it now, no further steps
        supabase.table("transactions").update(  # mark the selected transaction as void
            {"status": "anulada", "updated_at": now_iso()}  # set status and refresh the update timestamp
        ).eq("id", selected["id"]).execute()  # apply the update to the selected row
        user_states.pop(user_id, None)  # deletion flow is complete, return this user to State A
        await message.reply_text(f"Listo, eliminé esta transacción ✅ {format_transaction(selected)}")  # confirm it
        return  # stop processing this message here

    user_states[user_id] = {"state": "E", "match": selected}  # move to State E with the confirmed transaction
    await message.reply_text(NEW_VALUE_PROMPT)  # ask for the new value, description, or recurrence


async def handle_state_d_with_new_value(  # handle the three-way reply when Claude already extracted both the
    message, user_id: int, text: str, state_info: dict, new_value: str  # candidate transaction and the new value
) -> None:
    matches = state_info.get("matches", [])  # the candidate transaction (always a single item in this flow)
    match = matches[0] if matches else None  # the transaction being proposed
    normalized = text.strip().lower()  # normalize the reply for matching

    if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # option 1: confirm both the transaction and new value
        updated_row, updates = apply_correction(match, new_value)  # apply the stored new value, as in State E
        user_states.pop(user_id, None)  # correction flow is complete, return this user to State A
        await message.reply_text(build_correction_confirmation(updated_row, updates))  # confirm the correction
        return  # stop processing this message here

    if normalized == "2" or normalized == "otro valor":  # option 2: right transaction, wrong proposed new value
        user_states[user_id] = {"state": "E", "match": match}  # move to State E awaiting a fresh new value
        await message.reply_text(NEW_VALUE_PROMPT)  # ask for the new value, description, or recurrence
        return  # stop processing this message here

    if normalized == "3" or normalized == "otra":  # option 3: this isn't the right transaction at all
        user_states[user_id] = {  # move to State C to ask for the time window instead, clearing the stored new value
            "state": "C",  # State C: awaiting the time window
            "correction_hint": state_info.get("correction_hint"),  # keep the original hint from classification
            "date_range": None,  # no date range chosen yet
        }
        await message.reply_text(TIME_WINDOW_QUESTION)  # ask the user when the transaction happened
        return  # stop processing this message here

    await message.reply_text(  # the reply didn't match any of the three options, ask again
        "No entendí tu respuesta. Responde 1 (sí), 2 (otro valor), o 3 (otra)."
    )  # state is left unchanged, still State D


async def handle_state_e(message, user_id: int, text: str, state_info: dict) -> None:  # handle the new-value reply
    match = state_info.get("match")  # the transaction row selected for correction in State D

    if is_cancellation(text):  # check deterministically whether the user wants to abandon the correction
        user_states[user_id] = {"state": "E_CANCEL", "match": match}  # move to the sub-state asking what to cancel
        await message.reply_text(  # ask whether to cancel just the correction or the whole transaction
            "¿Olvido solo la corrección o también la transacción completa? Responde con el número o la palabra:\n"
            "1. la corrección\n2. la transacción"
        )
        return  # stop processing this message here

    normalized = text.strip().lower()  # normalize the reply for menu lookups
    if normalized in FIELD_MENU:  # user picked "1. el monto" or "2. la descripción" — needs a follow-up value
        user_states[user_id] = {"state": "E", "match": match, "field": FIELD_MENU[normalized]}  # stay in E, remember field
        prompt = "¿Cuál es el nuevo monto?" if FIELD_MENU[normalized] == "amount" else "¿Cuál es la nueva descripción?"
        await message.reply_text(prompt)  # ask specifically for that field's value
        return  # stop processing this message here

    if normalized in VALUE_MENU:  # user picked a menu option that is itself a complete value (frequency/category)
        updated_row, updates = apply_correction(match, VALUE_MENU[normalized])  # apply the chosen value directly
        user_states.pop(user_id, None)  # correction flow is complete, return this user to State A
        await message.reply_text(build_correction_confirmation(updated_row, updates))  # confirm the correction
        return  # stop processing this message here

    if references_frequency_without_value(text):  # user named the concept (e.g. "frecuencia") but no specific value
        user_states[user_id] = {"state": "E_FREQUENCY", "match": match}  # wait for the specific value, don't guess
        await message.reply_text(FREQUENCY_QUESTION)  # ask exactly which frequency they mean
        return  # stop processing this message here

    if references_category_without_value(text):  # user named the concept (e.g. "categoría") but no specific value
        user_states[user_id] = {"state": "E_CATEGORY", "match": match}  # wait for the specific value, don't guess
        await message.reply_text(CATEGORY_QUESTION)  # ask exactly which category they mean
        return  # stop processing this message here

    field = state_info.get("field")  # a prior FIELD_MENU choice ("amount"/"description") awaiting this value, if any
    if field == "amount":  # user is answering "¿Cuál es el nuevo monto?"
        parsed_amount = extract_amount_from_text(text)  # find a numeric amount among the words
        if parsed_amount is None:  # no recognizable number given
            await message.reply_text("No entendí el monto. ¿Cuál es el nuevo monto?")  # ask again, stay in State E
            return  # stay in State E with the same remembered field
        updated_row, updates = save_correction(match, {"amount": parsed_amount}, text)  # apply the amount only,
        # the description is left untouched here — auto-regenerating it would silently discard useful detail
        # (e.g. "leche y harina" becoming a generic "Gasto de X pesos."), so ask the user instead of guessing.
        user_states[user_id] = {"state": "E_DESCRIPTION_ASK", "match": updated_row}  # offer to also fix the description
        await message.reply_text(build_correction_confirmation(updated_row, updates))  # confirm the amount change
        await message.reply_text(DESCRIPTION_FOLLOWUP_QUESTION)  # ask whether the description needs updating too
        return  # stop processing this message here
    if field == "description":  # user is answering "¿Cuál es la nueva descripción?"
        updated_row, updates = save_correction(match, {"description": text.strip()}, text)  # apply the new description
        user_states.pop(user_id, None)  # correction flow is complete, return this user to State A
        await message.reply_text(build_correction_confirmation(updated_row, updates))  # confirm the correction
        return  # stop processing this message here

    # State E is purely deterministic: any non-cancellation, non-ambiguous reply is treated directly as the new
    # value, with no Claude call, to avoid the interrupt/clarification loop that could get stuck asking forever.
    updated_row, updates = apply_correction(match, text)  # apply the new value and get back the updated row

    user_states.pop(user_id, None)  # correction flow is complete, return this user to State A
    await message.reply_text(build_correction_confirmation(updated_row, updates))  # confirm the correction


async def handle_state_e_frequency(message, user_id: int, text: str, state_info: dict) -> None:  # handle a specific
    match = state_info.get("match")  # frequency reply — the transaction row selected for correction in State D
    normalized = text.strip().lower()  # normalize the reply for matching

    if is_cancellation(text):  # user wants to abandon the correction after all
        user_states[user_id] = {"state": "E_CANCEL", "match": match}  # move to the sub-state asking what to cancel
        await message.reply_text(  # ask whether to cancel just the correction or the whole transaction
            "¿Olvido solo la corrección o también la transacción completa? Responde con el número o la palabra:\n"
            "1. la corrección\n2. la transacción"
        )
        return  # stop processing this message here

    chosen = FREQUENCY_MENU.get(normalized) or extract_recurrence_from_text(text)  # accept a menu number or the word
    if chosen is None:  # still not a recognizable frequency value, do not guess
        await message.reply_text(FREQUENCY_RETRY)  # ask again
        return  # stay in State E_FREQUENCY (state is left unchanged)

    updated_row, updates = apply_correction(match, chosen)  # apply the now-confirmed frequency value
    user_states.pop(user_id, None)  # correction flow is complete, return this user to State A
    await message.reply_text(build_correction_confirmation(updated_row, updates))  # confirm the correction


async def handle_state_e_category(message, user_id: int, text: str, state_info: dict) -> None:  # handle a specific
    match = state_info.get("match")  # category reply — the transaction row selected for correction in State D
    normalized = text.strip().lower()  # normalize the reply for matching

    if is_cancellation(text):  # user wants to abandon the correction after all
        user_states[user_id] = {"state": "E_CANCEL", "match": match}  # move to the sub-state asking what to cancel
        await message.reply_text(  # ask whether to cancel just the correction or the whole transaction
            "¿Olvido solo la corrección o también la transacción completa? Responde con el número o la palabra:\n"
            "1. la corrección\n2. la transacción"
        )
        return  # stop processing this message here

    chosen = CATEGORY_MENU.get(normalized) or extract_category_from_text(text)  # accept a menu number or the word
    if chosen is None:  # still not a recognizable category value, do not guess
        await message.reply_text(CATEGORY_RETRY)  # ask again
        return  # stay in State E_CATEGORY (state is left unchanged)

    updated_row, updates = apply_correction(match, chosen)  # apply the now-confirmed category value
    user_states.pop(user_id, None)  # correction flow is complete, return this user to State A
    await message.reply_text(build_correction_confirmation(updated_row, updates))  # confirm the correction


async def handle_state_e_description_ask(message, user_id: int, text: str, state_info: dict) -> None:  # handle the
    match = state_info.get("match")  # yes/no reply to "also update the description?" after an amount-only correction
    normalized = text.strip().lower()  # normalize the reply for matching

    if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # user wants to update the description too
        user_states[user_id] = {"state": "E_DESCRIPTION_VALUE", "match": match}  # wait for the new description text
        await message.reply_text("¿Cuál es la nueva descripción?")  # ask for it
        return  # stop processing this message here

    if normalized == "2" or normalized in NEGATIVE_REPLIES:  # user is fine leaving the description as-is
        user_states.pop(user_id, None)  # correction flow is complete, return this user to State A
        await message.reply_text("Listo, la descripción queda igual ✅")  # confirm nothing else changed
        return  # stop processing this message here

    await message.reply_text(DESCRIPTION_FOLLOWUP_RETRY)  # unrecognized reply, ask again
    # state is left unchanged, still E_DESCRIPTION_ASK


async def handle_state_e_description_value(message, user_id: int, text: str, state_info: dict) -> None:  # handle the
    match = state_info.get("match")  # new description text after the user opted to also update it
    updated_row, updates = save_correction(match, {"description": text.strip()}, text)  # apply the new description
    user_states.pop(user_id, None)  # correction flow is complete, return this user to State A
    await message.reply_text(build_correction_confirmation(updated_row, updates))  # confirm the correction


async def handle_state_e_cancel(message, user_id: int, text: str, state_info: dict) -> None:  # handle cancel-scope reply
    match = state_info.get("match")  # the transaction that was being corrected
    normalized = text.strip().lower()  # normalize the reply for simple keyword matching

    if normalized == "1" or "corrección" in normalized or "correccion" in normalized:  # user only wants to abandon it
        supabase.table("transactions").update({"status": "activa"}).eq("id", match["id"]).execute()  # restore status
        user_states.pop(user_id, None)  # correction is abandoned, return this user to State A
        await message.reply_text(  # confirm that only the correction was cancelled
            "Listo, cancelé la corrección. La transacción original quedó sin cambios ✅"
        )
        return  # stop processing this message here

    if normalized == "2" or "transacción" in normalized or "transaccion" in normalized:  # void the whole transaction
        supabase.table("transactions").update(  # mark the transaction as void
            {"status": "anulada", "updated_at": now_iso()}  # set status and refresh the update timestamp
        ).eq("id", match["id"]).execute()  # apply the update to the matched row
        user_states.pop(user_id, None)  # transaction is void, return this user to State A
        await message.reply_text("Listo, anulé la transacción ✅")  # confirm the transaction was voided
        return  # stop processing this message here

    await message.reply_text(  # the reply didn't match either option, ask again
        "No entendí. Responde 1 (la corrección) o 2 (la transacción)."
    )  # state is left unchanged, still E_CANCEL


async def handle_state_r(message, user_id: int, text: str, state_info: dict) -> None:  # handle a report follow-up reply
    normalized = text.strip().lower()  # normalize the reply for matching
    today = today_local()  # reference date for "semana"/"mes"

    if normalized == "1" or "esta semana" in normalized:  # option 1: show this week's report instead
        start = week_start(today)  # Monday of the current week
        await send_period_report(message, user_id, start, today, f"Resumen de esta semana — {start:%d/%m/%Y} al {today:%d/%m/%Y}")
    elif normalized == "2" or "este mes" in normalized:  # option 2: show this month's report instead
        start = today.replace(day=1)  # first day of the current month
        await send_period_report(message, user_id, start, today, f"Resumen de este mes — {start:%d/%m/%Y} al {today:%d/%m/%Y}")
    elif normalized == "3" or "otra fecha" in normalized:  # option 3: ask for a custom start date
        user_states[user_id] = {"state": "R_DATE"}  # await the free-text date expression
        await message.reply_text(R_DATE_QUESTION)  # ask since when
    elif normalized == "4" or normalized in NEGATIVE_REPLIES or "no, gracias" in normalized or "no gracias" in normalized:
        # option 4, or any other decline — exit cleanly, no need to reclassify a bare "no"/"4" through Claude
        user_states.pop(user_id, None)  # return to State A
        await message.reply_text("OK✅")  # confirm the decline was understood
    else:  # any other message is unrelated to this menu — drop the follow-up and process it as a new message
        user_states.pop(user_id, None)  # return to State A
        await handle_state_a(message, user_id, text)  # classify and handle this message normally


async def handle_state_r_date(message, user_id: int, text: str, state_info: dict) -> None:  # handle the free-text
    parsed = parse_date_expression(text)  # date-expression reply to "¿Desde cuándo?", via Claude
    today = today_local()  # end of the report range is always today
    if parsed is not None:  # Claude confidently parsed a start date
        header = f"Resumen desde {parsed:%d/%m/%Y} al {today:%d/%m/%Y}"  # date-range header
        await send_period_report(message, user_id, parsed, today, header)  # show the report
    else:  # Claude couldn't parse the expression, fall back to asking for a manual date
        user_states[user_id] = {"state": "R_DATE_MANUAL"}  # await a DD/MM/YYYY reply
        await message.reply_text(R_DATE_MANUAL_QUESTION)  # ask for the manual format


async def handle_state_r_date_manual(message, user_id: int, text: str, state_info: dict) -> None:  # handle a
    today = today_local()  # DD/MM/YYYY reply, end of the report range is always today
    try:  # parse the manual date format directly, no Claude call needed
        parsed = datetime.strptime(text.strip(), "%d/%m/%Y").date()  # exact DD/MM/YYYY format
    except ValueError:  # the reply still isn't a valid DD/MM/YYYY date
        await message.reply_text(R_DATE_MANUAL_RETRY)  # ask again, stay in State R_DATE_MANUAL
        return  # stop processing this message here

    header = f"Resumen desde {parsed:%d/%m/%Y} al {today:%d/%m/%Y}"  # date-range header
    await send_period_report(message, user_id, parsed, today, header)  # show the report


async def handle_state_r_debtors_followup(message, user_id: int, text: str, state_info: dict) -> None:  # handle the
    normalized = text.strip().lower()  # sí/no reply to "¿Quieres ver la lista de acreedores también?"
    if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # user wants to see the creditor list too
        await send_creditor_list(message, user_id)  # show it
    elif normalized == "2" or normalized in NEGATIVE_REPLIES:  # user is done with reports
        user_states.pop(user_id, None)  # return to State A
        await message.reply_text("OK✅")  # confirm the decline was understood
    else:  # unrelated message, drop the follow-up and process it as a new message
        user_states.pop(user_id, None)  # return to State A
        await handle_state_a(message, user_id, text)  # classify and handle this message normally


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # entry point for every message
    message = update.message  # the incoming Telegram message object
    if message is None or message.text is None:  # ignore updates that are not plain text messages
        return  # nothing to do

    dedup_key = (message.chat_id, message.message_id)  # unique identifier for this exact Telegram message
    if not mark_message_processed(dedup_key):  # this message was already processed (duplicate delivery)
        logger.info("Skipping duplicate message_id %s in chat %s", message.message_id, message.chat_id)  # log it
        return  # skip silently: no reply, no Supabase writes, no state change

    user_id = message.from_user.id  # Telegram user ID, used as the conversation-state key
    text = message.text.strip()  # the message text with surrounding whitespace removed
    state_info = user_states.get(user_id, {"state": "A"})  # look up this user's state, defaulting to State A
    state = state_info.get("state", "A")  # extract the state name, defaulting to State A

    try:  # guard the whole dispatch so a single failure doesn't crash the bot
        if state == "B":  # user is answering a clarification question
            await handle_state_b(message, user_id, text, state_info)  # delegate to the State B handler
        elif state == "C":  # user is specifying the time window for a correction
            await handle_state_c(message, user_id, text, state_info)  # delegate to the State C handler
        elif state == "D":  # user is confirming which transaction to correct
            await handle_state_d(message, user_id, text, state_info)  # delegate to the State D handler
        elif state == "E":  # user is providing the new value for a correction
            await handle_state_e(message, user_id, text, state_info)  # delegate to the State E handler
        elif state == "E_FREQUENCY":  # user is naming a specific frequency after an ambiguous reference to it
            await handle_state_e_frequency(message, user_id, text, state_info)  # delegate to the E-frequency handler
        elif state == "E_CATEGORY":  # user is naming a specific category after an ambiguous reference to it
            await handle_state_e_category(message, user_id, text, state_info)  # delegate to the E-category handler
        elif state == "E_DESCRIPTION_ASK":  # user is answering whether to also update the description
            await handle_state_e_description_ask(message, user_id, text, state_info)  # delegate to that handler
        elif state == "E_DESCRIPTION_VALUE":  # user is providing the new description after opting in above
            await handle_state_e_description_value(message, user_id, text, state_info)  # delegate to that handler
        elif state == "E_CANCEL":  # user is choosing what exactly to cancel mid-correction
            await handle_state_e_cancel(message, user_id, text, state_info)  # delegate to the E-cancel handler
        elif state == "R":  # user is answering the post-report "quieres ver más?" follow-up
            await handle_state_r(message, user_id, text, state_info)  # delegate to the report follow-up handler
        elif state == "R_DATE":  # user is naming a custom start date in free text
            await handle_state_r_date(message, user_id, text, state_info)  # delegate to the R_DATE handler
        elif state == "R_DATE_MANUAL":  # user is naming a custom start date in DD/MM/YYYY format
            await handle_state_r_date_manual(message, user_id, text, state_info)  # delegate to that handler
        elif state == "R_DEBTORS_FOLLOWUP":  # user is answering whether to also see the creditor list
            await handle_state_r_debtors_followup(message, user_id, text, state_info)  # delegate to that handler
        elif state == "OVERGUARD_CONFIRM":  # user is confirming a pago_deuda/cobro with no debt on record
            await handle_state_overguard_confirm(message, user_id, text, state_info)  # delegate to that handler
        else:  # default: no conversation flow in progress
            await handle_state_a(message, user_id, text)  # delegate to the State A handler
    except Exception:  # catch any unexpected failure from the handlers above
        logger.exception("Failed to process message in state %s", state)  # log the full traceback for debugging
        user_states.pop(user_id, None)  # reset this user's state so they aren't stuck in a broken flow
        await message.reply_text("Hubo un error al procesar tu mensaje. Intenta de nuevo.")  # tell the user


EXAMPLES_MESSAGE = (  # sent right after the welcome message, showing the range of things the bot understands
    "Aquí tienes ejemplos de lo que puedes escribirme:\n\n"
    "📝 Registrar una transacción:\n"
    "• Ingreso: \"vendí 50.000 de tacos\"\n"
    "• Gasto: \"pagué 30.000 de gas\"\n"
    "• Préstamo (tú le prestas a alguien): \"le presté 20.000 a Bryan\"\n"
    "• Deuda (alguien te presta a ti): \"Wilmer me prestó 100.000\"\n"
    "• Cobro (te pagan lo que te debían): \"Bryan me pagó 50.000\"\n"
    "• Pago de deuda (tú le pagas a alguien): \"le pagué 100.000 a Wilmer\"\n\n"
    "📊 Consultar tu balance:\n"
    "• De hoy: \"cómo voy\" o \"cómo voy hoy\"\n"
    "• Del mes: \"cómo voy este mes\"\n"
    "• De una fecha específica: \"desde el 10 de julio\"\n"
    "• Quién te debe: \"quién me debe\"\n"
    "• A quién le debes: \"a quién le debo\""
)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # sent when a user opens the bot
    await update.message.reply_text(  # for the first time (Telegram sends /start automatically on "Start" tap)
        "Bienvenido(a)! Soy tu asistente contable. Aquí comienzas a incrementar tu control sobre tus finanzas. Vamos!"
    )
    await update.message.reply_text(EXAMPLES_MESSAGE)  # follow up with usage examples


def main() -> None:  # build and run the Telegram bot
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()  # construct the bot application
    application.add_handler(CommandHandler("start", handle_start))  # welcome message for new users
    application.add_handler(  # register the handler for incoming text messages
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)  # match any text message that isn't a command
    )
    application.run_polling()  # start polling Telegram for updates until interrupted


if __name__ == "__main__":  # only run the bot when this file is executed directly
    main()  # start the bot
