# Channel-agnostic business logic: the full state machine, Claude classification, Whisper transcription, and every
# piece of Spanish reply text. Nothing in this module knows what a Telegram message is.
#
# Entry points for any channel:
#   handle_incoming_message(channel, user_id, text)                     -> list[str]
#   handle_incoming_voice(channel, user_id, audio_bytes, mime_type)     -> list[str]
#   welcome_messages()                                                  -> list[str]
#
# Each returns the reply text(s) to send, in order. A list is used because two flows legitimately produce two
# separate messages (the amount-correction confirmation + its follow-up question, and the welcome + examples pair).

import difflib  # standard library module used for typo-tolerant fuzzy text matching
import json  # standard library module used to parse and serialize JSON data
import logging  # standard library module used to log bot activity and errors
import os  # standard library module used to read environment variables
import re  # standard library module used for regular expression matching
import unicodedata  # standard library module used to strip accents when matching user replies
from datetime import date, datetime, timedelta, timezone  # date/time utilities for date-range parsing and timestamps

import anthropic  # official Anthropic SDK used to call the Claude API
import openai  # official OpenAI SDK, used only for Whisper speech-to-text on voice notes

import db  # the Supabase data layer; also owns load_dotenv(), LOCAL_TZ and today_local()
from db import today_local  # today's date in the vendor's local timezone, used for every report boundary

logger = logging.getLogger(__name__)  # module-level logger, configured by the entry point

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # Anthropic API key used to call Claude
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # OpenAI API key used to transcribe voice notes via Whisper

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)  # Anthropic client instance used for all Claude API calls
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)  # OpenAI client instance, used only for Whisper transcription

CLAUDE_MODEL = "claude-sonnet-4-6"  # Claude model used to classify every incoming message
WHISPER_MODEL = "whisper-1"  # OpenAI speech-to-text model used to transcribe voice notes

VOICE_TRANSCRIPTION_ERROR = "No pude entender la nota de voz. ¿Puedes escribirlo?"  # shown when transcription fails

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


def strip_accents(text: str) -> str:  # drop diacritics so "sí"/"si" and "descripción"/"descripcion" compare equal
    decomposed = unicodedata.normalize("NFKD", text)  # split each accented character into base letter + accent mark
    return "".join(char for char in decomposed if not unicodedata.combining(char))  # keep only the base letters


def normalize_reply(text: str) -> str:  # canonical form for every user reply compared against an expected word
    return strip_accents(text.strip().lower())  # trim, lowercase, and de-accent so "SÍ", "Sí", "si" all match


# Every collection below is compared against a normalize_reply() result, so its LOOKUP KEYS are stored de-accented.
# Dictionary VALUES are the canonical values written to Supabase and must keep their accents ("única vez").
CANCELLATION_KEYWORDS = [  # phrases meaning "cancel" (accent-free: "olvídalo" normalizes to "olvidalo")
    "olvidalo", "no importa", "cancela", "borra eso",
]
AFFIRMATIVE_REPLIES = {"si", "s", "yes", "correcto", "esa es", "esa"}  # normalized replies treated as "yes"
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
    "2": "description", "descripcion": "description",  # "descripción" normalizes to this same de-accented key
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

CANCEL_SCOPE_QUESTION = (  # asked when the user cancels mid-correction: just the edit, or the whole transaction?
    "¿Olvido solo la corrección o también la transacción completa? Responde con el número o la palabra:\n"
    "1. la corrección\n2. la transacción"
)

OVERGUARD_QUESTION_TEMPLATE = (  # asked when a pago_deuda/cobro names someone with no debt on record
    "⚠️ No tengo registrada ninguna deuda con {name}. ¿Quieres registrar este pago de todas formas?\n1. sí\n2. no"
)

WELCOME_MESSAGE = (  # first message a brand-new user sees
    "Bienvenido(a)! Soy tu asistente contable. Aquí comienzas a incrementar tu control sobre tus finanzas. Vamos!"
)

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

# Conversation state lives in Supabase (table user_state, keyed by channel + user_id), so it survives restarts and
# is shared across processes. Within a single message it is held in a plain dict — the "session" — which handlers
# read and mutate via transition()/finish(); handle_incoming_message loads it before dispatch and saves it after.

STATE_TIMEOUT = timedelta(minutes=60)  # how long an unfinished flow survives before it is abandoned
STATE_EXPIRED_MESSAGE = (  # sent when a stale flow is dropped; the triggering message itself is NOT processed
    "Pasó más de una hora desde tu última respuesta, así que cerré lo que estábamos haciendo. "
    "Vuelve a escribirme lo que necesitas 👍"
)


def transition(session: dict, new_state: dict) -> None:  # move this flow to a new state, replacing its working data
    session.clear()  # drop the previous state's fields so nothing leaks between flows
    session.update(new_state)  # adopt the new state name and its data


def finish(session: dict) -> None:  # end the flow and return the user to State A
    session.clear()  # an empty session means State A
    session["state"] = "A"  # spelled out so callers reading session["state"] always find a value


def state_is_expired(state: str, updated_at: datetime) -> bool:  # idle longer than STATE_TIMEOUT, mid-flow?
    if state == "A":  # State A is the resting state — there is no flow to abandon, so it never expires
        return False  # and the user is never told anything
    if updated_at is None:  # no timestamp stored — treat it as still fresh rather than risk spurious messages
        return False
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC, matching what Supabase stores
    return now - updated_at > STATE_TIMEOUT  # expired once the idle gap exceeds the limit


# state_data must survive a round trip through JSONB. Everything handlers store is already JSON-native (strings,
# booleans, Supabase rows, Claude's parsed classification) with one exception: date_range, a tuple of two
# datetimes. These two functions convert just that field, so nothing else has to care.

def serialize_state_data(session: dict) -> dict:  # session -> JSON-safe dict for Supabase (excludes "state")
    data = {key: value for key, value in session.items() if key != "state"}  # "state" is its own column
    date_range = data.get("date_range")  # the only non-JSON-native field a handler ever stores
    if date_range:  # a (start, end) pair of datetimes
        data["date_range"] = [date_range[0].isoformat(), date_range[1].isoformat()]  # store as ISO strings
    return data


def deserialize_state_data(data: dict) -> dict:  # JSON dict from Supabase -> session fields
    restored = dict(data or {})  # copy so the caller's dict is not mutated
    date_range = restored.get("date_range")  # was written as a two-item list of ISO strings
    if date_range:  # rebuild the tuple of datetimes the handlers expect
        restored["date_range"] = (datetime.fromisoformat(date_range[0]), datetime.fromisoformat(date_range[1]))
    return restored


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
    normalized = normalize_reply(text)  # normalize the text for case-insensitive matching
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


RECURRENCE_PHRASES = ["unica vez", "una vez", "unico"]  # one-off phrases (de-accented: "única"/"único" normalize here)
FREQUENCY_CONCEPT_ROOTS = ["frecuenc", "recurrenc"]  # roots of "frecuencia"/"recurrencia", the CATEGORY noun, not a value
RECURRENCE_VALUE_WORDS = {"recurrente": "recurrente", "variable": "variable"}  # canonical value words, fuzzy-matched below


def extract_recurrence_from_text(text: str) -> str | None:  # find an explicit recurrence category among free text
    normalized = normalize_reply(text)  # normalize for case-insensitive matching
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
    normalized = normalize_reply(text)  # normalize for matching
    mentions_concept = any(root in normalized for root in FREQUENCY_CONCEPT_ROOTS)  # e.g. "frecuencia", "recurrencia"
    return mentions_concept and extract_recurrence_from_text(text) is None  # true only if no concrete value was also given


CATEGORY_CONCEPT_ROOTS = ["categor"]  # root of "categoría"/"categoria", the CATEGORY noun, not a value
CATEGORY_VALUE_WORDS = {"negocio": "negocio", "personal": "personal"}  # canonical value words, fuzzy-matched below


def extract_category_from_text(text: str) -> str | None:  # find an explicit category value among free text
    normalized = normalize_reply(text)  # normalize for case-insensitive matching
    for token in normalized.split():  # check each whitespace-separated word
        cleaned = token.strip(".,;:!?")  # drop trailing punctuation, e.g. "negocio." -> "negocio"
        match = fuzzy_match(cleaned, list(CATEGORY_VALUE_WORDS))  # typo-tolerant match against the value words
        if match:  # this token is close enough to a known value, misspelled or not
            return CATEGORY_VALUE_WORDS[match]  # return the canonical value it matched
    return None  # no category was named in the text


def references_category_without_value(text: str) -> bool:  # user named the concept ("categoría") but no specific value
    normalized = normalize_reply(text)  # normalize for matching
    mentions_concept = any(root in normalized for root in CATEGORY_CONCEPT_ROOTS)  # e.g. "categoría", "categoria"
    return mentions_concept and extract_category_from_text(text) is None  # true only if no concrete value was also given


def parse_date_range(text: str) -> tuple[datetime, datetime] | None:  # map a Spanish time-window phrase to a range
    normalized = normalize_reply(text)  # normalize the text for matching
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


def build_correction_confirmation(updated_row: dict, updates: dict) -> str:  # build the "Corregido ✅ ..." reply text
    detail = format_transaction(updated_row)  # the standard transaction summary (description, amount, type, date)
    if "recurrence" in updates:  # this correction specifically changed the recurrence category
        detail += f" | Frecuencia: {updates['recurrence']}"  # surface the new recurrence so the change is visible
    if "category" in updates:  # this correction specifically changed the category
        detail += f" | Categoría: {updates['category']}"  # surface the new category so the change is visible
    return f"Corregido ✅ {detail}"  # combine into the final confirmation text


def description_matches_hint(description: str | None, hint: str) -> bool:  # typo-tolerant description match
    if not description:  # nothing to compare against
        return False  # can't match an empty description
    # Both sides are de-accented so a hint typed without accents still finds the stored description that has them
    # (e.g. searching "administracion" matches a transaction saved as "Pago de administración").
    normalized_description = normalize_reply(description)  # normalize for matching
    normalized_hint = normalize_reply(hint)  # normalize the hint the same way
    if normalized_hint in normalized_description:  # fast path: exact substring, no typo (handles most real cases)
        return True  # already an exact match, no need for fuzzy comparison
    words = normalized_description.split()  # compare the hint against each individual word in the description
    return fuzzy_match(normalized_hint, words) is not None  # true if any word is close enough, typos included


def search_transactions(  # search for transactions matching a date range and an optional correction hint
    channel: str, user_id: str, start: datetime, end: datetime, hint: str | None
) -> list[dict]:
    parsed_amount = try_parse_amount(hint) if hint else None  # check whether the hint looks like a numeric amount
    rows = db.search_transactions(channel, user_id, start, end, parsed_amount)  # query, narrowed by amount if numeric
    if hint and parsed_amount is None:  # the hint is descriptive text, apply typo-tolerant filtering in Python
        rows = [row for row in rows if description_matches_hint(row.get("description"), hint)]  # keep close matches
    return rows  # return the matching rows, or an empty list if none


def find_transactions_by_keyword(channel: str, user_id: str, keyword: str) -> list[dict]:  # typo-tolerant, newest first
    rows = db.get_all_active_transactions(channel, user_id)  # every active transaction for this channel + user
    return [row for row in rows if description_matches_hint(row.get("description"), keyword)]  # keep close matches


def resolve_shortcut_candidates(channel: str, user_id: str, keyword: str | None) -> list[dict]:  # shortcut candidates
    candidates = find_transactions_by_keyword(channel, user_id, keyword) if keyword else []  # typo-tolerant, if named
    if not candidates:  # no keyword was given, or nothing matched it even fuzzily
        most_recent = db.get_most_recent_transaction(channel, user_id)  # fall back to the most recent active transaction
        candidates = [most_recent] if most_recent else []  # wrap it as a single-item list, or none if none exists
    return candidates  # zero, one, or many candidate transactions


def build_multi_match_prompt(candidates: list[dict], verb: str) -> str:  # numbered-list prompt for several candidates
    lines = [f"{i}. {format_transaction(row)}" for i, row in enumerate(candidates, start=1)]  # number each candidate
    return "\n".join(lines) + f"\n¿Cuál es la que quieres {verb}? Responde con el número"  # ask which one


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

    return db.save_correction(match, field_updates, new_value_text)  # apply it and return updated row + changed fields


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


def send_period_report(channel: str, user_id: str, session: dict, start: date, end: date, header: str) -> list[str]:  # show a report,
    totals = db.get_period_report(channel, user_id, start, end)  # then enter the follow-up state
    transition(session, {"state": "R"})  # await a follow-up choice (another period, or "no")
    return [format_period_report(totals, header)]  # the formatted report


def send_debtor_list(channel: str, user_id: str, session: dict) -> list[str]:  # show the debtor list, then offer the creditor list
    debtors, overpaid = db.get_debtor_list(channel, user_id)  # who owes the vendor, and who they overcollected from
    transition(session, {"state": "R_DEBTORS_FOLLOWUP"})  # await sí/no on the creditor list
    return [format_debtor_list(debtors, overpaid)]  # the formatted list


def send_creditor_list(channel: str, user_id: str, session: dict) -> list[str]:  # show the creditor list; nothing to follow up with
    creditors, overpaid = db.get_creditor_list(channel, user_id)  # who the vendor owes, and who they overpaid
    finish(session)  # this report has no further follow-up, return to State A
    return [format_creditor_list(creditors, overpaid)]  # the formatted list


def handle_custom_date_query(channel: str, user_id: str, session: dict, date_hint: str | None) -> list[str]:  # "custom" query_period
    today = today_local()  # end of the report range is always today
    parsed = parse_date_expression(date_hint) if date_hint else None  # try to resolve the hint to a start date
    if parsed is not None:  # Claude confidently parsed a start date
        header = f"Resumen desde {parsed:%d/%m/%Y} al {today:%d/%m/%Y}"  # date-range header
        return send_period_report(channel, user_id, session, parsed, today, header)  # show the report
    transition(session, {"state": "R_DATE_MANUAL"})  # await a DD/MM/YYYY reply
    return [R_DATE_MANUAL_QUESTION]  # ask for the manual format


def cancel_last_transaction(channel: str, user_id: str) -> list[str]:  # void the most recent active transaction
    row = db.get_most_recent_transaction(channel, user_id)  # look up the most recent active transaction
    if row is None:  # no active transaction exists for this user
        return ["No encontré ninguna transacción activa para anular."]  # nothing to void
    db.void_transaction(row["id"])  # void that transaction
    return [f"Listo, anulé la última transacción ✅ {format_transaction(row)}"]  # confirm, including its details


def save_new_transaction(channel: str, user_id: str, session: dict, raw_text: str, classification: dict) -> list[str]:  # finalize a
    transaction_type = classification.get("type")  # pago_deuda/cobro against the existing balance
    amount = classification.get("amount") or 0
    category = classification.get("category")
    recurrence = classification.get("recurrence")

    if transaction_type == "pago_deuda":  # repaying a creditor — check it against what's actually owed
        creditor_name = classification.get("creditor_name")
        balance = db.get_creditor_balance(channel, user_id, creditor_name, category) if creditor_name else 0

        if balance <= 0:  # no debt on record for this creditor/category — confirm before saving anything
            transition(session, {"state": "OVERGUARD_CONFIRM", "classification": classification, "raw_text": raw_text})
            return [OVERGUARD_QUESTION_TEMPLATE.format(name=creditor_name or "esa persona")]

        if amount > balance:  # paid more than what was owed — split into a debt-closing part and an excess loan
            excess = amount - balance
            db.save_transaction(channel, user_id, raw_text, build_split_classification("pago_deuda", balance, category, recurrence, creditor_name=creditor_name))
            db.save_transaction(channel, user_id, raw_text, build_split_classification("préstamo", excess, category, recurrence, debtor_name=creditor_name))
            return [
                f"Anotado ✅ Registré ${format_amount_es(balance)} como pago a {creditor_name}. "
                f"Los ${format_amount_es(excess)} restantes quedaron como un préstamo a tu favor — "
                f"ahora {creditor_name} te debe ese dinero."
            ]

        db.save_transaction(channel, user_id, raw_text, classification)  # amount <= balance, save normally
        return [build_new_transaction_confirmation(classification)]

    if transaction_type == "cobro":  # collecting from a debtor — check it against what's actually owed
        debtor_name = classification.get("debtor_name")
        balance = db.get_debtor_balance(channel, user_id, debtor_name) if debtor_name else 0

        if balance <= 0:  # no debt on record for this debtor — confirm before saving anything
            transition(session, {"state": "OVERGUARD_CONFIRM", "classification": classification, "raw_text": raw_text})
            return [OVERGUARD_QUESTION_TEMPLATE.format(name=debtor_name or "esa persona")]

        if amount > balance:  # collected more than what was owed — split into a debt-closing part and an excess debt
            excess = amount - balance
            db.save_transaction(channel, user_id, raw_text, build_split_classification("cobro", balance, category, recurrence, debtor_name=debtor_name))
            db.save_transaction(channel, user_id, raw_text, build_split_classification("deuda", excess, category, recurrence, creditor_name=debtor_name))
            return [
                f"Anotado ✅ Registré ${format_amount_es(balance)} como cobro de {debtor_name}. "
                f"Los ${format_amount_es(excess)} restantes quedaron como una deuda a su favor — "
                f"ahora le debes ese dinero a {debtor_name}."
            ]

        db.save_transaction(channel, user_id, raw_text, classification)  # amount <= balance, save normally
        return [build_new_transaction_confirmation(classification)]

    db.save_transaction(channel, user_id, raw_text, classification)  # any other type, no balance to check against
    return [build_new_transaction_confirmation(classification)]


def handle_state_overguard_confirm(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # handle the
    normalized = normalize_reply(text)
    classification = session.get("classification")  # the originally classified transaction, unmodified
    raw_text = session.get("raw_text")  # the message that produced it

    if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # vendor confirmed: save it anyway, as-is
        finish(session)  # return to State A
        db.save_transaction(channel, user_id, raw_text, classification)  # no balance to split against, save it all
        return [build_new_transaction_confirmation(classification)]

    if normalized == "2" or normalized in NEGATIVE_REPLIES:  # vendor declined: discard it
        finish(session)  # return to State A
        return ["OK ✅ No se registró nada."]

    return ["No entendí. Responde con el número o la palabra:\n1. sí\n2. no"]  # unrecognized reply


def handle_state_a(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # handle a message with no flow in progress
    classification = classify_message(text)  # ask Claude to classify the message
    logger.info("Classification for %r: %s", text, json.dumps(classification, ensure_ascii=False))  # debug visibility
    intent = classification.get("intent")  # extract the classified intent

    if intent == "new_transaction":  # the user is registering a new transaction
        if classification.get("needs_clarification"):  # Claude needs more information before saving
            transition(session, {"state": "B", "original_text": text})  # remember the text, move to State B
            return [  # choose the clarification question to ask
                classification.get("clarification_question")  # Claude's suggested question
                or "¿Puedes dar más detalles?"  # generic fallback if none was provided
            ]
        return save_new_transaction(channel, user_id, session, text, classification)  # save it, splitting overpayments

    if intent == "correction":  # the user wants to fix a past transaction
        hint = normalize_classification_value(classification.get("correction_hint"))  # the OLD field value, if named
        new_value = normalize_classification_value(classification.get("new_value"))  # the NEW value, as a string
        keyword = normalize_classification_value(  # a description keyword identifying WHICH transaction, if named
            classification.get("transaction_keyword")
        )
        candidates = resolve_shortcut_candidates(channel, user_id, keyword)  # typo-tolerant search, most recent fallback

        if len(candidates) == 1 and new_value:  # exactly one candidate, and Claude already knows the new value too
            candidate = candidates[0]  # the single candidate transaction
            transition(session, {  # go straight to State D with the candidate and the proposed new value
                "state": "D",  # State D: awaiting confirmation
                "matches": [candidate],  # only one candidate to confirm
                "date_range": None,  # no date range has been resolved yet
                "correction_hint": hint,  # keep the original hint in case the user rejects this candidate
                "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                "new_value": new_value,  # the new value to apply immediately if the user confirms both
                "action": "correct",  # this confirmation leads to a correction, not a deletion
            })
            return [  # show the transaction, the proposed new value, and ask for confirmation
                f"¿Te refieres a esta transacción? {format_transaction(candidate)}\n"
                f"¿Y el nuevo valor es {new_value}? Responde con el número o la palabra:\n"
                "1. sí\n2. otro valor\n3. otra"
            ]
        if len(candidates) == 1:  # exactly one candidate, but no new value was extracted yet
            candidate = candidates[0]  # the single candidate transaction
            transition(session, {  # go straight to State D with this transaction as the candidate
                "state": "D",  # State D: awaiting confirmation
                "matches": [candidate],  # only one candidate to confirm
                "date_range": None,  # no date range has been resolved yet
                "correction_hint": hint,  # keep the original hint in case the user rejects this candidate
                "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                "action": "correct",  # this confirmation leads to a correction, not a deletion
            })
            return [  # show the transaction and ask for confirmation
                f"¿Te refieres a esta transacción? {format_transaction(candidate)}\n{YES_NO_MENU}"
            ]
        if candidates:  # multiple candidates matched (typos included), ask which one instead of guessing
            transition(session, {  # go straight to State D with every candidate to choose from
                "state": "D",  # State D: awaiting confirmation
                "matches": candidates,  # every candidate the fuzzy search found
                "date_range": None,  # no date range has been resolved yet
                "correction_hint": hint,  # keep the original hint in case all candidates are rejected
                "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                "action": "correct",  # this confirmation leads to a correction, not a deletion
            })
            return [build_multi_match_prompt(candidates, "corregir")]  # show the list, ask which one
        transition(session, {  # this user has no transactions at all yet, fall back to the time-window question
            "state": "C",  # State C: awaiting the time window
            "correction_hint": hint,  # the OLD value/description to search for
            "date_range": None,  # no date range chosen yet
            "action": "correct",  # this search leads to a correction, not a deletion
        })
        return [TIME_WINDOW_QUESTION]  # ask the user when the transaction happened

    if intent == "query":  # the user is asking for a balance or report
        query_period = classification.get("query_period")  # which report Claude classified this query as
        today = today_local()  # reference date for "hoy"/"semana"/"mes"

        if query_period == "semana":  # this week, Monday through today
            start = week_start(today)  # Monday of the current week
            return send_period_report(channel, user_id, session, start, today, f"Resumen de esta semana — {start:%d/%m/%Y} al {today:%d/%m/%Y}")
        if query_period == "mes":  # this month, the 1st through today
            start = today.replace(day=1)  # first day of the current month
            return send_period_report(channel, user_id, session, start, today, f"Resumen de este mes — {start:%d/%m/%Y} al {today:%d/%m/%Y}")
        if query_period == "deudores":  # who owes the vendor money
            return send_debtor_list(channel, user_id, session)  # show the debtor list
        if query_period == "acreedores":  # who the vendor owes money to
            return send_creditor_list(channel, user_id, session)  # show the creditor list
        if query_period == "custom":  # the user named a specific starting point
            return handle_custom_date_query(channel, user_id, session, classification.get("query_date_hint"))  # resolve it
        if query_period == "hoy":  # today's status
            return send_period_report(channel, user_id, session, today, today, f"Resumen de hoy — {today:%d/%m/%Y}")
        return [  # query_period is null — ambiguous, ask the user to be specific instead of guessing
            "No entendí qué quieres consultar. ¿Hoy, esta semana, este mes, deudores, o acreedores?"
        ]

    if intent == "cancellation":  # the user wants to undo the last transaction, or delete a specific one
        keyword = normalize_classification_value(  # a description keyword identifying WHICH transaction, if named
            classification.get("transaction_keyword")
        )
        if not keyword:  # no specific transaction named, undo the most recent one (fast path, no confirmation)
            return cancel_last_transaction(channel, user_id)  # void the most recent active transaction
        candidates = resolve_shortcut_candidates(channel, user_id, keyword)  # typo-tolerant search, recent fallback
        if len(candidates) == 1:  # exactly one candidate
            candidate = candidates[0]  # the single candidate transaction
            transition(session, {  # go straight to State D with this transaction as the candidate
                "state": "D",  # State D: awaiting confirmation
                "matches": [candidate],  # only one candidate to confirm
                "date_range": None,  # no date range has been resolved yet
                "correction_hint": keyword,  # keep the keyword in case the user rejects this candidate
                "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                "action": "delete",  # this confirmation leads to a deletion, not a correction
            })
            return [  # show the transaction and ask for confirmation before deleting it
                f"¿Quieres eliminar esta transacción? {format_transaction(candidate)}\n{YES_NO_MENU}"
            ]
        if candidates:  # multiple candidates matched (typos included), ask which one instead of guessing
            transition(session, {  # go straight to State D with every candidate to choose from
                "state": "D",  # State D: awaiting confirmation
                "matches": candidates,  # every candidate the fuzzy search found
                "date_range": None,  # no date range has been resolved yet
                "correction_hint": keyword,  # keep the keyword in case all candidates are rejected
                "from_shortcut": True,  # mark this D state as reached via the last-transaction shortcut
                "action": "delete",  # this confirmation leads to a deletion, not a correction
            })
            return [build_multi_match_prompt(candidates, "eliminar")]  # show the list, ask which
        return ["No encontré ninguna transacción activa para eliminar."]  # this user has no transactions at all

    # intent is "confirmation" (or anything unexpected) with nothing pending to confirm
    logger.info("Intent %r with nothing pending to confirm/act on for user %s/%s", intent, channel, user_id)
    return ["OK ✅"]  # user-facing reply stays simple; stay in State A (no state change needed)


def handle_state_b(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # a clarification reply
    original = session.get("original_text", "")  # the message that originally triggered the clarification
    combined = (  # build the combined message to re-classify
        f"Mensaje original: {original}\n"  # include the original message
        f"Respuesta del usuario a la pregunta de aclaración: {text}"  # include the user's clarifying reply
    )
    classification = classify_message(combined)  # re-classify using the combined context
    intent = classification.get("intent")  # extract the classified intent

    if intent == "cancellation":  # the user cancelled instead of clarifying
        finish(session)  # abandon the pending transaction, return this user to State A
        return ["Listo, no anoto nada ✅"]  # confirm nothing was saved

    if classification.get("needs_clarification"):  # Claude still needs more information
        transition(session, {"state": "B", "original_text": combined})  # stay in State B with updated context
        return [  # choose the next clarification question to ask
            classification.get("clarification_question")  # Claude's suggested question
            or "¿Puedes dar más detalles?"  # generic fallback if none was provided
        ]

    finish(session)  # clarification flow is complete; save_new_transaction may set a fresh state below
    return save_new_transaction(channel, user_id, session, combined, classification)  # save it, splitting overpayments


def handle_state_c(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # a time-window reply
    if is_cancellation(text):  # the user wants to abandon the correction entirely
        finish(session)  # return this user to State A
        return ["Listo, no cambio nada ✅"]  # confirm nothing was changed

    date_range = session.get("date_range")  # the previously resolved date range, if any
    hint = session.get("correction_hint")  # the current correction hint (the old value to search for)

    if date_range is None:  # we don't have a date range yet, so this reply should specify one
        parsed_range = parse_date_range(text)  # try to parse a time-window phrase from the reply
        if parsed_range is None:  # the reply didn't match a known time window
            return [TIME_WINDOW_RETRY]  # ask the user to reply with one of the supported options
        date_range = parsed_range  # use the newly parsed range
    else:  # we already have a date range, so this reply is a new description of the transaction
        hint = text  # replace the hint with the user's new description

    start, end = date_range  # unpack the resolved date range
    matches = search_transactions(channel, user_id, start, end, hint)  # search Supabase for matching transactions

    action = session.get("action", "correct")  # whether this search leads to a correction or a deletion
    verb = "corregir" if action == "correct" else "eliminar"  # the verb to use in the confirmation prompts below

    if not matches:  # no transactions matched the range and hint
        transition(session, {  # stay in State C, keeping the date range for the next attempt
            "state": "C",  # still State C
            "correction_hint": hint,  # remember the hint that produced no matches
            "date_range": date_range,  # keep the resolved date range
            "action": action,  # keep the original action (correct or delete)
        })
        return ["No encontré ninguna transacción con esos datos. ¿Puedes describirla diferente?"]

    transition(session, {  # move to State D
        "state": "D",  # State D: awaiting confirmation
        "matches": matches,  # the candidate transactions found above
        "date_range": date_range,  # keep the resolved date range in case of rejection
        "action": action,  # keep the original action (correct or delete)
    })

    if len(matches) == 1:  # exactly one transaction matched
        return [  # show it and ask for confirmation
            f"Encontré esta transacción: {format_transaction(matches[0])}\n"
            f"¿Es esta la que quieres {verb}? {YES_NO_MENU}"
        ]
    return [build_multi_match_prompt(matches, verb)]  # multiple matched: show the list, ask which one


def handle_state_d(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # a match-selection reply
    if is_cancellation(text):  # the user wants to abandon the correction entirely
        finish(session)  # return this user to State A
        return ["Listo, no cambio nada ✅"]  # confirm nothing was changed

    new_value = session.get("new_value")  # a new value Claude already extracted alongside the candidate, if any
    if new_value is not None:  # this D state offers a three-way choice instead of a plain yes/no
        return handle_state_d_with_new_value(channel, user_id, text, session, new_value)  # delegate to that branch

    matches = session.get("matches", [])  # the candidate transactions found in State C
    normalized = normalize_reply(text)  # normalize the reply for matching
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

    action = session.get("action", "correct")  # whether this confirmation leads to a correction or a deletion

    if went_back_to_c:  # the user rejected the candidate(s) and the search must be redone
        if session.get("from_shortcut"):  # this candidate came from the last-transaction shortcut, not a real search
            transition(session, {  # move to State C to ask for the time window instead
                "state": "C",  # State C: awaiting the time window
                "correction_hint": session.get("correction_hint"),  # keep the original hint from classification
                "date_range": None,  # no date range chosen yet
                "action": action,  # keep the original action (correct or delete)
            })
            return [TIME_WINDOW_QUESTION]  # ask the user when the transaction happened
        transition(session, {  # this candidate came from an actual State C search, ask for a new description
            "state": "C",  # back to State C
            "correction_hint": None,  # await a new description from the user
            "date_range": session.get("date_range"),  # keep the previously resolved date range
            "action": action,  # keep the original action (correct or delete)
        })
        return ["¿Puedes describirla de otra forma?"]  # ask for a new description

    if selected is None:  # the reply didn't resolve to a valid selection
        return ["No entendí tu respuesta. Responde sí, no, o el número de la transacción."]  # stay in State D

    if action == "delete":  # this confirmation was for a deletion — void it now, no further steps
        db.void_transaction(selected["id"])  # mark the selected transaction as void
        finish(session)  # deletion flow is complete, return this user to State A
        return [f"Listo, eliminé esta transacción ✅ {format_transaction(selected)}"]  # confirm it

    transition(session, {"state": "E", "match": selected})  # move to State E with the confirmed transaction
    return [NEW_VALUE_PROMPT]  # ask for the new value, description, or recurrence


def handle_state_d_with_new_value(  # handle the three-way reply when Claude already extracted both the
    channel: str, user_id: str, text: str, session: dict, new_value: str  # candidate transaction and the new value
) -> list[str]:
    matches = session.get("matches", [])  # the candidate transaction (always a single item in this flow)
    match = matches[0] if matches else None  # the transaction being proposed
    normalized = normalize_reply(text)  # normalize the reply for matching

    if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # option 1: confirm both the transaction and new value
        updated_row, updates = apply_correction(match, new_value)  # apply the stored new value, as in State E
        finish(session)  # correction flow is complete, return this user to State A
        return [build_correction_confirmation(updated_row, updates)]  # confirm the correction

    if normalized == "2" or normalized == "otro valor":  # option 2: right transaction, wrong proposed new value
        transition(session, {"state": "E", "match": match})  # move to State E awaiting a fresh new value
        return [NEW_VALUE_PROMPT]  # ask for the new value, description, or recurrence

    if normalized == "3" or normalized == "otra":  # option 3: this isn't the right transaction at all
        transition(session, {  # move to State C to ask for the time window, clearing the stored new value
            "state": "C",  # State C: awaiting the time window
            "correction_hint": session.get("correction_hint"),  # keep the original hint from classification
            "date_range": None,  # no date range chosen yet
        })
        return [TIME_WINDOW_QUESTION]  # ask the user when the transaction happened

    return ["No entendí tu respuesta. Responde 1 (sí), 2 (otro valor), o 3 (otra)."]  # still State D


def handle_state_e(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # the new-value reply
    match = session.get("match")  # the transaction row selected for correction in State D

    if is_cancellation(text):  # check deterministically whether the user wants to abandon the correction
        transition(session, {"state": "E_CANCEL", "match": match})  # move to the sub-state asking what to cancel
        return [CANCEL_SCOPE_QUESTION]  # ask whether to cancel just the correction or the whole transaction

    normalized = normalize_reply(text)  # normalize the reply for menu lookups
    if normalized in FIELD_MENU:  # user picked "1. el monto" or "2. la descripción" — needs a follow-up value
        transition(session, {"state": "E", "match": match, "field": FIELD_MENU[normalized]})  # stay in E, remember field
        return [  # ask specifically for that field's value
            "¿Cuál es el nuevo monto?" if FIELD_MENU[normalized] == "amount" else "¿Cuál es la nueva descripción?"
        ]

    if normalized in VALUE_MENU:  # user picked a menu option that is itself a complete value (frequency/category)
        updated_row, updates = apply_correction(match, VALUE_MENU[normalized])  # apply the chosen value directly
        finish(session)  # correction flow is complete, return this user to State A
        return [build_correction_confirmation(updated_row, updates)]  # confirm the correction

    if references_frequency_without_value(text):  # user named the concept (e.g. "frecuencia") but no specific value
        transition(session, {"state": "E_FREQUENCY", "match": match})  # wait for the specific value, don't guess
        return [FREQUENCY_QUESTION]  # ask exactly which frequency they mean

    if references_category_without_value(text):  # user named the concept (e.g. "categoría") but no specific value
        transition(session, {"state": "E_CATEGORY", "match": match})  # wait for the specific value, don't guess
        return [CATEGORY_QUESTION]  # ask exactly which category they mean

    field = session.get("field")  # a prior FIELD_MENU choice ("amount"/"description") awaiting this value, if any
    if field == "amount":  # user is answering "¿Cuál es el nuevo monto?"
        parsed_amount = extract_amount_from_text(text)  # find a numeric amount among the words
        if parsed_amount is None:  # no recognizable number given
            return ["No entendí el monto. ¿Cuál es el nuevo monto?"]  # ask again, stay in State E
        updated_row, updates = db.save_correction(match, {"amount": parsed_amount}, text)  # apply the amount only,
        # the description is left untouched here — auto-regenerating it would silently discard useful detail
        # (e.g. "leche y harina" becoming a generic "Gasto de X pesos."), so ask the user instead of guessing.
        transition(session, {"state": "E_DESCRIPTION_ASK", "match": updated_row})  # offer to also fix the description
        # Two separate messages, exactly as before: the confirmation, then the follow-up question.
        return [build_correction_confirmation(updated_row, updates), DESCRIPTION_FOLLOWUP_QUESTION]
    if field == "description":  # user is answering "¿Cuál es la nueva descripción?"
        updated_row, updates = db.save_correction(match, {"description": text.strip()}, text)  # apply the description
        finish(session)  # correction flow is complete, return this user to State A
        return [build_correction_confirmation(updated_row, updates)]  # confirm the correction

    # State E is purely deterministic: any non-cancellation, non-ambiguous reply is treated directly as the new
    # value, with no Claude call, to avoid the interrupt/clarification loop that could get stuck asking forever.
    updated_row, updates = apply_correction(match, text)  # apply the new value and get back the updated row
    finish(session)  # correction flow is complete, return this user to State A
    return [build_correction_confirmation(updated_row, updates)]  # confirm the correction


def handle_state_e_frequency(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # a specific
    match = session.get("match")
    normalized = normalize_reply(text)  # normalize the reply for matching

    if is_cancellation(text):  # user wants to abandon the correction after all
        transition(session, {"state": "E_CANCEL", "match": match})  # move to the sub-state asking what to cancel
        return [CANCEL_SCOPE_QUESTION]  # ask whether to cancel just the correction or the whole transaction

    chosen = FREQUENCY_MENU.get(normalized) or extract_recurrence_from_text(text)  # a menu number or the word
    if chosen is None:  # still not a recognizable frequency value, do not guess
        return [FREQUENCY_RETRY]  # ask again, stay in State E_FREQUENCY

    updated_row, updates = apply_correction(match, chosen)  # apply the now-confirmed frequency value
    finish(session)  # correction flow is complete, return this user to State A
    return [build_correction_confirmation(updated_row, updates)]  # confirm the correction


def handle_state_e_category(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # a specific
    match = session.get("match")
    normalized = normalize_reply(text)  # normalize the reply for matching

    if is_cancellation(text):  # user wants to abandon the correction after all
        transition(session, {"state": "E_CANCEL", "match": match})  # move to the sub-state asking what to cancel
        return [CANCEL_SCOPE_QUESTION]  # ask whether to cancel just the correction or the whole transaction

    chosen = CATEGORY_MENU.get(normalized) or extract_category_from_text(text)  # a menu number or the word
    if chosen is None:  # still not a recognizable category value, do not guess
        return [CATEGORY_RETRY]  # ask again, stay in State E_CATEGORY

    updated_row, updates = apply_correction(match, chosen)  # apply the now-confirmed category value
    finish(session)  # correction flow is complete, return this user to State A
    return [build_correction_confirmation(updated_row, updates)]  # confirm the correction


def handle_state_e_description_ask(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # yes/no to
    match = session.get("match")
    normalized = normalize_reply(text)  # normalize the reply for matching

    if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # user wants to update the description too
        transition(session, {"state": "E_DESCRIPTION_VALUE", "match": match})  # wait for the new description text
        return ["¿Cuál es la nueva descripción?"]  # ask for it

    if normalized == "2" or normalized in NEGATIVE_REPLIES:  # user is fine leaving the description as-is
        finish(session)  # correction flow is complete, return this user to State A
        return ["Listo, la descripción queda igual ✅"]  # confirm nothing else changed

    return [DESCRIPTION_FOLLOWUP_RETRY]  # unrecognized reply, ask again; state is left unchanged


def handle_state_e_description_value(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # the new
    match = session.get("match")
    updated_row, updates = db.save_correction(match, {"description": text.strip()}, text)  # apply the description
    finish(session)  # correction flow is complete, return this user to State A
    return [build_correction_confirmation(updated_row, updates)]  # confirm the correction


def handle_state_e_cancel(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # cancel-scope reply
    match = session.get("match")  # the transaction that was being corrected
    normalized = normalize_reply(text)  # normalize the reply for simple keyword matching

    if normalized == "1" or "correccion" in normalized:  # de-accented, so "corrección" matches too
        db.reactivate_transaction(match["id"])  # restore status
        finish(session)  # correction is abandoned, return this user to State A
        return ["Listo, cancelé la corrección. La transacción original quedó sin cambios ✅"]

    if normalized == "2" or "transaccion" in normalized:  # de-accented, so "transacción" matches too
        db.void_transaction(match["id"])  # mark the transaction as void
        finish(session)  # transaction is void, return this user to State A
        return ["Listo, anulé la transacción ✅"]  # confirm the transaction was voided

    return ["No entendí. Responde 1 (la corrección) o 2 (la transacción)."]  # still E_CANCEL


def handle_state_r(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # a report follow-up reply
    normalized = normalize_reply(text)  # normalize the reply for matching
    today = today_local()  # reference date for "semana"/"mes"

    if normalized == "1" or "esta semana" in normalized:  # option 1: show this week's report instead
        start = week_start(today)  # Monday of the current week
        return send_period_report(channel, user_id, session, start, today, f"Resumen de esta semana — {start:%d/%m/%Y} al {today:%d/%m/%Y}")
    if normalized == "2" or "este mes" in normalized:  # option 2: show this month's report instead
        start = today.replace(day=1)  # first day of the current month
        return send_period_report(channel, user_id, session, start, today, f"Resumen de este mes — {start:%d/%m/%Y} al {today:%d/%m/%Y}")
    if normalized == "3" or "otra fecha" in normalized:  # option 3: ask for a custom start date
        transition(session, {"state": "R_DATE"})  # await the free-text date expression
        return [R_DATE_QUESTION]  # ask since when
    if normalized == "4" or normalized in NEGATIVE_REPLIES or "no, gracias" in normalized or "no gracias" in normalized:
        # option 4, or any other decline — exit cleanly, no need to reclassify a bare "no"/"4" through Claude
        finish(session)  # return to State A
        return ["OK✅"]  # confirm the decline was understood
    finish(session)  # any other message is unrelated to this menu — drop the follow-up
    return handle_state_a(channel, user_id, text, session)  # classify and handle this message normally


def handle_state_r_date(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # the free-text
    parsed = parse_date_expression(text)
    today = today_local()  # end of the report range is always today
    if parsed is not None:  # Claude confidently parsed a start date
        header = f"Resumen desde {parsed:%d/%m/%Y} al {today:%d/%m/%Y}"  # date-range header
        return send_period_report(channel, user_id, session, parsed, today, header)  # show the report
    transition(session, {"state": "R_DATE_MANUAL"})  # await a DD/MM/YYYY reply
    return [R_DATE_MANUAL_QUESTION]  # ask for the manual format


def handle_state_r_date_manual(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # a DD/MM/YYYY
    today = today_local()  # reply; end of the report range is always today
    try:  # parse the manual date format directly, no Claude call needed
        parsed = datetime.strptime(text.strip(), "%d/%m/%Y").date()  # exact DD/MM/YYYY format
    except ValueError:  # the reply still isn't a valid DD/MM/YYYY date
        return [R_DATE_MANUAL_RETRY]  # ask again, stay in State R_DATE_MANUAL

    header = f"Resumen desde {parsed:%d/%m/%Y} al {today:%d/%m/%Y}"  # date-range header
    return send_period_report(channel, user_id, session, parsed, today, header)  # show the report


def handle_state_r_debtors_followup(channel: str, user_id: str, text: str, session: dict) -> list[str]:  # sí/no to
    normalized = normalize_reply(text)
    if normalized == "1" or normalized in AFFIRMATIVE_REPLIES:  # user wants to see the creditor list too
        return send_creditor_list(channel, user_id, session)  # show it
    if normalized == "2" or normalized in NEGATIVE_REPLIES:  # user is done with reports
        finish(session)  # return to State A
        return ["OK✅"]  # confirm the decline was understood
    finish(session)  # unrelated message, drop the follow-up and process it as a new message
    return handle_state_a(channel, user_id, text, session)  # classify and handle this message normally


# Dispatch table: state name -> handler. Every handler takes (channel, user_id, text, session) and returns list[str].
STATE_HANDLERS = {
    "B": handle_state_b,  # user is answering a clarification question
    "C": handle_state_c,  # user is specifying the time window for a correction
    "D": handle_state_d,  # user is confirming which transaction to correct
    "E": handle_state_e,  # user is providing the new value for a correction
    "E_FREQUENCY": handle_state_e_frequency,  # user is naming a specific frequency after an ambiguous reference
    "E_CATEGORY": handle_state_e_category,  # user is naming a specific category after an ambiguous reference
    "E_DESCRIPTION_ASK": handle_state_e_description_ask,  # user is answering whether to also update the description
    "E_DESCRIPTION_VALUE": handle_state_e_description_value,  # user is providing the new description
    "E_CANCEL": handle_state_e_cancel,  # user is choosing what exactly to cancel mid-correction
    "R": handle_state_r,  # user is answering the post-report "quieres ver más?" follow-up
    "R_DATE": handle_state_r_date,  # user is naming a custom start date in free text
    "R_DATE_MANUAL": handle_state_r_date_manual,  # user is naming a custom start date in DD/MM/YYYY format
    "R_DEBTORS_FOLLOWUP": handle_state_r_debtors_followup,  # user is answering whether to also see the creditor list
    "OVERGUARD_CONFIRM": handle_state_overguard_confirm,  # confirming a pago_deuda/cobro with no debt on record
}


def run_state_machine(channel: str, user_id: str, text: str) -> list[str]:  # load state -> dispatch -> save state.
    # Typed messages and transcribed voice notes both land here, so a voice note is treated exactly like the same
    # words typed — including mid-flow replies ("sí" in State D confirms, it does not start a new transaction).
    #
    # State is loaded from Supabase here and written back at the end, so a restart mid-flow loses nothing and two
    # channel processes never disagree about where a user is.
    #
    # The abuse guards live in the two public entry points below, not here, so a voice note passes the gate and is
    # counted exactly once rather than being billed as both a voice note and a text message.
    text = text.strip()  # the message text with surrounding whitespace removed

    stored = db.get_user_state(channel, user_id)  # {"state", "state_data", "updated_at"} straight from Supabase
    state = stored["state"]  # the state name this user was left in

    if state_is_expired(state, stored["updated_at"]):  # they left a flow hanging for over an hour
        db.clear_user_state(channel, user_id)  # abandon it and return them to State A
        return [STATE_EXPIRED_MESSAGE]  # the triggering message is discarded; their NEXT message starts fresh
        # State A needs no notice, and state_is_expired() returns False for it, so this branch can't fire there.

    session = {"state": state, **deserialize_state_data(stored["state_data"])}  # the working copy handlers mutate

    try:  # guard the whole dispatch so a single failure doesn't crash the bot
        handler = STATE_HANDLERS.get(state)  # find the handler for this state, if any
        if handler is not None:  # a conversation flow is in progress
            replies = handler(channel, user_id, text, session)  # delegate to that state's handler
        else:  # default: no conversation flow in progress
            replies = handle_state_a(channel, user_id, text, session)  # delegate to the State A handler
    except Exception:  # catch any unexpected failure from the handlers above
        logger.exception("Failed to process message in state %s", state)  # log the full traceback for debugging
        finish(session)  # reset this user's state so they aren't stuck in a broken flow
        replies = ["Hubo un error al procesar tu mensaje. Intenta de nuevo."]  # tell the user

    # Persist whatever state the handler left behind. Every state change happens inside the dispatch above, so this
    # one write covers all of them — and it refreshes updated_at, restarting the 60-minute idle clock.
    new_state = session.get("state", "A")  # where the flow ended up
    if new_state == "A":  # the flow completed (or never started) — nothing worth keeping
        if state != "A":  # only write when there was actually something to clear, saving a round trip
            db.clear_user_state(channel, user_id)
    else:  # the user is still mid-flow, save the working data for the next message
        db.save_user_state(channel, user_id, new_state, serialize_state_data(session))

    return replies


# --------------------------------------------------------------------------------------------------
# Abuse guards, applied before any Claude/Whisper call so a stranger can never spend money.
# --------------------------------------------------------------------------------------------------

ACCESS_REQUIRED_MESSAGE = "Konta está en pruebas privadas. Si tienes un código de acceso, escríbelo aquí."
CODE_ALREADY_USED_MESSAGE = "Ese código ya fue usado. Pide uno nuevo a quien te invitó."
CODE_ACCEPTED_MESSAGE = "¡Bienvenido a Konta! ✅"

TEXT_DAILY_LIMIT = 500  # messages per user per day
VOICE_DAILY_LIMIT = 50  # voice notes per user per day
VOICE_MAX_SECONDS = 60  # anything longer is rejected before it reaches Whisper

TEXT_LIMIT_MESSAGE = "Llegaste al límite de mensajes por hoy. Vuelve mañana 👍"
VOICE_LIMIT_MESSAGE = "Llegaste al límite de notas de voz por hoy. Puedes seguir escribiendo 👍"
VOICE_TOO_LONG_MESSAGE = "Esa nota de voz es muy larga. Mándame una más corta, de menos de un minuto 🙏"


def normalize_code(text: str) -> str:  # invite codes are matched case-insensitively, ignoring stray whitespace
    return text.strip().upper()  # "  konta-a1b2 " -> "KONTA-A1B2"


def try_redeem_access(channel: str, user_id: str, text: str) -> list[str]:  # a non-allowlisted user's text is
    code = normalize_code(text)  # treated as a possible invite code
    outcome = db.redeem_code(code, channel, user_id)  # atomic: only one caller can claim a given code

    if outcome == "redeemed":  # valid and unused — admit them
        db.add_allowed_user(channel, user_id, code)  # they are now on the allowlist for good
        logger.info("Access code %s redeemed by %s/%s", code, channel, user_id)
        return [CODE_ACCEPTED_MESSAGE, *welcome_messages()]  # greeting, then the usual welcome + examples
    if outcome == "already_used":  # the code exists but somebody got there first
        return [CODE_ALREADY_USED_MESSAGE]
    return [ACCESS_REQUIRED_MESSAGE]  # not a code at all — explain the private beta


def check_voice_allowed(channel: str, user_id: str, duration_seconds: int | None) -> list[str] | None:
    # Every reason a voice note can be rejected, evaluated without the audio itself. A channel can call this
    # first to skip downloading bytes it is only going to throw away; handle_incoming_voice repeats the same
    # checks, so a channel that does not pre-check is still safe.
    if not db.is_allowed_user(channel, user_id):  # not in the private beta
        return [ACCESS_REQUIRED_MESSAGE]  # audio is never treated as a code, and never transcribed
    if duration_seconds is not None and duration_seconds > VOICE_MAX_SECONDS:  # too long to be worth transcribing
        return [VOICE_TOO_LONG_MESSAGE]
    return check_daily_limit(channel, user_id, "voice")  # None when under the daily voice quota


def check_daily_limit(channel: str, user_id: str, kind: str) -> list[str] | None:  # None means "under the limit"
    usage = db.get_usage(channel, user_id, today_local())  # counts for the vendor's local calendar day
    if kind == "text" and usage["text_count"] >= TEXT_DAILY_LIMIT:
        return [TEXT_LIMIT_MESSAGE]
    if kind == "voice" and usage["voice_count"] >= VOICE_DAILY_LIMIT:
        return [VOICE_LIMIT_MESSAGE]
    return None  # under the limit, carry on


def transcribe_voice(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str | None:  # voice note -> text via Whisper
    # Whisper accepts .ogg/opus directly, so no ffmpeg conversion is needed — the bytes go up exactly as downloaded.
    extension = mime_type.split("/")[-1] or "ogg"  # e.g. "audio/ogg" -> "ogg"; only used to name the upload
    try:  # a transcription failure must never crash the handler; the caller asks the user to type instead
        transcription = openai_client.audio.transcriptions.create(
            model=WHISPER_MODEL,  # OpenAI's speech-to-text model
            file=(f"voice.{extension}", audio_bytes, mime_type),  # (filename, bytes, MIME type) tuple the SDK expects
            language="es",  # the vendor speaks Spanish; naming it improves accuracy and avoids language drift
        )
    except Exception:  # network error, bad API key, unsupported audio, empty file, quota exhausted, ...
        logger.exception("Whisper transcription failed")  # log the full traceback for debugging
        return None  # signal failure to the caller

    text = (transcription.text or "").strip()  # the transcribed text, or an empty string if Whisper heard nothing
    return text or None  # treat a blank transcription (silence, pure noise) as a failure too


def handle_incoming_message(channel: str, user_id: str, text: str) -> list[str]:  # THE text entry point for any channel
    if not db.is_allowed_user(channel, user_id):  # not in the private beta — their text may be an invite code
        return try_redeem_access(channel, user_id, text)  # admits them, or explains why not; nothing else runs

    limited = check_daily_limit(channel, user_id, "text")  # over quota for today?
    if limited is not None:  # yes — stop before spending anything on Claude
        return limited

    replies = run_state_machine(channel, user_id, text)  # the message was accepted and processed
    db.increment_usage(channel, user_id, today_local(), "text")  # count it only now that it actually ran
    return replies


def handle_incoming_voice(  # transcribe a voice note, then process it exactly like a typed message
    channel: str, user_id: str, audio_bytes: bytes, mime_type: str = "audio/ogg",
    duration_seconds: int | None = None,  # the channel supplies this; Telegram has it on message.voice.duration
) -> list[str]:
    # Runs BEFORE transcribe_voice, so audio from a stranger — or an over-long note — never reaches Whisper and
    # never costs anything, whether or not the channel already called check_voice_allowed itself.
    blocked = check_voice_allowed(channel, user_id, duration_seconds)  # allowlist, length and quota in one call
    if blocked is not None:  # any of the three rejected it
        return blocked

    text = transcribe_voice(audio_bytes, mime_type)  # convert the audio to text via Whisper
    if text is None:  # transcription failed or produced nothing usable
        db.clear_user_state(channel, user_id)  # return this user to State A, as specified
        return [VOICE_TRANSCRIPTION_ERROR]  # ask the user to type it instead

    logger.info("Voice note from %s/%s transcribed as: %r", channel, user_id, text)  # surface it in the log
    replies = run_state_machine(channel, user_id, text)  # same dispatch a typed message takes
    db.increment_usage(channel, user_id, today_local(), "voice")  # counted as a voice note, not as text
    return replies


def welcome_messages() -> list[str]:  # the two messages a brand-new user sees, in order
    return [WELCOME_MESSAGE, EXAMPLES_MESSAGE]
