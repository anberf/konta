# Supabase data layer. Every function here is channel-aware: rows are keyed by (channel, user_id) so a Telegram
# user and a WhatsApp user with the same numeric ID can never see each other's transactions.
#
# This module contains NO Telegram imports and NO Spanish user-facing text — it only reads and writes rows.

import logging  # standard library module used to log query diagnostics
import os  # standard library module used to read environment variables
from datetime import date, datetime, timedelta, timezone  # date/time utilities for range boundaries and timestamps
from zoneinfo import ZoneInfo  # IANA timezone database lookup, used to resolve "today" in the vendor's local time

from dotenv import load_dotenv  # loads variables from the local .env file into the environment
from supabase import create_client, Client  # Supabase client used to read and write the transactions table

load_dotenv()  # populate os.environ from .env when running locally; on Railway the variables are already set

logger = logging.getLogger(__name__)  # module-level logger, configured by the entry point

SUPABASE_URL = os.environ.get("SUPABASE_URL")  # Supabase project REST URL
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")  # Supabase anon API key

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)  # Supabase client instance used for all database ops

LOCAL_TZ = ZoneInfo("Europe/Amsterdam")  # the vendor's local timezone, used to resolve "today" for reports


def today_local() -> date:  # today's calendar date in the vendor's local timezone (created_at is stored as UTC)
    return datetime.now(LOCAL_TZ).date()


def now_iso() -> str:  # produce the current UTC time as a naive ISO 8601 string, matching the DB column type
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()  # strip the timezone before formatting


def _scoped(query, channel: str, user_id: str):  # apply the (channel, user_id) scope every query must carry
    return query.eq("channel", channel).eq("user_id", str(user_id))  # both together identify one person on one channel


def save_transaction(channel: str, user_id: str, raw_message: str, classification: dict) -> None:  # insert a new row
    supabase.table("transactions").insert(  # build an insert request against the transactions table
        {
            "channel": channel,  # which messaging channel this transaction arrived on
            "user_id": str(user_id),  # the channel's user ID, stored as text
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


def search_transactions(  # active transactions in a date range, optionally narrowed to an exact amount
    channel: str, user_id: str, start: datetime, end: datetime, amount: float | None = None
) -> list[dict]:
    query = (  # start building the Supabase query
        supabase.table("transactions")  # target the transactions table
        .select("*")  # select every column
        .eq("status", "activa")  # exclude voided ("anulada") transactions
        .gte("created_at", start.isoformat())  # only transactions on or after the start of the range
        .lt("created_at", end.isoformat())  # only transactions strictly before the end of the range
    )
    query = _scoped(query, channel, user_id)  # restrict to this channel + user
    if amount is not None:  # the caller resolved the hint to a number, filter by exact amount at the database level
        query = query.eq("amount", amount)  # amounts have no "spelling", so an exact match is correct here
    result = query.order("created_at", desc=True).execute()  # run the query, newest transactions first
    return result.data or []  # the matching rows, or an empty list if none


def get_all_active_transactions(channel: str, user_id: str) -> list[dict]:  # every active row, newest first
    result = (  # the caller applies its own typo-tolerant description filtering on top of this
        _scoped(supabase.table("transactions").select("*"), channel, user_id)
        .eq("status", "activa")  # only consider active (not already voided) transactions
        .order("created_at", desc=True)  # newest first
        .execute()  # run the query
    )
    return result.data or []  # every active transaction for this channel + user


def get_most_recent_transaction(channel: str, user_id: str) -> dict | None:  # single most recent active transaction
    result = (  # look up the most recent active transaction
        _scoped(supabase.table("transactions").select("*"), channel, user_id)
        .eq("status", "activa")  # only consider active (not already voided) transactions
        .order("created_at", desc=True)  # newest first
        .limit(1)  # only need the single most recent row
        .execute()  # run the query
    )
    rows = result.data or []  # the matching rows, or an empty list if none
    return rows[0] if rows else None  # return the row, or None if this user has no active transaction


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


def get_period_report(channel: str, user_id: str, start_date, end_date) -> dict:  # summary over an inclusive date range
    # start_date/end_date are LOCAL (Europe/Amsterdam) calendar dates; created_at is stored as naive UTC, so local
    # midnight must be converted to UTC before comparing — a fixed UTC-midnight boundary would be off by 1-2 hours.
    start_local = datetime.combine(start_date, datetime.min.time(), tzinfo=LOCAL_TZ)  # local midnight, start_date
    end_local = datetime.combine(end_date, datetime.min.time(), tzinfo=LOCAL_TZ) + timedelta(days=1)  # local midnight, day after end_date
    start = start_local.astimezone(timezone.utc).replace(tzinfo=None)  # equivalent naive UTC instant
    end = end_local.astimezone(timezone.utc).replace(tzinfo=None)  # equivalent naive UTC instant

    logger.info(  # debug: the exact filter being sent to Supabase, in both UTC and local terms
        "get_period_report channel=%s user_id=%s local_range=[%s 00:00, %s 00:00) Europe/Amsterdam -> "
        "SQL-equivalent: created_at >= '%s' AND created_at < '%s' (UTC)",
        channel, user_id, start_date.isoformat(), (end_date + timedelta(days=1)).isoformat(),
        start.isoformat(), end.isoformat(),
    )

    result = (  # look up every active transaction created within the range
        _scoped(supabase.table("transactions").select("*"), channel, user_id)
        .eq("status", "activa")  # exclude voided ("anulada") transactions
        .gte("created_at", start.isoformat())  # only transactions on or after the start of the range
        .lt("created_at", end.isoformat())  # only transactions strictly before the end of the range
        .execute()  # run the query
    )
    rows = result.data or []  # the matching rows, or an empty list if none
    return calculate_report_totals(rows)  # aggregate them into the report totals


def get_daily_report(channel: str, user_id: str, date) -> dict:  # income/expense summary for a single calendar day
    return get_period_report(channel, user_id, date, date)  # a single day is just a one-day-long period


def get_debtor_list(channel: str, user_id: str) -> tuple[list[dict], list[dict]]:  # (still owes you, you overpaid)
    result = (  # look up every active préstamo/cobro transaction for this user
        _scoped(supabase.table("transactions").select("*"), channel, user_id)
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


def get_creditor_list(channel: str, user_id: str) -> tuple[list[dict], list[dict]]:  # (you still owe, you overpaid)
    result = (  # look up every active deuda/pago_deuda transaction for this user
        _scoped(supabase.table("transactions").select("*"), channel, user_id)
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


def get_creditor_balance(channel: str, user_id: str, creditor_name: str, category: str | None) -> float:  # one
    result = (  # creditor's balance in one category — deuda minus pago_deuda, restricted to that creditor and category
        _scoped(supabase.table("transactions").select("type,amount"), channel, user_id)
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


def get_debtor_balance(channel: str, user_id: str, debtor_name: str) -> float:  # one debtor's balance across both
    result = (  # categories — préstamo minus cobro (matches get_debtor_list's category-agnostic model)
        _scoped(supabase.table("transactions").select("type,amount"), channel, user_id)
        .eq("status", "activa")
        .eq("debtor_name", debtor_name)
        .in_("type", ["préstamo", "cobro"])
        .execute()
    )
    rows = result.data or []  # the matching rows, or an empty list if none
    return sum(  # préstamo increases the balance owed, cobro decreases it
        (row.get("amount") or 0) if row.get("type") == "préstamo" else -(row.get("amount") or 0) for row in rows
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


def parse_timestamp(value) -> datetime:  # parse a Supabase timestamp into a naive-UTC datetime
    if isinstance(value, datetime):  # already a datetime (some client versions parse it for us)
        return value.replace(tzinfo=None) if value.tzinfo else value  # normalize to naive
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))  # tolerate a trailing "Z"
    return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed  # naive UTC


def get_user_state(channel: str, user_id: str) -> dict:  # this user's conversation state, or a State A default
    result = (  # one row per (channel, user_id) — the table's primary key
        _scoped(supabase.table("user_state").select("*"), channel, user_id).limit(1).execute()
    )
    rows = result.data or []  # the stored row, or nothing if this user has never had state saved
    if not rows:  # no row yet — this user is in State A and has never started a flow
        return {"state": "A", "state_data": {}, "updated_at": datetime.now(timezone.utc).replace(tzinfo=None)}
    row = rows[0]  # the stored state row
    return {
        "state": row.get("state") or "A",  # the state name, defaulting to State A
        "state_data": row.get("state_data") or {},  # the flow's working data, as stored JSON
        "updated_at": parse_timestamp(row["updated_at"]) if row.get("updated_at") else
                      datetime.now(timezone.utc).replace(tzinfo=None),  # when this state was last written
    }


def save_user_state(channel: str, user_id: str, state: str, state_data: dict) -> None:  # upsert this user's state
    supabase.table("user_state").upsert(  # insert, or overwrite the existing row for this (channel, user_id)
        {
            "channel": channel,  # part of the composite primary key
            "user_id": str(user_id),  # part of the composite primary key
            "state": state,  # the state name the flow is now in
            "state_data": state_data,  # JSON-serializable working data for that flow
            "updated_at": now_iso(),  # written from the app clock so the expiry check compares like with like
        },
        on_conflict="channel,user_id",  # the primary key, so a repeat message updates rather than duplicates
    ).execute()


def clear_user_state(channel: str, user_id: str) -> None:  # return this user to State A with no working data
    save_user_state(channel, user_id, "A", {})  # keep the row, reset its contents


# --------------------------------------------------------------------------------------------------
# Abuse guards: private-beta allowlist, single-use invite codes, per-user daily quotas.
# --------------------------------------------------------------------------------------------------

def is_allowed_user(channel: str, user_id: str) -> bool:  # may this person use the bot at all?
    result = _scoped(supabase.table("allowed_users").select("user_id"), channel, user_id).limit(1).execute()
    return bool(result.data)  # a row here is the allowlist entry; absent means blocked


def add_allowed_user(channel: str, user_id: str, code_used: str | None) -> None:  # admit someone to the beta
    supabase.table("allowed_users").upsert(  # upsert so re-admitting an existing user is harmless
        {"channel": channel, "user_id": str(user_id), "code_used": code_used},
        on_conflict="channel,user_id",  # the primary key
    ).execute()


def redeem_code(code: str, channel: str, user_id: str) -> str:  # "redeemed" | "already_used" | "not_found"
    # The update is conditional on redeemed_at still being NULL, so the database itself decides the winner:
    # if two people send the same code at the same instant, exactly one update matches a row.
    claimed = (
        supabase.table("access_codes")
        .update({"redeemed_by_channel": channel, "redeemed_by_user_id": str(user_id), "redeemed_at": now_iso()})
        .eq("code", code)  # this specific code
        .is_("redeemed_at", "null")  # ...but only while it is still unredeemed
        .execute()
    )
    if claimed.data:  # our update matched a row, so this caller won the race
        return "redeemed"
    exists = supabase.table("access_codes").select("code").eq("code", code).limit(1).execute()
    return "already_used" if exists.data else "not_found"  # distinguish a spent code from a wrong one


def get_usage(channel: str, user_id: str, usage_date) -> dict:  # today's message counts for this user
    result = (
        _scoped(supabase.table("usage_counters").select("text_count,voice_count"), channel, user_id)
        .eq("usage_date", usage_date.isoformat())  # the Europe/Amsterdam calendar date
        .limit(1)
        .execute()
    )
    rows = result.data or []  # no row yet means nothing has been counted today
    if not rows:
        return {"text_count": 0, "voice_count": 0}
    return {"text_count": rows[0].get("text_count") or 0, "voice_count": rows[0].get("voice_count") or 0}


def increment_usage(channel: str, user_id: str, usage_date, kind: str) -> None:  # kind is "text" or "voice"
    # Read-then-upsert. A user would have to send two messages in the same instant to lose a count, and the
    # worst case is one message not counted against their own daily quota — acceptable for a rate limit.
    current = get_usage(channel, user_id, usage_date)  # what has been counted so far today
    counts = {
        "text_count": current["text_count"] + (1 if kind == "text" else 0),
        "voice_count": current["voice_count"] + (1 if kind == "voice" else 0),
    }
    supabase.table("usage_counters").upsert(
        {"channel": channel, "user_id": str(user_id), "usage_date": usage_date.isoformat(), **counts},
        on_conflict="channel,user_id,usage_date",  # the primary key
    ).execute()


def void_transaction(transaction_id) -> None:  # mark one transaction as voided ("anulada")
    supabase.table("transactions").update(  # void that transaction
        {"status": "anulada", "updated_at": now_iso()}  # mark it voided and refresh the update timestamp
    ).eq("id", transaction_id).execute()  # apply the update to that specific row


def reactivate_transaction(transaction_id) -> None:  # restore a transaction's "activa" status
    supabase.table("transactions").update({"status": "activa"}).eq("id", transaction_id).execute()
