create table if not exists transactions (
    id serial primary key,
    user_id text not null,
    raw_message text,
    amount numeric,
    type text check (type in ('ingreso', 'gasto', 'préstamo', 'deuda', 'cobro', 'pago_deuda')),
    category text check (category in ('negocio', 'personal')),
    recurrence text check (recurrence in ('única vez', 'recurrente', 'variable')),
    description text,
    debtor_name text,
    creditor_name text,
    status text default 'activa',
    is_test boolean default true,
    channel text default 'telegram',
    created_at timestamp default now(),
    updated_at timestamp
);

-- Marks rows created during testing so they can be found and cleaned up separately from real vendor data
alter table transactions add column if not exists is_test boolean default true;

-- Which messaging channel a transaction arrived on. Existing rows default to 'telegram', which is correct
-- because Telegram was the only channel before this column existed. Every query filters on (channel, user_id)
-- so the same numeric user ID on two different channels can never collide.
alter table transactions add column if not exists channel text default 'telegram';

-- Postgres-level grants
grant select, insert, update on public.transactions to anon;
grant usage, select on sequence public.transactions_id_seq to anon;

-- Row Level Security policies (RLS is enabled on this table)
create policy "bot can select transactions" on public.transactions
    for select to anon using (true);

create policy "bot can insert transactions" on public.transactions
    for insert to anon with check (true);

create policy "bot can update transactions" on public.transactions
    for update to anon using (true) with check (true);

-- ---------------------------------------------------------------------------
-- Conversation state, one row per (channel, user_id). Lives in the database
-- rather than process memory so an unfinished flow survives a restart and two
-- channel processes never disagree about where a user is.
-- ---------------------------------------------------------------------------
create table if not exists user_state (
    channel text not null,
    user_id text not null,
    state text not null default 'A',
    state_data jsonb default '{}'::jsonb,
    updated_at timestamp default now(),
    primary key (channel, user_id)
);

-- Without these the anon key gets "permission denied for table user_state".
-- The bot only reads, inserts and updates (clearing a flow writes state='A'), so no delete grant is needed.
grant select, insert, update on public.user_state to anon;

-- Row Level Security policies, matching the transactions table's setup
alter table public.user_state enable row level security;

create policy "bot can select user_state" on public.user_state
    for select to anon using (true);

create policy "bot can insert user_state" on public.user_state
    for insert to anon with check (true);

create policy "bot can update user_state" on public.user_state
    for update to anon using (true) with check (true);

-- ---------------------------------------------------------------------------
-- Abuse guards: private-beta allowlist, single-use invite codes, daily quotas.
-- ---------------------------------------------------------------------------

-- One row per invite code. redeemed_at IS NULL means the code is still available;
-- redemption is a conditional update on that column, so two people sending the
-- same code at the same time cannot both win.
create table if not exists access_codes (
    code text primary key,
    label text,
    redeemed_by_channel text,
    redeemed_by_user_id text,
    redeemed_at timestamp,
    created_at timestamp default now()
);

-- Who may use the bot at all. A (channel, user_id) absent from here is blocked.
create table if not exists allowed_users (
    channel text not null,
    user_id text not null,
    code_used text,
    created_at timestamp default now(),
    primary key (channel, user_id)
);

-- Per-user daily message counts. usage_date is the Europe/Amsterdam calendar
-- date, matching the timezone the reports already use.
create table if not exists usage_counters (
    channel text not null,
    user_id text not null,
    usage_date date not null,
    text_count integer default 0,
    voice_count integer default 0,
    primary key (channel, user_id, usage_date)
);

-- Same grant/RLS pattern as user_state: read, insert and update for anon; never delete.
grant select, insert, update on public.access_codes to anon;
grant select, insert, update on public.allowed_users to anon;
grant select, insert, update on public.usage_counters to anon;

alter table public.access_codes enable row level security;
alter table public.allowed_users enable row level security;
alter table public.usage_counters enable row level security;

create policy "bot can select access_codes" on public.access_codes
    for select to anon using (true);
create policy "bot can insert access_codes" on public.access_codes
    for insert to anon with check (true);
create policy "bot can update access_codes" on public.access_codes
    for update to anon using (true) with check (true);

create policy "bot can select allowed_users" on public.allowed_users
    for select to anon using (true);
create policy "bot can insert allowed_users" on public.allowed_users
    for insert to anon with check (true);
create policy "bot can update allowed_users" on public.allowed_users
    for update to anon using (true) with check (true);

create policy "bot can select usage_counters" on public.usage_counters
    for select to anon using (true);
create policy "bot can insert usage_counters" on public.usage_counters
    for insert to anon with check (true);
create policy "bot can update usage_counters" on public.usage_counters
    for update to anon using (true) with check (true);

-- The existing Telegram vendor, seeded so enabling the allowlist cannot lock them out.
insert into allowed_users (channel, user_id, code_used)
values ('telegram', '7960646753', 'seed-existing-user')
on conflict (channel, user_id) do nothing;

-- Period index for faster report queries
create index if not exists idx_transactions_user_created
    on transactions(user_id, created_at, status);

-- Channel-scoped variant of the index above, matching the (channel, user_id) filter every query now applies
create index if not exists idx_transactions_channel_user_created
    on transactions(channel, user_id, created_at, status);
