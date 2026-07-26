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
    created_at timestamp default now(),
    updated_at timestamp
);

-- Marks rows created during testing so they can be found and cleaned up separately from real vendor data
alter table transactions add column if not exists is_test boolean default true;

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

-- Period index for faster report queries
create index if not exists idx_transactions_user_created
    on transactions(user_id, created_at, status);
