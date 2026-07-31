# Mint single-use invite codes for the private beta. Run locally, then hand the printed codes to testers.
#
#   python generate_codes.py 5                        -> 5 unlabelled codes
#   python generate_codes.py 3 --label "feria julio"  -> 3 codes sharing one label
#   python generate_codes.py --label "Ana" --label "Luis"   -> one code per label
#
# A code is spent the first time someone sends it to the bot; a second person sending the same code is refused.

import argparse  # standard library command-line argument parsing
import secrets  # cryptographically strong randomness, so codes are not guessable
import sys  # standard library module used to exit with a non-zero status on failure

import db  # the Supabase data layer (also loads .env)

ALPHABET = "ABCDEFGHIJKLMNPQRSTUVWXYZ23456789"  # no O/0 or I/1 — these get read aloud and typed by hand
CODE_LENGTH = 4  # KONTA-XXXX
PREFIX = "KONTA-"


def random_code() -> str:  # one candidate code, e.g. "KONTA-7FQ2"
    return PREFIX + "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))


def existing_codes() -> set[str]:  # every code already minted, so we never hand out a duplicate
    rows = db.supabase.table("access_codes").select("code").execute().data or []
    return {row["code"] for row in rows}


def mint(count: int, labels: list[str]) -> list[tuple[str, str | None]]:  # generate `count` unused codes
    taken = existing_codes()  # collision check against what is already in the table
    minted: list[tuple[str, str | None]] = []
    for index in range(count):
        for _ in range(100):  # retry on the (unlikely) chance of a collision
            code = random_code()
            if code not in taken:
                break
        else:  # 100 collisions in a row means the keyspace is effectively full
            raise RuntimeError("could not find an unused code — increase CODE_LENGTH")
        taken.add(code)  # reserve it within this run too
        label = labels[index] if index < len(labels) else (labels[0] if len(labels) == 1 else None)
        minted.append((code, label))
    return minted


def main() -> int:
    parser = argparse.ArgumentParser(description="Mint single-use Konta invite codes.")
    parser.add_argument("count", nargs="?", type=int, help="how many codes to mint (default: one per --label)")
    parser.add_argument("--label", action="append", default=[],
                        help="note stored with the code (repeat for one label per code)")
    args = parser.parse_args()

    count = args.count if args.count is not None else max(len(args.label), 1)
    if count < 1:
        print("count must be at least 1")
        return 1

    minted = mint(count, args.label)  # generate them, checked against the table for collisions
    db.supabase.table("access_codes").insert(  # one insert for the whole batch
        [{"code": code, "label": label} for code, label in minted]
    ).execute()

    print(f"Minted {len(minted)} code(s):\n")
    for code, label in minted:
        print(f"  {code}" + (f"   ({label})" if label else ""))
    print("\nEach code admits exactly one person, the first time it is sent to the bot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
