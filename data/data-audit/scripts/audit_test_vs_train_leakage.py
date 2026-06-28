#!/usr/bin/env python3
import sys


MESSAGE = """The old all-in-one leakage audit script has been retired.

Use the separated entry points instead:

  python data/data-audit/scripts/audit_composition_leakage.py
  python data/data-audit/scripts/audit_prototype_leakage.py --workers 4
  python data/data-audit/scripts/audit_structure_similarity_leakage.py
"""


def main() -> None:
    print(MESSAGE, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
