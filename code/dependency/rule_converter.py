import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Create a skeleton dependency-aware rule file from dep-test significant results."
    )
    parser.add_argument("--significant_csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rules = []
    with open(args.significant_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("rule_id") or "").strip()
            if not rid:
                continue
            # dep-test summary does not contain machine-readable if/then clauses.
            # Keep a traceable skeleton so the final rule can be filled explicitly.
            rules.append(
                {
                    "id": rid,
                    "type": "control",
                    "if": {},
                    "then": {},
                    "prior_fail_exist": 0.5,
                    "source_summary": {
                        "experiment_label": row.get("experiment_label", ""),
                        "objective": row.get("objective", ""),
                        "direction": row.get("direction", ""),
                        "relative_change": row.get("relative_change", ""),
                        "p_value": row.get("p_value", ""),
                    },
                }
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rules": rules}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rules)} skeleton online rules to {args.out}")


if __name__ == "__main__":
    main()
