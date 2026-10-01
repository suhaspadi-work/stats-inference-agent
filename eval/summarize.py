"""
Run after eval/run_all.py completes (or partially completes). Reads
eval_results.jsonl and reports the aggregate statistics that individual
scenario checks can't capture on their own -- principally the A/A
false-positive rate, which only means something in aggregate.
"""
import json


def main():
    rows = []
    with open("eval_results.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    aa_rows = [r for r in rows if r["scenario"].startswith("aa_seed_")]
    effect_rows = [r for r in rows if r["scenario"].startswith("effect_")]
    messy_rows = [r for r in rows if r["scenario"].startswith("messy_")]

    print(f"Total scenarios recorded: {len(rows)}\n")

    if aa_rows:
        rejections = sum(1 for r in aa_rows if "rejected=True" in r["reason"])
        rate = rejections / len(aa_rows)
        print(f"A/A false-positive rate: {rejections}/{len(aa_rows)} = {rate:.1%}")
        sane = 0.0 <= rate <= 0.15
        print(f"  {'OK' if sane else 'FLAG'} -- expected roughly 5%, sane range is 0-15% given only {len(aa_rows)} trials")

    if effect_rows:
        passed = sum(1 for r in effect_rows if r["passed"])
        print(f"\nEffect-detection scenarios: {passed}/{len(effect_rows)} passed")
        for r in effect_rows:
            print(f"  {r['scenario']}: {'PASS' if r['passed'] else 'FAIL'} -- {r['reason']}")

    if messy_rows:
        passed = sum(1 for r in messy_rows if r["passed"])
        print(f"\nMessy-data scenarios: {passed}/{len(messy_rows)} passed")
        for r in messy_rows:
            print(f"  {r['scenario']}: {'PASS' if r['passed'] else 'FAIL'} -- {r['reason']}")

    overall_passed = sum(1 for r in rows if r["passed"])
    print(f"\nOverall: {overall_passed}/{len(rows)} scenarios passed")


if __name__ == "__main__":
    main()