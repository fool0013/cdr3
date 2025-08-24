import argparse, csv
from collections import Counter

AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDRO = set("VILFWY")

def has_only_std_aa(s): return set(s) <= AA
def has_nglyc(seq):
    for i in range(len(seq)-2):
        if seq[i] == "N" and seq[i+1] != "P" and seq[i+2] in ("S","T"):
            return True
    return False
def max_hydrophobic_run(seq):
    run = best = 0
    for ch in seq:
        if ch in HYDRO: run += 1; best = max(best, run)
        else: run = 0
    return best
def net_charge(seq):
    c = Counter(seq)
    return (c["K"] + c["R"] + 0.1*c["H"]) - (c["D"] + c["E"])

def hard_fail_reason(seq):
    if not has_only_std_aa(seq): return "non_std_aa"
    if seq.count("C") > 2: return "too_many_cys"     # allow up to 2 total Cys
    if has_nglyc(seq): return "NXS_T_motif"
    if max_hydrophobic_run(seq) >= 6: return "hydrophobic_run>=6"
    if abs(net_charge(seq)) > 12: return "charge_out_of_range"
    return ""

def main(inp, outp):
    rows = []
    with open(inp, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seq = row["cdr3_seq"].strip()
            reason = hard_fail_reason(seq)
            row["length"] = len(seq)
            row["charge"] = f"{net_charge(seq):.1f}"
            row["hydro_run"] = max_hydrophobic_run(seq)
            row["has_nglyc"] = has_nglyc(seq)
            row["warn_no_leading_C"] = (seq[:1] != "C")
            row["warn_len"] = (len(seq) < 9 or len(seq) > 22)
            row["passes"] = "Y" if reason == "" else "N"
            row["fail_reason"] = reason
            rows.append(row)

    # sort: pass first then by score desc
    rows.sort(key=lambda x: (x["passes"] != "Y", -float(x["score"])))
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {sum(r['passes'] == 'Y' for r in rows)} passing / {len(rows)} total -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.inp, args.out)
