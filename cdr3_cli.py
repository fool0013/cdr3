# i fucking edited this file like 50 fucking times because it kept breaking when i called for steps
import os, sys, json, time, subprocess
from pathlib import Path

# ---------- colored output (optional) ----------
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
except Exception:
    class _N: RESET_ALL=""; CYAN=""; GREEN=""; YELLOW=""; RED=""
    Fore = Style = _N()

# ---------- State I/O ----------
STATE_FILE = "ui_state.json"

DEFAULTS = {
    "antigen": "",                  # paste full target protein sequence here
    "seed_cdr3": "CARDRSTGYVYFDYW",
    "esm": "t6_8M",                 # "t12_35M" also valid, use for more accuracy
    "checkpoint": "score_model.pt",
    "steps": 200,
    "beam": 50,
    "k_mut": 1,
    "keep_top": 50,
    "embed_batch": 64,
    "clusters": 12,
    "out_folder": "runs",
    # last outputs
    "last_raw": "",
    "last_filtered": "",
    "last_panel": ""
}

def load_state():
    if Path(STATE_FILE).exists():
        try:
            return {**DEFAULTS, **json.loads(Path(STATE_FILE).read_text())}
        except Exception:
            pass
    return DEFAULTS.copy()

def save_state(S):
    Path(STATE_FILE).write_text(json.dumps(S, indent=2))

def ensure_out_folder(S):
    out = S.get("out_folder") or "runs"
    S["out_folder"] = out
    os.makedirs(out, exist_ok=True)
    save_state(S)
    return out

# ---------- UI helpers ----------
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def header(title):
    print(Fore.CYAN + "="*30 + Style.RESET_ALL)
    print(Fore.CYAN + f" {title}" + Style.RESET_ALL)
    print(Fore.CYAN + "="*30 + Style.RESET_ALL)

def press_enter():
    try:
        input("\nPress Enter to continue…")
    except EOFError:
        pass

def show_config(S):
    print(Fore.YELLOW + "Current config:" + Style.RESET_ALL)
    print(f"  Antigen length : {len(S.get('antigen',''))}")
    print(f"  Seed CDR3      : {S['seed_cdr3']}")
    print(f"  ESM backbone   : {S['esm']}")
    print(f"  Checkpoint     : {S['checkpoint']}")
    print(f"  Steps/Beam/k   : {S['steps']} / {S['beam']} / {S['k_mut']}")
    print(f"  Keep top       : {S['keep_top']}")
    print(f"  Embed batch    : {S['embed_batch']}")
    print(f"  Clusters (k)   : {S['clusters']}")
    print(f"  Output folder  : {S['out_folder']}")

def run_py(script, *args, check=True):
    cmd = [sys.executable, script, *map(str, args)]
    print(Fore.CYAN + "$ " + " ".join(cmd) + Style.RESET_ALL)
    return subprocess.run(cmd, check=check)

# ---------- actions (fuck you step_* names) ----------
def action_edit(S):
    clear(); header("Edit config"); show_config(S)
    print("\nEdit which field?")
    print(" 1) Antigen sequence (paste)")
    print(" 2) Seed CDR3")
    print(" 3) ESM model (t6_8M or t12_35M)")
    print(" 4) Checkpoint path")
    print(" 5) Steps")
    print(" 6) Beam width")
    print(" 7) Mutations per step (k_mut)")
    print(" 8) Keep top")
    print(" 9) Embed batch size")
    print("10) Clusters (k)")
    print("11) Output folder")
    print("12) Done")
    while True:
        ch = input("> ").strip()
        if ch == "1":
            S["antigen"] = input("Paste antigen (single line, no spaces):\n> ").strip()
        elif ch == "2":
            S["seed_cdr3"] = input("Seed CDR3:\n> ").strip()
        elif ch == "3":
            v = input("ESM backbone [t6_8M/t12_35M]: ").strip()
            if v in ("t6_8M","t12_35M"): S["esm"] = v
        elif ch == "4":
            S["checkpoint"] = input("Checkpoint path:\n> ").strip()
        elif ch == "5":
            S["steps"] = int(input("Steps: ").strip() or S["steps"])
        elif ch == "6":
            S["beam"] = int(input("Beam width: ").strip() or S["beam"])
        elif ch == "7":
            S["k_mut"] = int(input("Mutations per step (k_mut): ").strip() or S["k_mut"])
        elif ch == "8":
            S["keep_top"] = int(input("Keep top: ").strip() or S["keep_top"])
        elif ch == "9":
            S["embed_batch"] = int(input("Embed batch: ").strip() or S["embed_batch"])
        elif ch == "10":
            S["clusters"] = int(input("Clusters (k): ").strip() or S["clusters"])
        elif ch == "11":
            S["out_folder"] = input("Output folder: ").strip() or S["out_folder"]
        elif ch == "12":
            break
        else:
            print("Enter 1–12")
        save_state(S); clear(); header("Edit config"); show_config(S)
    save_state(S)

def action_optimize(S):
    clear(); header("Optimize (generate candidates)")
    if not S.get("antigen"):
        print(Fore.RED + "No antigen set. Edit config first." + Style.RESET_ALL); press_enter(); return
    out_dir = ensure_out_folder(S)
    out_csv = str(Path(out_dir, f"opt_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    try:
        run_py("optimize_cdr3.py",
               "--antigen", S["antigen"],
               "--start_cdr3", S["seed_cdr3"],
               "--steps", S["steps"],
               "--beam", S["beam"],
               "--topk", 20,
               "--k_mut", S["k_mut"],
               "--ckpt", S["checkpoint"],
               "--out_csv", out_csv)
        print(Fore.GREEN + f"Wrote {out_csv}" + Style.RESET_ALL)
        S["last_raw"] = out_csv; save_state(S)
    except subprocess.CalledProcessError:
        print(Fore.RED + "Optimization failed." + Style.RESET_ALL)
    press_enter()

def action_filter(S):
    clear(); header("Filter candidates")
    if not S.get("last_raw"):
        print(Fore.RED + "No raw candidate CSV. Run optimize first." + Style.RESET_ALL); press_enter(); return
    inp = S["last_raw"]
    out_dir = ensure_out_folder(S)
    out_csv = str(Path(out_dir, f"{Path(inp).stem}_filtered.csv"))
    try:
        run_py("filter_cdr3s.py", "--inp", inp, "--out", out_csv, "--keep_top", S["keep_top"])
        print(Fore.GREEN + f"Wrote {out_csv}" + Style.RESET_ALL)
        S["last_filtered"] = out_csv; save_state(S)
    except subprocess.CalledProcessError:
        print(Fore.RED + "Filtering failed." + Style.RESET_ALL)
    press_enter()

def action_cluster(S):
    clear(); header("Cluster into panel")
    if not S.get("last_filtered"):
        print(Fore.RED + "No filtered CSV. Run filtering first." + Style.RESET_ALL); press_enter(); return
    inp = S["last_filtered"]
    out_dir = ensure_out_folder(S)
    out_csv = str(Path(out_dir, f"{Path(inp).stem}_panel.csv"))
    try:
        run_py("cluster_candidates.py",
               "--inp", inp, "--out", out_csv,
               "--k", S["clusters"], "--esm", S["esm"])
        print(Fore.GREEN + f"Wrote {out_csv}" + Style.RESET_ALL)
        S["last_panel"] = out_csv; save_state(S)
    except subprocess.CalledProcessError:
        print(Fore.RED + "Clustering failed." + Style.RESET_ALL)
    press_enter()

def action_quick(S):
    # 2 -> 3 -> 4 pipeline
    action_optimize(S)
    if S.get("last_raw"): action_filter(S)
    if S.get("last_filtered"): action_cluster(S)

def action_outputs(S):
    clear(); header("Outputs")
    print("Last candidates CSV : ", S.get("last_raw") or "(none)")
    print("Last filtered CSV   : ", S.get("last_filtered") or "(none)")
    print("Last panel CSV      : ", S.get("last_panel") or "(none)")
    press_enter()

def action_evaluate(S):
    clear(); header("Evaluate")
    # global AUC
    try:
        print(Fore.YELLOW + "[AUC] eval_auc.py" + Style.RESET_ALL)
        out = subprocess.check_output([sys.executable, "eval_auc.py"], text=True)
        print(out.strip())
    except subprocess.CalledProcessError as e:
        print(Fore.RED + "eval_auc.py failed:\n" + (e.output or "") + Style.RESET_ALL)

    # Seed vs Panel (needs panel + antigen + seed)
    if not S.get("last_panel"):
        print(Fore.RED + "Missing panel CSV (run clustering first)." + Style.RESET_ALL)
    elif not S.get("antigen") or not S.get("seed_cdr3"):
        print(Fore.RED + "Need antigen and seed_cdr3 set in config." + Style.RESET_ALL)
    else:
        try:
            print(Fore.YELLOW + "[Seed vs Panel] seed_vs_panel.py" + Style.RESET_ALL)
            out = subprocess.check_output(
                [sys.executable, "seed_vs_panel.py",
                 S["antigen"], S["seed_cdr3"], S["last_panel"]],
                text=True
            )
            print(out.strip())
        except subprocess.CalledProcessError as e:
            print(Fore.RED + "seed_vs_panel.py failed:\n" + (e.output or "") + Style.RESET_ALL)
    press_enter()

# ---------- main loop ----------
def main():
    S = load_state()
    ensure_out_folder(S)  # guarantee folder exists

    while True:
        clear()
        header("CDR3 Optimizer - CLI UI")
        show_config(S)
        print("\nMenu:")
        print(" 1) Edit config")
        print(" 2) Optimize (generate candidates)")
        print(" 3) Filter candidates")
        print(" 4) Cluster into panel + export FASTA")
        print(" 5) Quick pipeline (2 -> 3 -> 4)")
        print(" 6) Show last outputs")
        print(" 7) Evaluate (global AUC + seed vs panel)")
        print(" 8) Quit")
        choice = input("> ").strip()

        if   choice == "1": action_edit(S)
        elif choice == "2": action_optimize(S)
        elif choice == "3": action_filter(S)
        elif choice == "4": action_cluster(S)
        elif choice == "5": action_quick(S)
        elif choice == "6": action_outputs(S)
        elif choice == "7": action_evaluate(S)
        elif choice == "8": print("Bye."); return
        else:
            print(Fore.RED + "Invalid choice." + Style.RESET_ALL); time.sleep(0.8)

if __name__ == "__main__":
    main()
