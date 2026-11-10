"""Launch independent seeds with one output directory per seed."""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

TRAINER = Path(__file__).resolve().with_name("10_train_cellgraph.py")

def seed_dir(out_root, seed):
    return Path(out_root) / f"seed{seed}"


def status(out_root, seed, tag):
    """done / partial / pending, from what is on disk."""
    d = seed_dir(out_root, seed)
    csv = d / tag / "fold_records.csv"
    cks = sorted((d / f"{tag}_seed{seed}").glob("*.pt"))
    if csv.exists():
        try:
            import pandas as pd
            n = len(pd.read_csv(csv))
        except Exception:
            n = 0
        if n >= 5:
            return "done", len(cks), n
    return ("partial" if cks else "pending"), len(cks), 0


def launch(seed, gpu, out_root, tag, trainer, extra):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # the trainer sees one visible device, so it must address it as cuda:0
    cmd = [sys.executable, "-u", str(trainer),
           "--tag", tag, "--seeds", str(seed),
           "--out-dir", str(seed_dir(out_root, seed))] + extra
    log = Path(out_root).parent / f"log_{tag}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p, fh, log


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--tag", default="L1_concept")
    ap.add_argument("--out-root", default="runs/msi")
    ap.add_argument("--trainer", default=str(TRAINER))
    ap.add_argument("--force", action="store_true",
                    help="rerun seeds already marked done")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="passed through to the trainer; leave empty to keep "
                         "the trainer defaults")
    args = ap.parse_args()

    trainer = Path(args.trainer)
    if not trainer.exists():
        ap.error(f"Trainer not found: {trainer}")

    print(f"trainer  {trainer}")
    print(f"tag      {args.tag}    selection: holdout_best (v17 protocol)")
    print(f"out      {args.out_root}/seed<n>/")
    print(f"gpus     {args.gpus}")
    if args.extra:
        print(f"extra    {args.extra}   <-- not a v17 default; check this is "
              f"intended")
    print()

    todo = []
    for s in args.seeds:
        st, n_ck, n_rows = status(args.out_root, s, args.tag)
        mark = {"done": "done", "partial": "partial", "pending": "pending"}[st]
        note = f"{n_ck} ckpt"
        if st == "done":
            note += f", {n_rows} fold rows"
        elif st == "partial":
            note += ", no fold_records.csv -> restarts from fold 0"
        print(f"  seed{s:<3}{mark:<10}{note}")
        if st != "done" or args.force:
            todo.append(s)

    if not todo:
        print("\nnothing to do; pass --force to rerun")
        return
    print(f"\nto run: {todo}")
    if args.dry_run:
        for index, s in enumerate(todo):
            gpu = args.gpus[index % len(args.gpus)]
            command = [sys.executable, "-u", str(trainer), "--tag", args.tag,
                       "--seeds", str(s), "--out-dir", str(seed_dir(args.out_root, s)),
                       *args.extra]
            print(f"  CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}")
        return

    q, live = list(todo), {}
    failed = []
    t0 = time.time()
    print(f"\n{len(q)} seed-run(s) over {len(args.gpus)} GPU(s), five folds each\n")
    try:
        while q or live:
            for g in args.gpus:
                if g in live and live[g][0].poll() is not None:
                    p, fh, log, s = live.pop(g)
                    fh.close()
                    st, n_ck, _ = status(args.out_root, s, args.tag)
                    if p.returncode or st != "done":
                        failed.append(s)
                    print(f"  done   seed={s} gpu={g} exit={p.returncode} "
                          f"{st} {n_ck} ckpt   [{(time.time()-t0)/60:.0f} min]",
                          flush=True)
                if g not in live and q:
                    s = q.pop(0)
                    p, fh, log = launch(s, g, args.out_root, args.tag,
                                        trainer, args.extra)
                    live[g] = (p, fh, log, s)
                    print(f"  start  seed={s} gpu={g}   {log}", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\ninterrupted; running seeds are left alone. Re-run this script "
              "to pick up\nwhatever did not finish.")
        return

    if failed:
        sys.exit(f"Failed seeds: {failed}. See training logs.")

    print(f"\nfinished in {(time.time()-t0)/60:.0f} min\n")
    import pandas as pd
    print(f"{'seed':<7}{'ckpt':>6}{'pooled OOF':>13}   fold-mean best")
    for s in args.seeds:
        st, n_ck, _ = status(args.out_root, s, args.tag)
        log = Path(args.out_root).parent / f"log_{args.tag}_seed{s}.log"
        got = None
        if log.exists():
            m = re.findall(r"pooled OOF AUROC\s+(\d\.\d+)", log.read_text())
            got = float(m[-1]) if m else None
        fm = ""
        fr = seed_dir(args.out_root, s) / args.tag / "fold_records.csv"
        if fr.exists():
            r = pd.read_csv(fr)
            fm = f"{r.auc_best.mean():.4f} ± {r.auc_best.std(ddof=1):.4f}"
        line = f"seed{s:<3}{n_ck:>6}"
        if got is not None:
            line += f"{got:>13.4f}"
            line += f"   {fm}"
        else:
            line += f"{'no pooled OOF in log':>13}"
        print(line)

    print(f"  Checkpoints carry selected_by=holdout_best with node_mu/node_sd "
          f"and\n  concept_mu/concept_sd, which is everything CPTAC inference "
          f"needs.")


if __name__ == "__main__":
    main()
