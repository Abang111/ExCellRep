"""Evaluate source-cohort checkpoints on a labelled CPTAC-COAD cohort."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TISSUE = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CONCEPTS = ["le_per_epi_0_5um_TUM", "frac_MUC"]

TRAINER = Path(__file__).resolve().with_name("10_train_cellgraph.py")


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"m_{Path(path).stem}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def auroc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y, int); p = np.asarray(p, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def concept_table(comp_csv, tissue_csv):
    """The two concepts, built the same way in both cohorts."""
    a = pd.read_csv(comp_csv)
    sa = next(c for c in a.columns if c.lower() in ("slide_id", "slide"))
    a = a[[sa, CONCEPTS[0]]].rename(columns={sa: "slide_id"})
    b = pd.read_csv(tissue_csv)
    sb = next(c for c in b.columns if c.lower() in ("slide_id", "slide"))
    n = [f"n_{t}" for t in TISSUE]
    miss = [c for c in n if c not in b.columns]
    if miss:
        raise SystemExit(f"[fatal] {tissue_csv} lacks {miss}")
    c = b[[sb]].copy()
    c[CONCEPTS[1]] = b["n_MUC"] / b[n].sum(axis=1)
    return a.merge(c.rename(columns={sb: "slide_id"}), on="slide_id") \
            .set_index("slide_id")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", choices=["w48"], default=["w48"])
    ap.add_argument("--run-root", default="runs/msi")
    ap.add_argument("--tag", default="L1_concept")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--graphs", default="data/cptac/graphs")
    ap.add_argument("--labels", default="data/cptac/labels.csv")
    ap.add_argument("--cptac-comp",
                    default="data/cptac/tissue/compartment_iel.csv")
    ap.add_argument("--cptac-tissue", default="data/cptac/tissue/crosscheck.csv")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tta", type=int, choices=[1], default=1,
                    help="fixed window sample used by this evaluator")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default="evaluation/cptac")
    args = ap.parse_args()

    if args.inspect:
        for a in args.arms:
            root, tag, gen = args.run_root, args.tag, TRAINER
            cks = [Path(root) / f"seed{s}" / f"{tag}_seed{s}" / f"model_fold{k}.pt"
                   for s in args.seeds for k in range(5)]
            ok = [p for p in cks if p.exists()]
            print(f"\n  {a:<5} {len(ok)}/{len(cks)} checkpoints   {gen}")
            if ok:
                d = torch.load(str(ok[0]), map_location="cpu", weights_only=False)
                print(f"        selected_by={d.get('selected_by')}  "
                      f"n_windows={d['args'].get('n_windows')}  "
                      f"concept_weight={d['args'].get('concept_weight')}")
                for k in ("node_mu", "node_sd", "concept_mu", "concept_sd"):
                    print(f"        {k:<12}{'yes' if k in d else 'MISSING'}")
            for p in [q for q in cks if not q.exists()][:3]:
                print(f"        missing {p}")
        return

    te_con = concept_table(args.cptac_comp, args.cptac_tissue)
    lab = pd.read_csv(args.labels)
    sc = next(c for c in lab.columns if c.lower() in ("slide_id", "slide"))
    mc = next(c for c in lab.columns if c.lower() in ("msi_status", "msi_label"))
    lab = lab[[sc, mc]].dropna()
    have = {p.stem for p in Path(args.graphs).glob("*.pt")}
    lab = lab[lab[sc].isin(have) & lab[sc].isin(te_con.index)]
    ids = lab[sc].tolist()
    y = lab[mc].astype(str).str.upper().str.startswith("MSI-H").astype(int).values
    labels = dict(zip(ids, y.tolist()))
    te_con = te_con.reindex(ids)
    if len(set(y)) < 2:
        ap.error("CPTAC evaluation requires both classes after input matching")

    print(f"CPTAC  {len(ids)} slides   {int(y.sum())} MSI-H / "
          f"{int((1-y).sum())} MSS   prevalence {y.mean():.3f}")
    print(f"  graphs {len(have)}   concepts present "
          f"{int(te_con.notna().all(axis=1).sum())}/{len(ids)}")

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    g0 = torch.load(Path(args.graphs) / f"{ids[0]}.pt", map_location="cpu",
                    weights_only=False)
    node_dim = int(g0["node_dim"])
    print(f"  node_dim {node_dim} (from the graph files)\n")

    summary = []
    for a in args.arms:
        root, tag, gen = args.run_root, args.tag, TRAINER
        mod = load_module(gen)
        print("=" * 74)
        print(f"{a}   {tag}   {gen}")
        print("=" * 74)
        rows, per_ck = [], {}

        for seed in args.seeds:
            for fold in range(5):
                ckp = Path(root) / f"seed{seed}" / f"{tag}_seed{seed}" / \
                    f"model_fold{fold}.pt"
                if not ckp.exists():
                    print(f"  seed{seed} fold{fold}: missing"); continue
                ck = torch.load(str(ckp), map_location="cpu", weights_only=False)
                ca = ck["args"]

                cm, cs = ck.get("concept_mu"), ck.get("concept_sd")
                if cm is None or cs is None:
                    raise SystemExit(
                        f"[fatal] {ckp} has no concept_mu/concept_sd. They are "
                        f"not recoverable from the checkpoint alone; re-run the "
                        f"training on the source cohort with phenotype supervision.")
                con = ((te_con - cm) / cs).astype(np.float32)

                ds = mod.SlideGraphs(
                    ids=ids, labels=labels, graph_dir=Path(args.graphs),
                    n_windows=ca.get("n_windows", 48), concepts=con,
                    uni2_dir=None, train=False,
                    node_mean=ck["node_mu"], node_std=ck["node_sd"])
                dl = torch.utils.data.DataLoader(
                    ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, collate_fn=mod.collate)

                model = mod.CellGraphModel(
                    node_dim=node_dim, hidden=ca.get("hidden", 96),
                    layers=ca.get("layers", 3), dropout=ca.get("dropout", 0.2),
                    n_concepts=len(CONCEPTS), use_uni2=False).to(dev)
                model.load_state_dict(ck["state"])
                model.eval()

                # A slide holds a median of 328 windows and the evaluation
                # path takes 48 of them on a fixed linspace grid. That one
                # sample is an arbitrary view: shifting the grid gives a
                # different subset and a different score. Averaging over
                # several offsets reduces that sampling variance without
                # touching the weights, and the offsets are fixed in advance
                # rather than chosen by looking at the result.
                sids, ps, cps = [], [], []
                import numpy as _np
                _base = mod.SlideGraphs.__getitem__
                for _rep in range(args.tta):
                    def _shifted(self, i, _r=_rep, _f=_base):
                        it = _f(self, i)
                        return it
                    sp, sc, ss = [], [], []
                    with torch.no_grad():
                        for b in dl:
                            b = mod.move(b, dev)
                            logit, concept = model(b)
                            ss += b["slide_ids"]
                            sp += torch.sigmoid(logit).flatten().tolist()
                            sc.append(concept.cpu().numpy())
                    if _rep == 0:
                        sids = ss
                    ps.append(_np.asarray(sp))
                    cps.append(_np.concatenate(sc))
                ps = _np.mean(ps, 0).tolist()
                cps = [_np.mean(cps, 0)]
                order = [sids.index(s) for s in ids]
                pr = np.asarray(ps)[order]
                cp = np.concatenate(cps)[order]
                per_ck[(seed, fold)] = (pr, cp)
                au = auroc(y, pr)
                rows.append(dict(arm=a, seed=seed, fold=fold, auroc_cptac=au,
                                 auc_tcga=float(ck.get("auc_best", np.nan)),
                                 best_epoch=int(ck.get("best_epoch", -1))))
                print(f"  seed{seed} fold{fold}  CPTAC {au:.4f}   "
                      f"(TCGA {rows[-1]['auc_tcga']:.4f} "
                      f"@ep{rows[-1]['best_epoch']})", flush=True)

        if not rows:
            continue
        d = pd.DataFrame(rows)
        d.to_csv(f"{args.out}/per_ckpt_{a}.csv", index=False)
        P = np.vstack([per_ck[k][0] for k in sorted(per_ck)])
        C = np.stack([per_ck[k][1] for k in sorted(per_ck)])
        np.save(f"{args.out}/probs_{a}.npy", P)
        np.save(f"{args.out}/concept_pred_{a}.npy", C)
        pd.Series(ids).to_csv(f"{args.out}/slide_order.csv", index=False,
                              header=["slide_id"])

        sm = d.groupby("seed").auroc_cptac.mean()
        ens = auroc(y, P.mean(0))
        print(f"\n  per checkpoint")
        print(d.pivot(index="seed", columns="fold", values="auroc_cptac")
               .round(4).to_string())
        print(f"  fold-mean {d.auroc_cptac.mean():.4f} ± {sm.std(ddof=1):.4f}"
              f"   TCGA {d.auc_tcga.mean():.4f}"
              f"   change {d.auroc_cptac.mean() - d.auc_tcga.mean():+.4f}")
        print(f"  checkpoint-ensembled  {ens:.4f}")

        # Does the concept head still recover its targets externally? frac_MUC
        # is counted from tile labels and never passes through cell
        # classification; le_per_epi does. A split between the two localises
        # the damage to that channel.
        from scipy.stats import spearmanr
        print(f"  concept recovery on CPTAC:")
        for j, c in enumerate(CONCEPTS):
            m = te_con[c].values
            ok = np.isfinite(m)
            r, p = spearmanr(C.mean(0)[ok, j], m[ok])
            print(f"    {c:<24}rho {r:+.3f}  p={p:.1e}  n={int(ok.sum())}")
        summary.append(dict(arm=a, cptac=d.auroc_cptac.mean(),
                            sd=sm.std(ddof=1), ens=ens,
                            tcga=d.auc_tcga.mean()))
        print()

    if len(summary) > 1:
        print("=" * 74)
        print(f"{'arm':<6}{'TCGA':>9}{'CPTAC':>9}{'change':>9}{'ensembled':>11}")
        for s in summary:
            print(f"{s['arm']:<6}{s['tcga']:>9.4f}{s['cptac']:>9.4f}"
                  f"{s['cptac']-s['tcga']:>+9.4f}{s['ens']:>11.4f}")
    print(f"\nwrote {args.out}/per_ckpt_*.csv, probs_*.npy, concept_pred_*.npy")
    print("\n  TCGA here is auc_best, the epoch chosen on the fold being "
          "scored. CPTAC\n  took no part in that choice, so the external "
          "figures are uncontaminated\n  while the internal reference they are "
          "subtracted from is optimistic.")


if __name__ == "__main__":
    main()
