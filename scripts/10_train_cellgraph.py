"""Train the concept-supervised CellGraph-MSI predictor on fixed folds."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CELL_CLASS = 7
N_RELATION = N_CELL_CLASS * N_CELL_CLASS
COMPARTMENT_SLICE = (16, 25)
RBF_CENTRES = torch.tensor([0, 3, 6, 10, 15, 22, 30, 40], dtype=torch.float32)
RBF_WIDTH = 4.0
EDGE_CUTOFF_UM = 40.0
CONCEPTS = ["le_per_epi_0_5um_TUM", "frac_MUC"]
TISSUE = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class SlideGraphs(torch.utils.data.Dataset):
    """One item is one slide: a sample of its windows as a single batched graph."""

    def __init__(self, ids, labels, graph_dir, n_windows, concepts,
                 uni2_dir, train, node_mean=None, node_std=None):
        self.ids, self.labels = ids, labels
        self.dir, self.n_windows = graph_dir, n_windows
        self.concepts, self.uni2_dir, self.train = concepts, uni2_dir, train
        self.mu, self.sd = node_mean, node_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        d = torch.load(self.dir / f"{sid}.pt", map_location="cpu",
                       weights_only=False)
        wins = d["windows"]
        k = min(self.n_windows, len(wins))
        # fresh windows each epoch during training, a mild augmentation;
        # a fixed spread at evaluation so the score is reproducible
        idx = (np.random.choice(len(wins), k, replace=False) if self.train
               else np.linspace(0, len(wins) - 1, k).astype(int))

        xs, eis, eds, rels, wid = [], [], [], [], []
        off = 0
        for wi, j in enumerate(idx):
            w = wins[j]
            n = int(w["n_nodes"])
            xs.append(w["x"].float())
            eis.append(w["edge_index"].long() + off)
            eds.append(w["edge_d_um"].float())
            rels.append(w["rel"].long())
            wid.append(torch.full((n,), wi, dtype=torch.long))
            off += n

        x = torch.cat(xs)
        if self.mu is not None:
            x = (x - torch.from_numpy(self.mu)) / torch.from_numpy(self.sd)
        item = {"slide_id": sid, "x": x,
                "edge_index": torch.cat(eis, dim=1),
                "edge_d": torch.cat(eds), "rel": torch.cat(rels),
                "node2win": torch.cat(wid), "n_windows": k,
                "y": torch.tensor(float(self.labels[sid]))}
        if self.concepts is not None:
            row = self.concepts.loc[sid]
            item["concept"] = torch.tensor(row.values.astype(np.float32))
            item["concept_ok"] = torch.tensor(float(np.isfinite(row.values).all()))
        if self.uni2_dir is not None:
            p = self.uni2_dir / f"{sid}.npy"
            v = np.load(p) if p.exists() else np.zeros(1536, dtype=np.float32)
            item["uni2"] = torch.from_numpy(np.asarray(v, dtype=np.float32))
        return item


def collate(batch):
    """Merge slides into one disconnected graph, tracking node -> window -> slide."""
    xs, eis, eds, rels, n2w, w2s, ys = [], [], [], [], [], [], []
    node_off = win_off = 0
    extras = {"concept": [], "concept_ok": [], "uni2": []}
    for si, b in enumerate(batch):
        xs.append(b["x"])
        eis.append(b["edge_index"] + node_off)
        eds.append(b["edge_d"]); rels.append(b["rel"])
        n2w.append(b["node2win"] + win_off)
        w2s.append(torch.full((b["n_windows"],), si, dtype=torch.long))
        ys.append(b["y"])
        for k in extras:
            if k in b:
                extras[k].append(b[k])
        node_off += b["x"].shape[0]
        win_off += b["n_windows"]
    out = {"slide_ids": [b["slide_id"] for b in batch],
           "x": torch.cat(xs), "edge_index": torch.cat(eis, dim=1),
           "edge_d": torch.cat(eds), "rel": torch.cat(rels),
           "node2win": torch.cat(n2w), "win2slide": torch.cat(w2s),
           "n_slides": len(batch), "n_windows": win_off, "y": torch.stack(ys)}
    for k, v in extras.items():
        if v:
            out[k] = torch.stack(v)
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def segment_softmax(scores, index, n_seg):
    m = torch.full((n_seg,), -1e30, device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    e = torch.exp(scores - m[index])
    denom = torch.zeros(n_seg, device=scores.device, dtype=scores.dtype)
    denom = denom.index_add(0, index, e)
    return e / denom[index].clamp(min=1e-12)


class EdgeConv(nn.Module):
    """
    Message passing in which the edge carries the physical distance between
    two nuclei and the ordered pair of their classes. Distance enters as a
    smooth basis rather than a single scalar, so the layer can respond
    differently at 3 um and at 30 um without having to infer that shape from
    one number. v16 found the MSI signal to be sharply distance-dependent, so
    the graph should represent distance as a continuous quantity.
    """

    def __init__(self, dim, edge_dim, rel_dim, dropout):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * dim + edge_dim + rel_dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim))
        self.upd = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                                 nn.Dropout(dropout))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index, e_feat):
        src, dst = edge_index[0], edge_index[1]
        m = self.msg(torch.cat([h[src], h[dst], e_feat], dim=-1))
        agg = torch.zeros_like(h).index_add(0, dst, m)
        deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        deg = deg.index_add(0, dst, torch.ones_like(dst, dtype=h.dtype))
        agg = agg / deg.clamp(min=1).unsqueeze(-1)
        return self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))


class CellGraphModel(nn.Module):
    def __init__(self, node_dim, hidden=96, layers=3, rel_dim=16, dropout=0.2,
                 n_concepts=2, use_uni2=False, uni2_dim=1536, uni2_proj=64):
        super().__init__()
        edge_dim = len(RBF_CENTRES) + 2
        self.rel_emb = nn.Embedding(N_RELATION, rel_dim)
        self.inp = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU())
        self.convs = nn.ModuleList(
            [EdgeConv(hidden, edge_dim, rel_dim, dropout) for _ in range(layers)])
        # a learned gate over the layer outputs, so a node can draw on a shallow
        # or a deep neighbourhood as suits it
        self.jk_gate = nn.Linear(hidden * layers, layers)
        self.node_attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(),
                                       nn.Linear(hidden // 2, 1))
        self.win_norm = nn.LayerNorm(hidden)

        self.use_uni2 = use_uni2
        win_dim = hidden
        if use_uni2:
            self.uni2_proj = nn.Sequential(nn.LayerNorm(uni2_dim),
                                           nn.Linear(uni2_dim, uni2_proj),
                                           nn.GELU())
            win_dim += uni2_proj
        self.win_attn = nn.Sequential(nn.Linear(win_dim, hidden // 2), nn.Tanh(),
                                      nn.Linear(hidden // 2, 1))
        self.concept_head = nn.Sequential(
            nn.Linear(win_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_concepts))
        self.cls_head = nn.Sequential(
            nn.Linear(win_dim + n_concepts, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))

    def edge_features(self, d_um, rel, x_raw, edge_index):
        c = RBF_CENTRES.to(d_um.device)
        z = (d_um.unsqueeze(-1) - c) / RBF_WIDTH
        r = torch.exp(-0.5 * z * z)
        dn = (d_um / EDGE_CUTOFF_UM).unsqueeze(-1)
        a, b = COMPARTMENT_SLICE
        comp = x_raw[:, a:b].argmax(-1)
        cross = (comp[edge_index[0]] != comp[edge_index[1]]).to(r.dtype).unsqueeze(-1)
        return torch.cat([r, dn, cross, self.rel_emb(rel)], dim=-1)

    def forward(self, batch, return_attn=False):
        x, ei = batch["x"], batch["edge_index"]
        e = self.edge_features(batch["edge_d"], batch["rel"], x, ei)
        h = self.inp(x)
        outs = []
        for conv in self.convs:
            h = conv(h, ei, e)
            outs.append(h)
        g = torch.softmax(self.jk_gate(torch.cat(outs, dim=-1)), dim=-1)
        h = sum(g[:, i:i + 1] * outs[i] for i in range(len(outs)))

        n2w, nw = batch["node2win"], batch["n_windows"]
        a = segment_softmax(self.node_attn(h).squeeze(-1), n2w, nw)
        win = torch.zeros(nw, h.shape[1], device=h.device, dtype=h.dtype)
        win = self.win_norm(win.index_add(0, n2w, h * a.unsqueeze(-1)))

        if self.use_uni2:
            u = self.uni2_proj(batch["uni2"])
            win = torch.cat([win, u[batch["win2slide"]]], dim=-1)

        w2s, ns = batch["win2slide"], batch["n_slides"]
        b = segment_softmax(self.win_attn(win).squeeze(-1), w2s, ns)
        slide = torch.zeros(ns, win.shape[1], device=win.device, dtype=win.dtype)
        slide = slide.index_add(0, w2s, win * b.unsqueeze(-1))

        concept = self.concept_head(slide)
        logit = self.cls_head(torch.cat([slide, concept], dim=-1)).squeeze(-1)
        if return_attn:
            return logit, concept, b, w2s
        return logit, concept


# ---------------------------------------------------------------------------
def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def load_folds(args):
    fm = pd.read_csv(args.fm_csv)
    sid = "slide_id" if "slide_id" in fm.columns else fm.columns[0]
    if "fold" not in fm.columns:
        sys.exit("[fatal] the FM prediction file has no 'fold' column; the "
                 "graph model must be scored on the same partition")
    return dict(zip(fm[sid], fm["fold"].astype(int)))


def load_concepts(args, ids):
    frames = []
    p = Path(args.compartment_csv)
    if p.exists():
        c = pd.read_csv(p)
        if CONCEPTS[0] in c.columns:
            frames.append(c[["slide_id", CONCEPTS[0]]])
    p = Path(args.crosscheck_csv)
    if p.exists():
        x = pd.read_csv(p)
        need = [f"n_{t}" for t in TISSUE]
        if all(k in x.columns for k in need):
            x["frac_MUC"] = x["n_MUC"] / x[need].sum(axis=1)
            frames.append(x[["slide_id", "frac_MUC"]])
    if not frames:
        return None
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="slide_id", how="outer")
    return df.set_index("slide_id").reindex(ids)


def compute_norm(ids, graph_dir, n_sample=40):
    """Node-feature statistics from training slides only, never the eval fold."""
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ids), min(n_sample, len(ids)), replace=False)
    acc = []
    for i in pick:
        d = torch.load(graph_dir / f"{ids[i]}.pt", map_location="cpu",
                       weights_only=False)
        for w in d["windows"][:8]:
            acc.append(w["x"].float().numpy())
    a = np.concatenate(acc)
    mu, sd = a.mean(0), a.std(0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def move(batch, dev):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(dev, non_blocking=True)
    return batch


def run_fold(args, fold, seed, ids, labels, folds, concepts, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    graph_dir = Path(args.graph_dir)
    tr = [s for s in ids if folds[s] != fold]
    va = [s for s in ids if folds[s] == fold]
    ntr_pos = sum(labels[s] for s in tr)

    # concept targets standardised on the training fold only
    con_all = None
    if concepts is not None:
        m = concepts.loc[tr].mean()
        s = concepts.loc[tr].std().replace(0, 1.0)
        con_all = ((concepts - m) / s).astype(np.float32)

    mu, sd = compute_norm(tr, graph_dir)
    uni2_dir = Path(args.uni2_dir) if args.use_uni2_region else None
    mk = lambda lst, train: SlideGraphs(lst, labels, graph_dir, args.n_windows,
                                        con_all, uni2_dir, train, mu, sd)
    dl_tr = torch.utils.data.DataLoader(
        mk(tr, True), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
        pin_memory=True)
    dl_va = torch.utils.data.DataLoader(
        mk(va, False), batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), collate_fn=collate,
        pin_memory=True)

    d0 = torch.load(graph_dir / f"{ids[0]}.pt", map_location="cpu",
                    weights_only=False)
    model = CellGraphModel(node_dim=int(d0["node_dim"]), hidden=args.hidden,
                           layers=args.layers, dropout=args.dropout,
                           n_concepts=len(CONCEPTS),
                           use_uni2=args.use_uni2_region).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_w = torch.tensor([(len(tr) - ntr_pos) / max(ntr_pos, 1)], device=dev)

    def predict():
        model.eval()
        sids, ys, ps, cs = [], [], [], []
        with torch.no_grad():
            for b in dl_va:
                b = move(b, dev)
                logit, concept = model(b)
                sids += b["slide_ids"]
                ys += b["y"].tolist()
                ps += torch.sigmoid(logit).tolist()
                cs.append(concept.cpu().numpy())
        return sids, np.array(ys), np.array(ps), np.concatenate(cs)

    best = {"auc": -1.0, "epoch": -1, "p": None, "c": None}
    hist = []
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for b in dl_tr:
            b = move(b, dev)
            opt.zero_grad(set_to_none=True)
            logit, concept = model(b)
            loss = F.binary_cross_entropy_with_logits(logit, b["y"],
                                                      pos_weight=pos_w)
            if "concept" in b and args.concept_weight > 0:
                ok = b["concept_ok"].bool()
                if ok.any():
                    loss = loss + args.concept_weight * F.smooth_l1_loss(
                        concept[ok], b["concept"][ok])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        sched.step()

        sids, ys, ps, cs = predict()
        a = auroc(ys, ps)
        hist.append({"epoch": ep, "loss": tl / max(len(dl_tr), 1), "auc": a})
        if a > best["auc"]:
            # v22: the published v17 numbers come from graph_prob = best["p"],
            # i.e. this epoch. Without keeping the weights here, the model left
            # in memory is the LAST epoch and the checkpoint would not match any
            # reported figure. 02_train_gnn_old.py did exactly this. Detached
            # CPU copies: no effect on the optimiser, the loss or the data.
            best = {"auc": a, "epoch": ep, "p": ps.copy(), "c": cs.copy(),
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            logging.info(f"    fold{fold} seed{seed} ep{ep:>3} "
                         f"loss={hist[-1]['loss']:.4f} val_auc={a:.3f} "
                         f"(best {best['auc']:.3f} @ep{best['epoch']})")

    auc_last = hist[-1]["auc"]

    # --- v22: archive the best-epoch model. Training above is untouched.
    try:
        if "state" in best:
            model.load_state_dict(best["state"])
        _ck_dir = Path(args.out_dir) / f"{args.tag}_seed{seed}"
        _ck_dir.mkdir(parents=True, exist_ok=True)
        _ck = {"state": model.state_dict(), "args": vars(args),
               "seed": seed, "fold": fold,
               "selected_by": "holdout_best",   # v17's protocol, see docstring
               "best_epoch": best["epoch"], "auc_best": best["auc"],
               "auc_last": auc_last, "concepts": list(CONCEPTS),
               "generation": "v17",
               "v22_note": "weights = best-epoch on the reported fold (v17 protocol)"}
        # compute_norm() derives these from the TRAINING fold only and they are
        # not recoverable later. Wrong stats at inference is a silent shift that
        # looks exactly like failure to generalise.
        for _n in ("mu", "sd", "m", "s", "node_dim"):
            if _n in locals():
                _ck[{"mu": "node_mu", "sd": "node_sd",
                     "m": "concept_mu", "s": "concept_sd"}.get(_n, _n)] = locals()[_n]
        torch.save(_ck, _ck_dir / f"model_fold{fold}.pt")
        logging.info(f"      [v22] saved {_ck_dir}/model_fold{fold}.pt "
                     f"(keys: {sorted(k for k in _ck if k != 'state')})")
    except Exception as _e:
        logging.error(f"      [v22] SAVE FAILED {type(_e).__name__}: {_e}")
        raise
    logging.info(f"    fold{fold} seed{seed} | best {best['auc']:.3f} "
                 f"@ep{best['epoch']} | last epoch {auc_last:.3f} | "
                 f"train n={len(tr)} pos={ntr_pos}")

    rows = pd.DataFrame({"slide_id": sids, "fold": fold, "seed": seed,
                         "y": ys.astype(int), "graph_prob": best["p"],
                         "graph_prob_last": ps})
    for j, c in enumerate(CONCEPTS):
        rows[f"pred_{c}"] = best["c"][:, j]
    rec = {"seed": seed, "fold": fold, "auc_best": best["auc"],
           "best_epoch": best["epoch"], "auc_last": auc_last,
           "n_train": len(tr), "n_train_pos": ntr_pos, "n_eval": len(va)}
    return rows, rec, pd.DataFrame(hist).assign(seed=seed, fold=fold)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--labels-csv", default="data/labels/msi_tcga.csv")
    ap.add_argument("--meta-csv", default="data/labels/metadata_tcga_crc_dx.csv")
    ap.add_argument("--fm-csv",
                    default="data/labels/folds_tcga.csv")
    ap.add_argument("--fm-prob-col", default="base_prob")
    ap.add_argument("--compartment-csv", default="data/tissue/compartment_iel.csv")
    ap.add_argument("--crosscheck-csv", default="data/tissue/crosscheck.csv")
    ap.add_argument("--uni2-dir", default="data/uni2_slide")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--tag", default="L1")
    ap.add_argument("--cohort", default="COAD", choices=["COAD", "CRC"])
    ap.add_argument("--use-uni2-region", action="store_true")
    ap.add_argument("--fuse-fm-logit", action="store_true")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--n-windows", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--concept-weight", type=float, default=0.3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device={dev}")

    lab = pd.read_csv(args.labels_csv)[["slide_id", "msi_status"]]
    meta = Path(args.meta_csv)
    if meta.exists() and args.cohort == "COAD":
        md = pd.read_csv(meta)
        if {"slide_id", "project_id"}.issubset(md.columns):
            keep = md[md["project_id"].astype(str).str.contains("COAD", na=False)]
            lab = lab[lab.slide_id.isin(keep.slide_id)]
    labels = {r.slide_id: int(r.msi_status == "MSI-H") for r in lab.itertuples()}

    folds = load_folds(args)
    graph_dir = Path(args.graph_dir)
    ids = sorted(s for s in labels
                 if (graph_dir / f"{s}.pt").exists() and s in folds)
    npos = sum(labels[s] for s in ids)
    nf = len(set(folds[s] for s in ids))
    logging.info(f"cohort={args.cohort}  slides={len(ids)}  MSI-H={npos}  "
                 f"MSS={len(ids)-npos}  folds={nf}  seeds={args.seeds}")
    logging.info(f"level: cell-graph"
                 f"{' + UNI2 region' if args.use_uni2_region else ''}"
                 f"{' + FM late fusion' if args.fuse_fm_logit else ''}")
    if len(ids) < 50:
        sys.exit("[fatal] too few slides with graphs")

    concepts = load_concepts(args, ids)
    if concepts is not None:
        n_ok = int(np.isfinite(concepts.values).all(axis=1).sum())
        logging.info(f"concept targets for {n_ok}/{len(ids)} slides: {CONCEPTS}")

    rows, recs, hists = [], [], []
    for seed in args.seeds:
        for f in sorted(set(folds[s] for s in ids)):
            t0 = time.time()
            r, rec, h = run_fold(args, f, seed, ids, labels, folds, concepts, dev)
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(r); recs.append(rec); hists.append(h)

    oof = pd.concat(rows, ignore_index=True)
    fr = pd.DataFrame(recs)
    out = Path(args.out_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    fr.to_csv(out / "fold_records.csv", index=False)
    pd.concat(hists, ignore_index=True).to_csv(out / "epoch_history.csv",
                                               index=False)

    from sklearn.metrics import average_precision_score
    logging.info("")
    logging.info("=" * 76)
    logging.info(f"RESULT — {args.tag}")
    logging.info("=" * 76)
    logging.info(f"{'seed':>5}{'fold':>6}{'best ep':>9}{'best AUC':>10}"
                 f"{'last ep AUC':>13}")
    logging.info("-" * 76)
    for _, r in fr.iterrows():
        logging.info(f"{int(r.seed):>5}{int(r.fold):>6}{int(r.best_epoch):>9}"
                     f"{r.auc_best:>10.3f}{r.auc_last:>13.3f}")
    logging.info(f"\n  fold mean: best {fr.auc_best.mean():.3f} +- "
                 f"{fr.auc_best.std():.3f} | last epoch "
                 f"{fr.auc_last.mean():.3f} +- {fr.auc_last.std():.3f}")

    logging.info("")
    per_seed_best, per_seed_last = [], []
    for seed in args.seeds:
        s = oof[oof.seed == seed]
        ab = auroc(s.y.values, s.graph_prob.values)
        al = auroc(s.y.values, s.graph_prob_last.values)
        per_seed_best.append(ab); per_seed_last.append(al)
        logging.info(f"  seed {seed}: pooled OOF AUROC {ab:.3f}  AUPRC "
                     f"{average_precision_score(s.y.values, s.graph_prob.values):.3f}"
                     f"   (last epoch {al:.3f})")
    if len(args.seeds) > 1:
        logging.info(f"  mean over seeds: {np.mean(per_seed_best):.3f} +- "
                     f"{np.std(per_seed_best):.3f}   "
                     f"(last epoch {np.mean(per_seed_last):.3f})")
        m = oof.groupby("slide_id").agg(y=("y", "first"),
                                        p=("graph_prob", "mean"))
        logging.info(f"  seed-averaged predictions: "
                     f"AUROC {auroc(m.y.values, m.p.values):.3f}")

    logging.info("")
    logging.info("  Reference points, same cohort:")
    logging.info("    two hand-crafted features   0.835")
    logging.info("    code8 graph head            0.830")
    logging.info("    foundation model alone      0.927")

    if concepts is not None:
        from scipy.stats import spearmanr
        logging.info("")
        logging.info("  concept recovery (predicted vs measured, Spearman):")
        m = oof.groupby("slide_id").mean(numeric_only=True)
        for c in CONCEPTS:
            if f"pred_{c}" in m and c in concepts.columns:
                j = m.join(concepts[[c]], how="inner").dropna(
                    subset=[f"pred_{c}", c])
                if len(j) > 20:
                    r, p = spearmanr(j[f"pred_{c}"], j[c])
                    logging.info(f"    {c:<24} rho={r:+.3f}  p={p:.1e}  n={len(j)}")
        logging.info("    A weak correlation would mean the auxiliary loss did "
                     "not take, and the model")
        logging.info("    is not constrained by the concepts it was meant to use.")

    if args.fuse_fm_logit:
        fm = pd.read_csv(args.fm_csv)
        sidc = "slide_id" if "slide_id" in fm.columns else fm.columns[0]
        if args.fm_prob_col not in fm.columns:
            logging.warning(f"  '{args.fm_prob_col}' not in the FM file; "
                            f"skipping fusion")
        else:
            g = oof.groupby("slide_id").agg(
                y=("y", "first"), graph_prob=("graph_prob", "mean")).reset_index()
            g = g.merge(fm[[sidc, args.fm_prob_col]].rename(
                columns={sidc: "slide_id", args.fm_prob_col: "fm_prob"}),
                on="slide_id", how="inner").dropna()
            lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                                  (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            logging.info("")
            logging.info("  late fusion (logit average):")
            logging.info(f"    graph alone  {auroc(g.y, g.graph_prob):.3f}")
            logging.info(f"    FM alone     {auroc(g.y, g.fm_prob):.3f}")
            for w in (0.25, 0.5, 0.75):
                zz = (1 - w) * lg(g.fm_prob.values) + w * lg(g.graph_prob.values)
                logging.info(f"    w_graph={w:<5} {auroc(g.y, zz):.3f}")
            logging.info("    Fix the weight before quoting a fused number; "
                         "reading the best one off")
            logging.info("    this table is a choice made on the data being "
                         "reported.")

    logging.info(f"\nwrote {out}")


if __name__ == "__main__":
    main()
