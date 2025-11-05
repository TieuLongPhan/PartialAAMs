from __future__ import annotations
import argparse
import io
import os
import sys
import time
import logging
from typing import Optional
import pandas as pd
import re
from synkit.IO import load_from_pickle, save_to_pickle
from synkit.Graph.Hyrogen.hextend import HExtend



DEFAULT_IN = "./Data/hydrogen.pkl.gz"
DEFAULT_OUT = "./Data/hydrogen_hextend.pkl.gz"
DEFAULT_LOG = "./Data/hextend_run.log"

# -------------------------
# utilities
# -------------------------
def setup_root_logger(master_log_path: str, level=logging.INFO) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    # file handler
    fh = logging.FileHandler(master_log_path, mode="a")
    fh.setFormatter(formatter)
    root.addHandler(fh)
    # console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from math import isnan

def holm_adjust(pvals):
    """Holm (step-down) correction for a list/array of p-values; returns adjusted pvals in same order."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.full(n, np.nan)
    for i, idx in enumerate(order):
        if np.isnan(pvals[idx]):
            adjusted[idx] = np.nan
        else:
            adjusted[idx] = min(1.0, pvals[idx] * (n - i))
    # enforce monotonicity (non-decreasing in sorted order)
    # then re-map to original order
    sorted_adj = adjusted[order]
    for i in range(1, n):
        if not np.isnan(sorted_adj[i]) and not np.isnan(sorted_adj[i-1]):
            sorted_adj[i] = max(sorted_adj[i], sorted_adj[i-1])
    adjusted[order] = sorted_adj
    return adjusted.tolist()

def visualize(df,
                 outpath="./Data/Fig/hextend.pdf",
                 dpi=200,
                 figsize=(12,6),
                 logy_violin=True,
                 adjacent_pairwise=False,
                 alpha_points=0.7):
    """
    Create a two-panel figure:
      A) Violin + box + jitter + medians + 95th percentile markers
      B) Mean +/- 95% CI with sample sizes and adjacent-pair significance stars (Holm corrected)

    Returns: dict with figure, axes, and stats summary
    """
    # --- validate & prepare ---
    if 'time_s' not in df.columns:
        raise ValueError("DataFrame must contain a 'time_s' column (seconds).")
    df = df.copy()
    df['time_ms'] = df['time_s'].astype(float) * 1000.0
    df = df[df['hchange'].astype(int) >= 2].copy()
    if df.empty:
        raise ValueError("No rows with hchange >= 2.")
    df['hchange'] = df['hchange'].astype(int)

    # order groups
    hvals = sorted(df['hchange'].unique())
    groups = [df.loc[df['hchange'] == h, 'time_ms'].values for h in hvals]
    counts = [len(g) for g in groups]

    # Kruskal-Wallis overall
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) > 1:
        kw_stat, kw_p = stats.kruskal(*valid_groups)
    else:
        kw_stat, kw_p = np.nan, np.nan

    # --- build figure ---
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios':[1.1,1]})

    # PANEL A: Violin + box + jitter + medians + 95th
    parts = axA.violinplot(groups, showmeans=False, showmedians=False, widths=0.8)
    # style violin
    cmap = plt.get_cmap('tab10')
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cmap(i % 10))
        pc.set_edgecolor('black')
        pc.set_alpha(0.4)
    # small boxplots
    axA.boxplot(groups, positions=np.arange(1, len(hvals)+1),
                widths=0.12, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor='white', color='black'),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    # jittered points
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups, start=1):
        if g.size == 0:
            continue
        x_j = rng.normal(loc=i, scale=0.08, size=len(g))
        axA.scatter(x_j, g, s=18, alpha=alpha_points, color=cmap((i-1) % 10), edgecolors='k', linewidths=0.2)
        # median bar
        med = np.median(g)
        axA.hlines(med, i-0.22, i+0.22, colors='k', linewidth=2)
        # 95th percentile marker (red triangle down)
        p95 = np.quantile(g, 0.95)
        axA.plot(i, p95, marker='v', color='red', markersize=7)

    axA.set_xticks(range(1, len(hvals)+1))
    axA.set_xticklabels(hvals)
    axA.set_xlabel('Hcount change')
    axA.set_ylabel('Time (ms)')
    axA.set_title('A')
    if logy_violin:
        axA.set_yscale('log')
    axA.grid(True, linestyle=':', linewidth=0.5, which='both')

    # annotate sample sizes below ticks
    ymin, ymax = axA.get_ylim()
    for i, n in enumerate(counts, start=1):
        axA.text(i, ymin * (1.2 if logy_violin else 0.95), f'n={n}', ha='center', fontsize=9, color='gray')

    # PANEL B: Mean +/- 95% CI
    means = []
    cis = []
    for g in groups:
        if len(g) == 0:
            means.append(np.nan); cis.append(np.nan)
            continue
        m = np.mean(g); means.append(m)
        if len(g) > 1:
            sem = stats.sem(g)
            tcrit = stats.t.ppf(0.975, df=len(g)-1)
            cis.append(sem * tcrit)
        else:
            cis.append(np.nan)
    x = np.arange(len(hvals))
    axB.errorbar(x, means, yerr=cis, fmt='o-', capsize=5, linewidth=2, markersize=7, color='C0')
    axB.set_xticks(x); axB.set_xticklabels(hvals)
    axB.set_xlabel('Hcount change')
    axB.set_ylabel('Mean time (ms)')
    axB.set_title(f'B')
    axB.grid(True, linestyle=':', linewidth=0.5, axis='y')

    # annotate counts below each mean
    yref = np.nanmax(np.array(means) + np.nan_to_num(np.array(cis), nan=0.0))
    for xi, n in zip(x, counts):
        axB.text(xi, (means[xi] if not isnan(means[xi]) else 0) - 0.08*yref, f'n={n}', ha='center', fontsize=9, color='gray')

    # adjacent pairwise tests (Mann-Whitney U) + Holm correction
    sig_annotations = []
    if adjacent_pairwise and len(hvals) > 1:
        pvals = []
        pairs = []
        for i in range(len(groups)-1):
            a, b = groups[i], groups[i+1]
            pairs.append((i, i+1))
            if len(a) < 2 or len(b) < 2:
                pvals.append(np.nan)
            else:
                try:
                    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                except Exception:
                    p = np.nan
                pvals.append(p)
        adj_p = holm_adjust(pvals)
        # annotate significant ones
        for (i,j), p_raw, p_adj in zip(pairs, pvals, adj_p):
            if not isnan(p_adj) and p_adj < 0.05:
                # draw bar and star between xi and xj
                x1, x2 = i, j
                y_max = np.nanmax([ (means[k] + (cis[k] if not isnan(cis[k]) else 0)) for k in range(len(means)) ])
                height = y_max * 0.12  # spacing factor
                # compute vertical offset so multiple bars don't overlap much (use pair index)
                # here we stack by (i) to avoid overlap (simple heuristic)
                offset = 0.06 * y_max * (1 + (i % 3))
                y = y_max + height + offset
                axB.plot([x1, x1, x2, x2], [y-height*0.15, y, y, y-height*0.15], lw=1.2, color='k')
                # star text
                stars = '★'  # use one star for p<0.05
                if p_adj < 0.01:
                    stars = '★★'
                if p_adj < 0.001:
                    stars = '★★★'
                axB.text((x1+x2)/2, y + 0.02*y_max, stars, ha='center', va='bottom', fontsize=12, color='k')
                sig_annotations.append(((hvals[i], hvals[j]), p_raw, p_adj))

    # polish, save, return
    # fig.suptitle("HExtend — Panels A & B (hchange ≥ 2)", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()

    # summary to return
    perc_table = pd.DataFrame({h: np.quantile(groups[i], [0.5, 0.95]) if len(groups[i])>0 else [np.nan, np.nan]
                               for i,h in enumerate(hvals)},
                              index=['50% (median) ms', '95% ms']).T
    summary = {
        'figure': fig,
        'axA': axA,
        'axB': axB,
        'hvals': hvals,
        'counts': counts,
        'means_ms': means,
        'cis_ms': cis,
        'kruskal': (kw_stat, kw_p),
        'adjacent_pairwise_significant': sig_annotations,
        'percentiles_table': perc_table
    }
    # print a short summary
    print("\nKruskal-Wallis: H={:.3g}, p={:.3g}".format(kw_stat, kw_p))
    if sig_annotations:
        print("\nSignificant adjacent pairs (raw p, Holm-adj p):")
        for (hpair, p, adj) in sig_annotations:
            print(f"  {hpair[0]} vs {hpair[1]} : p={p:.3g}, adj={adj:.3g}")
    else:
        print("\nNo significant adjacent-pair differences found (adj Holm p<0.05).")
    print(f"\nSaved figure: {outpath}")
    return summary


def sanity_check_paths(inp: str, outp: str, logp: str) -> None:
    """Quick checks: input exists, output directory writable, log directory writable."""
    if not os.path.exists(inp):
        raise FileNotFoundError(f"Input not found: {inp}")
    out_dir = os.path.dirname(outp) or "."
    log_dir = os.path.dirname(logp) or "."
    if not os.path.isdir(out_dir):
        raise NotADirectoryError(f"Output directory does not exist: {out_dir}")
    if not os.access(out_dir, os.W_OK):
        raise PermissionError(f"Output directory not writable: {out_dir}")
    if not os.path.isdir(log_dir):
        raise NotADirectoryError(f"Log directory does not exist: {log_dir}")
    if not os.access(log_dir, os.W_OK):
        raise PermissionError(f"Log directory not writable: {log_dir}")


# -------------------------
# log parser (master log -> DataFrame)
# -------------------------
def parse_hextend_log_to_df(log_text_or_path):
    """
    Parse master log (string or path) and return DataFrame with columns:
      - R-id (e.g. 'R-58132')
      - hchange (int or NaN)
      - time_s (float seconds)
    Heuristics:
      - Looks for '=== START HExtend for R-XXXX' as block starts
      - Looks for 'Hcount change: N' in the block
      - Looks for 'Finished HExtend for R-XXXX in X.XXX s' for time
    """
    if os.path.exists(log_text_or_path):
        with open(log_text_or_path, "r", encoding="utf8") as fh:
            text = fh.read()
    else:
        text = str(log_text_or_path)

    finish_re = re.compile(r'Finished HExtend for (R-\d+) in ([0-9.]+) s')
    hcount_re = re.compile(r'Hcount change:\s*([0-9]+)')
    start_marker = '=== START HExtend for'

    rows = []
    for m in finish_re.finditer(text):
        rid = m.group(1)
        time_s = float(m.group(2))
        block_start = text.rfind(start_marker, 0, m.start())
        if block_start == -1:
            block_start = max(0, m.start() - 1000)
        block_text = text[block_start:m.end()]
        hc_matches = hcount_re.findall(block_text)
        hchange = int(hc_matches[-1]) if hc_matches else pd.NA
        rows.append({"R-id": rid, "hchange": hchange, "time_s": time_s})
    df = pd.DataFrame(rows)
    # ensure types
    if not df.empty:
        df['hchange'] = df['hchange'].astype('Int64')
        df['time_s'] = df['time_s'].astype(float)
    return df


# -------------------------
# runner
# -------------------------
def run_hextend_on_dataset(
    data_in: str,
    data_out: str,
    master_log: str,
    start_idx: Optional[int] = None,
    stop_idx: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    setup_root_logger(master_log)
    logger = logging.getLogger("hextend_loop")

    logger.info("Loading input data from: %s", data_in)
    data = load_from_pickle(data_in)

    if start_idx is None:
        start_idx = 0
    if stop_idx is None:
        stop_idx = len(data)

    logger.info("Processing indices [%d, %d) (total entries: %d)", start_idx, stop_idx, len(data))

    root = logging.getLogger()
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    results = []
    try:
        for idx in range(start_idx, min(stop_idx, len(data))):
            entry = data[idx]
            item_id = entry.get("R-id", f"idx-{idx}")
            logger.info("=== START HExtend for %s (data index %d) ===", item_id, idx)

            # per-entry in-memory capture of logging
            sio = io.StringIO()
            per_entry_handler = logging.StreamHandler(sio)
            per_entry_handler.setFormatter(formatter)
            root.addHandler(per_entry_handler)

            t0 = time.perf_counter()
            try:
                if dry_run:
                    # create lightweight synthetic result and short sleep for timing realism
                    out = {"_dryrun": True}
                    elapsed = 0.001
                    logger.info("DRY-RUN: skipping actual HExtend().fit for %s", item_id)
                else:
                    result = HExtend().fit([entry], "ITS", "RC", n_jobs=1, verbose=2)
                    out = result[0] if isinstance(result, (list, tuple)) and len(result) == 1 else result
                    elapsed = time.perf_counter() - t0
                    logger.info("Finished HExtend for %s in %.3f s", item_id, elapsed)

                # attach results + timing + per-entry log text
                data[idx]["HEXTEND_result"] = out
                data[idx]["HEXTEND_time_s"] = elapsed
                data[idx]["HEXTEND_log"] = sio.getvalue()
                results.append({"idx": idx, "R-id": item_id, "result": out, "time_s": elapsed})

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.exception("HExtend failed for %s (index %d) after %.3f s", item_id, idx, elapsed)
                data[idx]["HEXTEND_result"] = None
                data[idx]["HEXTEND_time_s"] = elapsed
                data[idx]["HEXTEND_error"] = str(exc)
                data[idx]["HEXTEND_log"] = sio.getvalue()
                results.append({"idx": idx, "R-id": item_id, "result": None, "time_s": elapsed, "error": str(exc)})

            finally:
                try:
                    root.removeHandler(per_entry_handler)
                except Exception:
                    logger.debug("Failed to remove per-entry handler for %s", item_id)
                sio.close()
                for h in root.handlers:
                    try:
                        h.flush()
                    except Exception:
                        pass
                try:
                    save_to_pickle(data, data_out)
                except Exception as exc:
                    logger.exception("Failed to save progress to %s after index %d: %s", data_out, idx, exc)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user at index %d. Saving progress...", idx)
        save_to_pickle(data, data_out)
        raise

    total_time = sum(r.get("time_s", 0) for r in results)
    logger.info("Processed %d entries; total wall time = %.3f s", len(results), total_time)
    try:
        save_to_pickle(data, data_out)
        logger.info("Final results saved to %s", data_out)
    except Exception as exc:
        logger.exception("Failed to write final output to %s: %s", data_out, exc)


# -------------------------
# CLI & postprocess glue
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HExtend on a pickled dataset.")
    p.add_argument("--in", dest="data_in", default=DEFAULT_IN, help=f"Input pickle path (default: {DEFAULT_IN})")
    p.add_argument("--out", dest="data_out", default=DEFAULT_OUT, help=f"Output pickle path (default: {DEFAULT_OUT})")
    p.add_argument("--log", dest="master_log", default=DEFAULT_LOG, help=f"Master log path (default: {DEFAULT_LOG})")
    p.add_argument("--start", dest="start_idx", default=None, type=int, help="Optional start index (inclusive)")
    p.add_argument("--stop", dest="stop_idx", default=None, type=int, help="Optional stop index (exclusive)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="Simulate run; do not call HExtend().fit")
    p.add_argument("--postprocess-log", dest="postprocess_log", action="store_true",
                   help="After run, parse master log into CSV (R-id,hchange,time_s) -> saved next to log as .summary.csv")
    p.add_argument("--plot", dest="plot", action="store_true",
                   help="If postprocessable, attempt to call visualize() on the parsed DataFrame. Requires that function in scope.")
    return p.parse_args()


def main():
    args = parse_args()

    # quick path checks (fail fast)
    try:
        sanity_check_paths(args.data_in, args.data_out, args.master_log)
    except Exception as exc:
        print(f"Path sanity check failed: {exc}", file=sys.stderr)
        sys.exit(2)

    run_hextend_on_dataset(
        data_in=args.data_in,
        data_out=args.data_out,
        master_log=args.master_log,
        start_idx=args.start_idx,
        stop_idx=args.stop_idx,
        dry_run=args.dry_run,
    )

    if args.postprocess_log:
        print("Parsing master log to CSV...")
        df = parse_hextend_log_to_df(args.master_log)
        csv_out = os.path.splitext(args.master_log)[0] + ".summary.csv"
        # df.to_csv(csv_out, index=False)
        print(f"Saved parsed summary -> {csv_out}")
        if args.plot:
            # try to produce the A+B figure; requires visualize to be defined/imported
            if 'visualize' in globals():
                # note: visualize expects DataFrame with time_s in seconds? our df has time_s in seconds.
                # earlier visualize converted to ms internally; but user's version may expect columns
                try:
                    # call with the parsed df
                    visualize(df, outpath="./Data/Fig/hextend.pdf")
                    print("Saved A+B figure.")
                except Exception as exc:
                    print(f"Plot generation failed: {exc}", file=sys.stderr)
            else:
                print("visualize() not found in scope. Import or define it before using --plot.", file=sys.stderr)

if __name__ == "__main__":
    main()
