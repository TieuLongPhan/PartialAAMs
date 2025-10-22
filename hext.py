import io
import time
import logging
from synkit.IO import load_from_pickle, save_to_pickle
from synkit.Graph.Hyrogen.hextend import HExtend

# --- paths ---
DATA_PICKLE_IN = "./Data/hydrogen.pkl.gz"
DATA_PICKLE_OUT = "./Data/hydrogen_hextend_enriched.pkl.gz"
MASTER_LOG = "./Data/hextend_run.log"

# --- logging setup (attach to root logger so module logs are captured) ---
fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

root = logging.getLogger()
root.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates (safe when running interactively)
for h in list(root.handlers):
    root.removeHandler(h)

# master (file) handler -> writes in real time
fh = logging.FileHandler(MASTER_LOG, mode="a")
fh.setFormatter(formatter)
root.addHandler(fh)

# console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
root.addHandler(ch)

logger = logging.getLogger("hextend_loop")  # convenience logger, uses root handlers

# load data
data = load_from_pickle(DATA_PICKLE_IN)

start_idx = 0
stop_idx = len(data)


results = []

for idx in range(start_idx, stop_idx):
    entry = data[idx]
    item_id = entry.get("R-id", f"idx-{idx}")
    logger.info("=== START HExtend for %s (data index %d) ===", item_id, idx)

    # per-entry in-memory capture of logging
    sio = io.StringIO()
    per_entry_handler = logging.StreamHandler(sio)
    per_entry_handler.setFormatter(formatter)
    root.addHandler(per_entry_handler)   # attach so all module logs are captured

    t0 = time.perf_counter()
    try:
        result = HExtend().fit([entry], "ITS", "RC", n_jobs=1, verbose=2)

        # normalize return if a single-element list/tuple
        if isinstance(result, (list, tuple)) and len(result) == 1:
            out = result[0]
        else:
            out = result

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
        # record failure info and per-entry log
        data[idx]["HEXTEND_result"] = None
        data[idx]["HEXTEND_time_s"] = elapsed
        data[idx]["HEXTEND_error"] = str(exc)
        data[idx]["HEXTEND_log"] = sio.getvalue()
        results.append({"idx": idx, "R-id": item_id, "result": None, "time_s": elapsed, "error": str(exc)})

    finally:
        # ensure per-entry handler is removed and buffer closed
        root.removeHandler(per_entry_handler)
        sio.close()
        # flush master file handler to make sure log is persisted
        for h in root.handlers:
            try:
                h.flush()
            except Exception:
                pass
        # persist progress after each item
        save_to_pickle(data, DATA_PICKLE_OUT)

# small summary
total_time = sum(r["time_s"] for r in results)
logger.info("Processed %d entries; total wall time = %.3f s", len(results), total_time)
