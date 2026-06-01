#!/usr/bin/env bash
# Overnight campaign orchestrator for the quantum-ids paper.
#
# Runs the whole experiment campaign SERIALLY (no GPU/CPU contention), in
# paper-priority order (cheapest + most critical first) so that even partial
# completion by morning yields a usable result set. IDEMPOTENT: each phase writes
# a .done marker; on relaunch (e.g. after a WSL sleep/restart killed it) completed
# phases are skipped and the interrupted phase is re-run cleanly (its results are
# cleared by name-prefix first to avoid duplicate-seed inflation).
#
# Launch (from Windows, detached so it survives this Claude turn):
#   wsl -d Ubuntu-22.04 -- bash -lc "nohup bash ~/overnight.sh >/tmp/overnight.out 2>&1 &"
# Monitor: tail results/overnight_progress.log
set +e  # never abort the whole night on a single failure

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml
export PYTHONNOUSERSITE=1
export QIDS_N_JOBS=4            # cap joblib so classical tuning can't thrash the box
cd "/mnt/c/Research work 2/quantum-ids/src"

RES="/mnt/c/Research work 2/quantum-ids/results"
FIG="/mnt/c/Research work 2/quantum-ids/figures"
MARK="$RES/.overnight"; mkdir -p "$MARK" "$FIG"
PROG="$RES/overnight_progress.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROG"; }

# run a phase once: phase <id> <result-name-prefix-to-clear> <command...>
phase() {
  local id="$1"; local prefix="$2"; shift 2
  if [ -f "$MARK/$id.done" ]; then log "SKIP $id (already done)"; return 0; fi
  log "START $id"
  # NOTE: do NOT clear prior results here — runner.py now skips per-run by slug,
  # so an interrupted phase (e.g. WSL restart) resumes at the next unfinished run
  # instead of recomputing the whole phase. (prefix kept for reference only.)
  "$@" >>"$RES/logs/overnight_$id.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$MARK/$id.done"; log "DONE  $id (rc=0)"; else log "FAIL  $id (rc=$rc) — see logs/overnight_$id.log, continuing"; fi
  # refresh aggregate + figures after every phase so partial results are usable
  python aggregate.py --results-dir "$RES" >>"$RES/logs/overnight_aggregate.log" 2>&1
  python figures.py --results-dir "$RES" --out "$FIG" >>"$RES/logs/overnight_figures.log" 2>&1
  return $rc
}

log "================ OVERNIGHT CAMPAIGN START ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$PROG"

# 1. Classical baselines on NSL-KDD, BOTH views, 5 seeds — the headline comparison
#    + the CV-vs-test distribution-shift gap (core honest-broker result). Cheapest
#    high-value result, so first.
phase classical_nslkdd "" \
  python run_baselines.py --dataset nslkdd --view both \
    --seeds 42 43 44 45 46 --n-iter 20

# 2. Quantum ablation on NSL-KDD (9 variants x 3 seeds) — the core novelty table.
phase quantum_phase1 "ovn1_" \
  python runner.py --config configs/overnight_q_phase1.yaml --results-dir "$RES"

# 3. Cross-dataset quantum (best 2 encodings + projected QSVM x 4 datasets x 3 seeds).
phase quantum_phase2 "ovn2_" \
  python runner.py --config configs/overnight_q_phase2.yaml --results-dir "$RES"

# 4. Classical baselines on the other 3 datasets (3 seeds to bound cost).
phase classical_unsw "" \
  python run_baselines.py --dataset unsw   --view both --seeds 42 43 44 --n-iter 15
phase classical_cicids "" \
  python run_baselines.py --dataset cicids --view both --seeds 42 43 44 --n-iter 15
phase classical_toniot "" \
  python run_baselines.py --dataset toniot --view both --seeds 42 43 44 --n-iter 15

# 5. NISQ noise-robustness sweep.
phase noise_sweep "ovnN_" \
  python runner.py --config configs/overnight_noise.yaml --results-dir "$RES"
# noise degradation figures (best-effort; helper may need a results dir flag)
python noise_plots.py --results-dir "$RES" >>"$RES/logs/overnight_noiseplots.log" 2>&1

# 6. Quantum-attribution audit (the Bellante rebuttal) — uses baselines + quantum
#    results that now exist on disk.
phase attribution "" \
  python attribution.py --config configs/attribution_audit.yaml --results-dir "$RES"

# 7. Final aggregate + compare (quantum vs best classical significance) + figures.
phase finalize "" bash -c '
  python aggregate.py --results-dir "'"$RES"'" --metric f1
  python compare.py   --results-dir "'"$RES"'" >>"'"$RES"'/logs/overnight_compare.log" 2>&1
  python figures.py   --results-dir "'"$RES"'" --out "'"$FIG"'"
'

log "================ OVERNIGHT CAMPAIGN END ================"
log "results JSONs: $(ls "$RES"/*.json 2>/dev/null | wc -l) | figures: $(ls "$FIG"/*.png 2>/dev/null | wc -l)"
# Only declare ALL_COMPLETE when EVERY phase actually succeeded (wrote its .done).
# Otherwise leave it unset so the self-healing launcher relaunches and retries the
# failed phases (e.g. UNSW/CICIDS after the OOM fixes) on its next cycle.
_all=1
for _p in classical_nslkdd quantum_phase1 quantum_phase2 classical_unsw classical_cicids classical_toniot noise_sweep attribution finalize; do
  if [ ! -f "$MARK/$_p.done" ]; then log "INCOMPLETE: $_p not done — will retry on next launcher cycle"; _all=0; fi
done
if [ "$_all" = "1" ]; then touch "$MARK/ALL_COMPLETE"; log "ALL phases complete."; else log "Some phases incomplete; ALL_COMPLETE NOT set."; fi
