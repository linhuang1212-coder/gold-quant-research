#!/usr/bin/env python3
"""R230 M30 Only: Run Part A (M30) with full Phase 3-10 using multiprocessing.

The initial parallel resume skipped M30 Phase 3+ due to a resume logic bug.
This script runs only M30 through the full pipeline with parallelism.
"""
from __future__ import annotations
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from itertools import combinations, product
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.m30_engine import M30BacktestEngine, load_m30_with_indicators
from backtest.engine import TradeRecord
from experiments.run_r230_mega_overnight import M30_STRATEGIES
from experiments.run_r230_parallel_resume import (
    OUTPUT_DIR, SPREAD, N_BOOTSTRAP, N_WORKERS,
    ERA_SEGMENTS, WF_CUTOFFS, SL_GRID, TP_GRID, TRAIL_GRID,
    print_flush, save, save_progress, calc_stats,
    monte_carlo_bootstrap, drawdown_analysis,
    _init_worker, _run_single_combo, _run_wf_window, _run_era,
    run_parallel_pipeline,
)


def main():
    t_global = time.time()
    print_flush('=' * 80)
    print_flush(f'R230 M30 ONLY ({N_WORKERS} workers on {cpu_count()} cores)')
    print_flush(f'Started: {pd.Timestamp.now()}')
    print_flush('=' * 80)

    # Load existing phase2 results
    m30_phase2_file = OUTPUT_DIR / 'm30_phase2_kfold.json'
    m30_phase2_data = None
    if m30_phase2_file.exists():
        m30_phase2_data = json.loads(m30_phase2_file.read_text())
        kf_check = [s for s in m30_phase2_data if m30_phase2_data[s].get('kfold_6', {}).get('verdict') == 'PASS']
        print_flush(f'  Loaded M30 Phase 2 from disk. K-Fold passers: {kf_check}')

    # Run M30 pipeline
    try:
        print_flush(f'\n{"#"*80}\n# PART A: M30 Strategy Universe (PARALLEL)\n{"#"*80}')
        m30_df = load_m30_with_indicators()
        m30_results = run_parallel_pipeline(
            tf_name='m30',
            df=m30_df,
            strategies=M30_STRATEGIES,
            engine_cls=M30BacktestEngine,
            default_params={
                'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0,
                'trailing_activate_atr': 0.3, 'trailing_distance_atr': 0.08,
                'max_hold': 48, 'cooldown_bars': 4, 'spread_cost': SPREAD,
            },
            sl_grid=SL_GRID,
            tp_grid=TP_GRID,
            trail_grid=TRAIL_GRID,
            max_hold_grid=[24, 36, 48, 72, 96],
            part_label='PART_A_M30',
            skip_phase1_2=(m30_phase2_data is not None),
            phase2_result=m30_phase2_data,
        )
        save_progress('PART_A_M30_PARALLEL', 'COMPLETE')

        # Print summary
        print_flush(f'\n{"#"*80}\n# M30 RESULTS SUMMARY\n{"#"*80}')
        p10 = m30_results.get('phase10', {})
        for sname, info in p10.items():
            v = info.get('final_verdict', 'REJECT')
            marker = '***' if v == 'STRONG_PASS' else ('**' if v == 'CONDITIONAL_PASS' else '')
            print_flush(f'  m30/{sname:<20} Sharpe={info.get("best_sharpe", 0):.3f} -> {v} {marker}')

    except Exception as e:
        print_flush(f'\n!!! M30 ERROR: {e}')
        traceback.print_exc()
        save_progress('PART_A_M30_PARALLEL', 'ERROR', str(e))

    elapsed_total = time.time() - t_global
    print_flush(f'\n  Total M30 runtime: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)')
    print_flush(f'  Finished: {pd.Timestamp.now()}')
    print_flush('=' * 80)


if __name__ == '__main__':
    main()
