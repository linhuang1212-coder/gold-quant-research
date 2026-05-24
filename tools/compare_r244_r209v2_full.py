"""Consolidated comparison: R244 v1 (no cap) | R244 v2 (cap) | R209v2 (engine).

Generates results/r244_r209v2_comparison.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Load R244 v1 (no cap) — from earlier run (results/r244_missing_strategies)
# Load R244 v2 (with cap) — from new run
# Load R209v2 — full engine results
# ============================================================

r244_v1_path = ROOT / 'results/r244_missing_strategies/results.json'
r244_v2_path = ROOT / 'results/r244_missing_strategies_v2_with_cap/results.json'
r209v2_path = ROOT / 'results/r209v2_non_keltner_audit/R209v2_summary.json'


def safe_load(p):
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


r244_v1 = safe_load(r244_v1_path)
r244_v2 = safe_load(r244_v2_path)
r209v2 = safe_load(r209v2_path)

out = []
out.append('# R244/R209v2 Consolidated Comparison')
out.append('')
out.append('_Comparing the three valid lenses for non-Keltner live strategies:_')
out.append('  - **R244 v1**: Standalone backtest, NO maxloss cap (pure signal+SL/TP/Trail)')
out.append('  - **R244 v2**: Standalone backtest, WITH cap_atr maxloss mechanism')
out.append('  - **R209v2**: Full BacktestEngine with all filters (Choppy Gate, ATR Pctl, ADX, Regime)')
out.append('')

# ============================================================
# Section 1 — Full-history Sharpe / PnL comparison
# ============================================================
out.append('## Full-history Performance (2015 -> 2026)')
out.append('')
out.append('| Strategy | R244 v1 (no cap) | R244 v2 (cap) | R209v2 Phase1 (all) | R209v2 Phase2 (isolated) |')
out.append('|---|---|---|---|---|')

strats = ['tsmom', 'sess_bo', 'dual_thrust']
phase1 = r209v2.get('phase1_all_together', {}) if r209v2 else {}
phase2 = r209v2.get('phase2_isolated', {}) if r209v2 else {}

for s in strats:
    s_up = s.upper()
    v1 = r244_v1.get(s_up, {}).get('full', {}) if r244_v1 else {}
    v2 = r244_v2.get(s_up, {}).get('full', {}) if r244_v2 else {}
    p1 = phase1.get(s, {})
    p2 = phase2.get(s, {}).get('full', {})

    def fmt(d, has_engine=False):
        if not d:
            return '—'
        if has_engine:
            return f"S={d.get('sharpe', 0):.2f} N={d.get('n', 0)} PnL=${d.get('pnl', 0):.0f}"
        return f"S={d.get('sharpe', 0):.2f} N={d.get('n', 0)} PnL=${d.get('pnl', 0):.0f}"

    out.append(f"| {s.upper()} | {fmt(v1)} | {fmt(v2)} | {fmt(p1, True)} | {fmt(p2, True)} |")

out.append('')

# ============================================================
# Section 2 — Recent-7mo comparison
# ============================================================
out.append('## Recent-7mo / Live Period Performance')
out.append('')
out.append('_R244 uses last 7 months (Oct-Apr). R209v2 uses live period (2026-03-25 -> 2026-05-13)._')
out.append('')
out.append('| Strategy | R244 v1 Recent | R244 v2 Recent | R209v2 Live (Phase1) | R209v2 Live (Phase2) |')
out.append('|---|---|---|---|---|')

for s in strats:
    s_up = s.upper()
    v1 = r244_v1.get(s_up, {}).get('recent_7mo', {}) if r244_v1 else {}
    v2 = r244_v2.get(s_up, {}).get('recent_7mo', {}) if r244_v2 else {}
    p1 = phase1.get('_live_period', {}).get(s, {})
    p2 = phase2.get(s, {}).get('live_period', {})

    def fmt(d):
        if not d:
            return '—'
        return f"S={d.get('sharpe', 0):.2f} N={d.get('n', 0)} PnL=${d.get('pnl', 0):.0f}"

    out.append(f"| {s.upper()} | {fmt(v1)} | {fmt(v2)} | {fmt(p1)} | {fmt(p2)} |")

out.append('')

# ============================================================
# Section 3 — Era breakdown
# ============================================================
out.append('## Era Breakdown (R209v2 Phase2 isolated)')
out.append('')
eras = ['Pre-COVID (2015-2019)', 'COVID+Recovery (2020-2021)', 'Tightening (2022-2023)', 'Recent (2024-2026)']
out.append('| Strategy | ' + ' | '.join(eras) + ' |')
out.append('|---|' + '---|' * len(eras))
# TSMOM era data is in R244 v2's era field
for s in strats:
    s_up = s.upper()
    if s == 'tsmom':
        # TSMOM only in R244 v2
        p2 = r244_v2.get(s_up, {}).get('era', {})
    else:
        p2 = phase2.get(s, {}).get('eras', {})
    row = [s.upper()]
    for era in eras:
        d = p2.get(era, {})
        row.append(f"S={d.get('sharpe', 0):.2f} N={d.get('n', 0)} PnL=${d.get('pnl', 0):.0f}" if d else '—')
    out.append('| ' + ' | '.join(row) + ' |')
out.append('')

# Add PSAR + Chandelier (only in R209v2)
out.append('### Additional strategies (R209v2 Phase2 only)')
out.append('')
out.append('| Strategy | ' + ' | '.join(eras) + ' |')
out.append('|---|' + '---|' * len(eras))
for s in ['psar', 'chandelier']:
    p2 = phase2.get(s, {}).get('eras', {})
    row = [s.upper()]
    for era in eras:
        d = p2.get(era, {})
        row.append(f"S={d.get('sharpe', 0):.2f} N={d.get('n', 0)} PnL=${d.get('pnl', 0):.0f}" if d else '—')
    out.append('| ' + ' | '.join(row) + ' |')
out.append('')

# ============================================================
# Section 4 — Sanity Gate (live vs backtest ratio)
# ============================================================
out.append('## R209v2 Phase3: Sanity Gate (Live vs BT in live period)')
out.append('')
out.append('| Strategy | Live N | BT N | Ratio | Live PnL | BT PnL | Sign Match? |')
out.append('|---|---|---|---|---|---|---|')
sanity = r209v2.get('phase3_sanity', {}) if r209v2 else {}
for s in ['keltner', 'm15_rsi', 'dual_thrust', 'sess_bo', 'psar', 'chandelier']:
    d = sanity.get(s, {})
    if not d:
        continue
    sign_match = '✓' if (d['live_pnl'] * d['bt_pnl'] >= 0) else '⚠ MISMATCH'
    out.append(f"| {s} | {d['live_n']} | {d['bt_n']} | {d['ratio']:.1f}x | ${d['live_pnl']:.0f} | ${d['bt_pnl']:.0f} | {sign_match} |")
out.append('')

# ============================================================
# Section 5 — Cap mechanism analysis
# ============================================================
out.append('## Cap Mechanism Effect (R244 v1 vs v2)')
out.append('')
out.append('| Strategy | v1 PnL | v2 PnL | Delta (v2-v1) | Cap-fired trades | Cap stops earlier than SL? |')
out.append('|---|---|---|---|---|---|')

# We need to recompute cap trade counts from trades.json
cap_stats = {}
import collections
for s in strats:
    p = ROOT / f'results/r244_missing_strategies_v2_with_cap/{s}_trades.json'
    if not p.exists():
        continue
    with open(p) as f:
        trades = json.load(f)
    cap_n = sum(1 for t in trades if t['reason'] == 'MaxLossCap')
    cap_stats[s] = {'n': cap_n, 'total': len(trades), 'pct': cap_n / len(trades) * 100 if trades else 0}

# cap stops earlier? → only when cap_atr_mult × ATR < sl_atr × ATR × LOT × PV / (LOT × PV) = sl_atr × ATR (in price)
# i.e. when cap_atr_mult < sl_atr (cap in price terms < SL in price terms)
cap_params = {
    'tsmom': (6.5, 3.5),       # cap=6.5×ATR loss = 6.5×ATR ÷ (LOT×PV) price drop;  SL=3.5×ATR price drop
    'sess_bo': (5.0, 4.5),
    'dual_thrust': (5.0, 6.0),
}

for s in strats:
    s_up = s.upper()
    v1_pnl = r244_v1.get(s_up, {}).get('full', {}).get('pnl', 0) if r244_v1 else 0
    v2_pnl = r244_v2.get(s_up, {}).get('full', {}).get('pnl', 0) if r244_v2 else 0
    delta = v2_pnl - v1_pnl
    cs = cap_stats.get(s, {'n': 0, 'total': 1, 'pct': 0})
    cap_a, sl_a = cap_params.get(s, (0, 0))
    earlier = 'YES (cap tighter)' if cap_a < sl_a else 'NO (cap looser, only catches gaps)'
    out.append(f"| {s.upper()} | ${v1_pnl:.0f} | ${v2_pnl:.0f} | ${delta:+.0f} | {cs['n']} ({cs['pct']:.1f}%) | {earlier} |")

out.append('')

# ============================================================
# Section 6 — Verdict
# ============================================================
out.append('## Verdict per Strategy')
out.append('')
out.append('### TSMOM')
out.append('- **Without cap (v1)**: Sharpe -0.01 full / +2.11 recent — break-even with strong recent regime.')
out.append('- **With cap (v2)**: Sharpe -0.13 / +2.04 — cap fires 4.2% of trades but slightly degrades because cap (6.5×ATR) is LOOSER than SL (3.5×ATR); only catches gap-downs where exit price is worse than SL would have been.')
out.append('- **Engine (R209v2)**: Not in R209v2 (TSMOM is M30-only).')
out.append('- **Verdict**: Cap is a sanity net for gap-downs but does not improve risk-adjusted return. Strategy depends on momentum regime.')
out.append('')
out.append('### SESS_BO')
out.append('- **Standalone v1**: Sharpe 1.38 / 689 trades — strong.')
out.append('- **Standalone v2 (cap)**: Sharpe 1.27 / 689 trades — slight degradation, cap fires 2.0%.')
out.append('- **Engine all-together**: Sharpe 0.60 / 521 trades — slot contention with Keltner reduces alpha.')
out.append('- **Engine isolated**: Sharpe 0.98 / 243 trades — engine filters cut trade count 65% but preserve Sharpe ~1.')
out.append('- **Live period (1 trade)**: BT -$13 vs live +$35 — too few samples to conclude.')
out.append('- **Verdict**: Robust across all three lenses, recent era Sharpe 1.91 (engine isolated) confirms continued edge.')
out.append('')
out.append('### Dual Thrust — KEY FINDING')
out.append('- **Standalone v1**: Sharpe -0.91 / -$10186 — HISTORICALLY LOSING.')
out.append('- **Standalone v2 (cap)**: Sharpe -0.64 / -$7070 — cap fires 9.4% and saves $3116, but still negative.')
out.append('- **Engine all-together**: Sharpe 1.21 / +$7630 — engine filters CUT trade count 58% (5314 -> 2222) and FLIP sign.')
out.append('- **Engine isolated**: Sharpe 1.21 / +$4143 — confirms engine filters are the alpha source.')
out.append('- **Live period (live vs BT)**: BT +$564 / 18 trades, live +$35 / 12 trades — sign match.')
out.append('- **Verdict**: ✓ The live system uses the full filter stack. The naive (no-filter) Dual Thrust loses money; the live (filtered) Dual Thrust makes money. **Live deployment is correct.**')
out.append('')
out.append('### PSAR (R209v2 only)')
out.append('- **Engine all-together**: Sharpe 0.87 / +$2553 — positive overall.')
out.append('- **Engine isolated**: Sharpe -0.13 / -$263 — marginally negative when alone!')
out.append('- **Tightening era**: Sharpe -3.32 / -$990 — RED FLAG for 2022-2023 regime.')
out.append('- **Live period sanity check**: BT +$614 vs live -$85 → ⚠ SIGN MISMATCH (only 3 live trades vs 13 BT trades; live is producing 4.3x fewer signals).')
out.append('- **Verdict**: ⚠ Live deployment under-firing AND sign mismatch — investigate why live PSAR signals are not matching backtest. Tightening era loss is a structural concern.')
out.append('')
out.append('### Chandelier (R209v2 only)')
out.append('- **Engine all-together**: Sharpe 0.22 / +$760 — barely positive.')
out.append('- **Engine isolated**: Sharpe 0.05 / +$170 — essentially zero.')
out.append('- **Recent era (2024-2026)**: Sharpe -0.15 / -$191 — RED FLAG, edge has decayed.')
out.append('- **Live period**: BT -$297 / 15 trades, live -$36 / 4 trades — both negative, but trade count ratio 3.8x suggests live is firing less.')
out.append('- **Verdict**: ⚠ Chandelier shows no edge in isolation. Currently profitable only as portfolio member through engine slot contention. Consider disabling.')
out.append('')

# ============================================================
# Section 7 — Action Items
# ============================================================
out.append('## Recommended Actions')
out.append('')
out.append('1. **Dual Thrust**: KEEP deployed — confirmed positive Sharpe through full engine path (1.21). The standalone loss is from filter-absent baseline, not the actual live system.')
out.append('2. **SESS_BO**: KEEP deployed — robust across all eras and lenses.')
out.append('3. **TSMOM**: KEEP under regime-conditional logic — recent Sharpe 2+ but full-history -0.01 means it depends on continued momentum regime. The cap is OK as gap-protection but does not need re-tuning.')
out.append('4. **PSAR**: INVESTIGATE — Live live-period signal generation is producing 4.3x fewer trades than backtest AND wrong sign. Audit the live PSAR implementation for parameter drift.')
out.append('5. **Chandelier**: CONSIDER DISABLING — no isolated edge, negative recent-era. Currently only positive via slot contention with Keltner.')
out.append('6. **Keltner sanity flag**: Sharpe 7.98 / WR 91.8% triggers the result-sanity-gate (Sharpe>5, WR>85%). Re-validate via era-segmented Sharpe (should already be in r243).')
out.append('')

target = ROOT / 'results/r244_r209v2_comparison.md'
target.write_text('\n'.join(out), encoding='utf-8')
print(f'Wrote {target} ({len(out)} lines)')
