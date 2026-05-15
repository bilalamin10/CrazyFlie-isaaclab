from tbparse import SummaryReader
from pathlib import Path

print(f"{'Run':30s}  {'Final reward':>14s}")
print("-" * 46)

for run_dir in sorted(Path("logs/rsl_rl/").glob("*_seed*/*/")):
    reader = SummaryReader(str(run_dir))
    df = reader.scalars
    if df.empty:
        continue
    rows = df[df.tag == "Train/mean_reward"]
    if rows.empty:
        continue
    final = rows.iloc[-1]
    print(f"{run_dir.parent.name:30s}  {final['value']:>14.1f}")
