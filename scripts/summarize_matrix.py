import pandas as pd

df = pd.read_csv("logs/eval_matrix_v2.csv")
shapes_order = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]

print("=" * 70)
print("Success rate — mean across seeds")
print("(rows = train shape, columns = eval shape)")
print("=" * 70)
mean = df.groupby(["train_shape", "eval_shape"])["success_rate"].mean().unstack()
mean = mean.reindex(index=shapes_order, columns=shapes_order)
print(mean.round(3).to_string())

print("\n" + "=" * 70)
print("Success rate — std across seeds")
print("=" * 70)
std = df.groupby(["train_shape", "eval_shape"])["success_rate"].std().unstack()
std = std.reindex(index=shapes_order, columns=shapes_order)
print(std.round(3).to_string())

print("\n" + "=" * 70)
print("Tracking error mean (m) — mean across seeds")
print("=" * 70)
te = df.groupby(["train_shape", "eval_shape"])["tracking_err_mean"].mean().unstack()
te = te.reindex(index=shapes_order, columns=shapes_order)
print(te.round(3).to_string())
