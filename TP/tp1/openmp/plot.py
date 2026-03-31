import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pi_results.csv")
df["Speedup"] = df["Serial"] / df["Parallel"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Runtime comparison
ax1.plot(df["N Threads"], df["Serial"], marker='o', label="Serial")
ax1.plot(df["N Threads"], df["Parallel"], marker='o', label="Parallel")
ax1.set_xlabel("Number of threads")
ax1.set_ylabel("Time (seconds)")
ax1.set_title("Runtime vs Number of threads")
ax1.set_xscale("log", base=2)
ax1.set_xticks(df["N Threads"])
ax1.set_xticklabels(df["N Threads"])
ax1.legend()
ax1.grid(True)

# Speedup
ax2.plot(df["N Threads"], df["Speedup"], marker='o', color='green', label="Actual speedup")
ax2.set_xlabel("Number of threads")
ax2.set_ylabel("Speedup")
ax2.set_title("Speedup vs Number of threads")
ax2.set_xscale("log", base=2)
ax2.set_xticks(df["N Threads"])
ax2.set_xticklabels(df["N Threads"])
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("pi_speedup.png", dpi=150)
plt.show()
