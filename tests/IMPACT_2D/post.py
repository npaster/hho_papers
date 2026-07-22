import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Lecture du fichier
# ============================================================

df = pd.read_csv(
"pointA.csv",
sep=";",
engine="python"
)

df.columns = [c.strip() for c in df.columns]

t = df["Time"].to_numpy()
ux = df["ux"].to_numpy()
uy = df["uy"].to_numpy()
vx = df["vx"].to_numpy()
vy = df["vy"].to_numpy()
syy = df["syy"].to_numpy()

# ============================================================
# Solution Analytical périodique (PERIODE = 3)
# ============================================================

tau = np.mod(t, 3.0)

ux_ana = np.zeros_like(t)
vx_ana = np.zeros_like(t)

uy_ana = np.zeros_like(t)
vy_ana = np.zeros_like(t)
syy_ana = np.zeros_like(t)

# 0 <= tau < 1
m1 = (tau >= 0.0) & (tau < 1.0)
uy_ana[m1] = 0.5 * (1.0 - tau[m1])
vy_ana[m1] = -0.5
syy_ana[m1] = 0.0

# 1 <= tau < 2
m2 = (tau >= 1.0) & (tau < 2.0)
uy_ana[m2] = 0.0
vy_ana[m2] = 0.0
syy_ana[m2] = -0.5

# 2 <= tau < 3
m3 = (tau >= 2.0) & (tau < 3.0)
uy_ana[m3] = 0.5 * (tau[m3] - 2.0)
vy_ana[m3] = 0.5
syy_ana[m3] = 0.0

# ============================================================
# Figure 2x2
# ============================================================

fig, axs = plt.subplots(
2,
2,
figsize=(14, 10),
constrained_layout=True
)

# ------------------------------------------------------------
# ux
# ------------------------------------------------------------
ax = axs[0, 0]

ax.plot(t, ux, lw=2, label="Numerical")
ax.plot(t, ux_ana, "k--", lw=2, label="Analytical")

ax.set_title(r"$u_x(t)$")
ax.set_xlabel("Time")
ax.set_ylabel(r"$u_x$")
ax.grid(True)
ax.legend()

# ------------------------------------------------------------
# uy
# ------------------------------------------------------------
ax = axs[0, 1]

ax.plot(t, uy, lw=2, label="Numerical")
ax.plot(t, uy_ana, "k--", lw=2, label="Analytical")

ax.set_title(r"$u_y(t)$")
ax.set_xlabel("Time")
ax.set_ylabel(r"$u_y$")
ax.grid(True)
ax.legend()

# ------------------------------------------------------------
# vx
# ------------------------------------------------------------
ax = axs[1, 0]

ax.plot(t, vy, lw=2, label="Numerical")
ax.plot(t, vy_ana, "k--", lw=2, label="Analytical")

ax.set_title(r"$v_y(t)$")
ax.set_xlabel("Time")
ax.set_ylabel(r"$v_y$")
ax.grid(True)
ax.legend()

# ------------------------------------------------------------
# vy
# ------------------------------------------------------------
ax = axs[1, 1]

ax.plot(t, syy, lw=2, label="Numerical")
ax.plot(t, syy_ana, "k--", lw=2, label="Analytical")

ax.set_title(r"$\sigma_n(t)$")
ax.set_xlabel("Time")
ax.set_ylabel(r"$\sigma_n$")
ax.grid(True)
ax.legend()

# ============================================================
# Erreurs norme infinie
# ============================================================

print()
print("Erreurs norme infinie")
print("-" * 40)
print(f"||ux-ux_exact||inf = {np.max(np.abs(ux-ux_ana)):.6e}")
print(f"||uy-uy_exact||inf = {np.max(np.abs(uy-uy_ana)):.6e}")
print(f"||vx-vx_exact||inf = {np.max(np.abs(vx-vx_ana)):.6e}")
print(f"||vy-vy_exact||inf = {np.max(np.abs(vy-vy_ana)):.6e}")
print(f"||sn-sn_exact||inf = {np.max(np.abs(syy-syy_ana)):.6e}")

# ============================================================
# Sauvegarde
# ============================================================

plt.savefig("IMPACT_2D.png", dpi=300)
plt.savefig("IMPACT_2D.pdf")

plt.show()
