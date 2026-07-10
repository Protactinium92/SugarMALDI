import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from pathlib import Path

# ------------------------------------------------------------------
# Conserver le texte comme du texte dans le SVG
# ------------------------------------------------------------------
plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------
# Données
# ------------------------------------------------------------------
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

gm = np.array([38, 45, 104, 228, 350, 410, 490, 755, 1305])
ag = np.array([10, 17, 24, 40, 75, 118, 205, 347, 735])

# ------------------------------------------------------------------
# Courbes lissées passant par tous les points
# ------------------------------------------------------------------
x_smooth = np.linspace(x.min(), x.max(), 300)

gm_trend = make_interp_spline(x, gm, k=3)(x_smooth)
ag_trend = make_interp_spline(x, ag, k=3)(x_smooth)

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

color_gm = "#e64980"
color_ag = "#228be6"

# Points
ax.scatter(
    x,
    gm,
    color=color_gm,
    edgecolor="white",
    s=50,
    zorder=3,
    label="Execution time (s) - GM",
)

ax.scatter(
    x,
    ag,
    color=color_ag,
    edgecolor="white",
    s=50,
    zorder=3,
    label="Execution time (s) - AG",
)

# Courbes
ax.plot(
    x_smooth,
    gm_trend,
    color=color_gm,
    linestyle="--",
    linewidth=2,
)

ax.plot(
    x_smooth,
    ag_trend,
    color=color_ag,
    linestyle="--",
    linewidth=2,
)

# ------------------------------------------------------------------
# Habillage
# ------------------------------------------------------------------
ax.set_xlabel("Number of selected monosaccharides")
ax.set_ylabel("Execution time (s)")
ax.set_title("Execution time versus number of selected monosaccharides")

ax.set_xticks(x)

ax.grid(True, alpha=0.2)

ax.legend(frameon=False)

# Fond transparent
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.tight_layout()

# ------------------------------------------------------------------
# Export SVG
# ------------------------------------------------------------------
output = Path("chart.svg")

plt.savefig(
    output,
    format="svg",
    transparent=True,
)

plt.close()

# ------------------------------------------------------------------
# Adapter automatiquement au thème GitHub
# ------------------------------------------------------------------
svg = output.read_text(encoding="utf-8")

replacements = {
    "#000000": "currentColor",
    "#000": "currentColor",
    "black": "currentColor",
    "rgb(0%,0%,0%)": "currentColor",
}

for old, new in replacements.items():
    svg = svg.replace(old, new)

output.write_text(svg, encoding="utf-8")

print("✅ chart.svg generated successfully.")