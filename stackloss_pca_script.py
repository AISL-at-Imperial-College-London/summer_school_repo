"""
Stack Loss PCA Example
----------------------
This script:
1. Loads the Stack Loss process dataset.
2. Standardises the four process variables.
3. Fits a PCA model.
4. Prints eigenvalues, explained variance, eigenvectors, loadings, and scores.
5. Produces PC1-PC2 scores and loadings plots.
6. Calculates Hotelling's T^2 and Q statistics using two retained PCs.
7. Saves the plots and numerical results.

Install dependencies with:
    pip install numpy pandas matplotlib scikit-learn statsmodels
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm


# ============================================================
# Settings
# ============================================================

N_COMPONENTS_MONITORING = 2
SAVE_RESULTS = True
OUTPUT_FOLDER = "stackloss_pca_results"


# ============================================================
# 1. Load data
# ============================================================

data = sm.datasets.stackloss.load_pandas().data.copy()

variables = ["AIRFLOW", "WATERTEMP", "ACIDCONC", "STACKLOSS"]
X = data[variables].to_numpy()

print("\nStack Loss dataset:")
print(data[variables])


# ============================================================
# 2. Standardise variables
# ============================================================

scaler = StandardScaler()
Z = scaler.fit_transform(X)

standardised_data = pd.DataFrame(Z, columns=variables)

print("\nStandardised data:")
print(standardised_data.round(3))


# ============================================================
# 3. Fit PCA
# ============================================================

pca = PCA()
scores = pca.fit_transform(Z)

eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Columns are principal directions
eigenvectors = pca.components_.T

# Correlation loadings
loadings = eigenvectors * np.sqrt(eigenvalues)


# ============================================================
# 4. Numerical summaries
# ============================================================

summary = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(eigenvalues))],
    "Eigenvalue": eigenvalues,
    "Explained variance (%)": 100 * explained_variance_ratio,
    "Cumulative variance (%)": 100 * cumulative_variance,
})

pc12_eigenvectors = pd.DataFrame(
    eigenvectors[:, :2],
    index=variables,
    columns=["PC1", "PC2"],
)

pc12_loadings = pd.DataFrame(
    loadings[:, :2],
    index=variables,
    columns=["PC1", "PC2"],
)

pc12_scores = pd.DataFrame(
    scores[:, :2],
    columns=["PC1 score", "PC2 score"],
)

pc12_scores.insert(
    0,
    "Observation",
    np.arange(1, len(pc12_scores) + 1),
)

print("\nPCA summary:")
print(summary.round(4).to_string(index=False))

print("\nPC1 and PC2 eigenvectors:")
print(pc12_eigenvectors.round(4))

print("\nPC1 and PC2 correlation loadings:")
print(pc12_loadings.round(4))

print("\nPC1 and PC2 scores:")
print(pc12_scores.round(4).to_string(index=False))


# ============================================================
# 5. Print PC equations
# ============================================================

pc1_terms = [
    f"({eigenvectors[i, 0]:.4f}) z_{variables[i]}"
    for i in range(len(variables))
]

pc2_terms = [
    f"({eigenvectors[i, 1]:.4f}) z_{variables[i]}"
    for i in range(len(variables))
]

print("\nPC1 score equation:")
print("t1 = " + " + ".join(pc1_terms))

print("\nPC2 score equation:")
print("t2 = " + " + ".join(pc2_terms))


# ============================================================
# 6. Prepare output folder
# ============================================================

output_path = Path(OUTPUT_FOLDER)

if SAVE_RESULTS:
    output_path.mkdir(exist_ok=True)


# ============================================================
# 7. Scores plot
# ============================================================

plt.figure(figsize=(8, 6))
plt.scatter(scores[:, 0], scores[:, 1], s=45)

for i, (pc1, pc2) in enumerate(scores[:, :2], start=1):
    plt.annotate(
        str(i),
        (pc1, pc2),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel(
    f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)"
)
plt.ylabel(
    f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)"
)

plt.title("Stack Loss PCA Scores Plot")
plt.tight_layout()

if SAVE_RESULTS:
    plt.savefig(
        output_path / "stackloss_scores_pc1_pc2.png",
        dpi=220,
        bbox_inches="tight",
    )

plt.show()


# ============================================================
# 8. Loadings plot
# ============================================================

plt.figure(figsize=(8, 6))
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

label_offsets = {
    "AIRFLOW": (8, 10),
    "WATERTEMP": (8, -12),
    "ACIDCONC": (8, 8),
    "STACKLOSS": (8, -1),
}

for i, variable in enumerate(variables):
    pc1_loading = loadings[i, 0]
    pc2_loading = loadings[i, 1]

    plt.arrow(
        0,
        0,
        pc1_loading,
        pc2_loading,
        length_includes_head=True,
        head_width=0.03,
        linewidth=1.4,
    )

    offset_x, offset_y = label_offsets[variable]

    plt.annotate(
        variable,
        (pc1_loading, pc2_loading),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        fontsize=10,
    )

plt.xlim(-1.05, 1.15)
plt.ylim(-1.05, 1.05)
plt.gca().set_aspect("equal", adjustable="box")

plt.xlabel("PC1 correlation loading")
plt.ylabel("PC2 correlation loading")
plt.title("Stack Loss PCA Loadings Plot")
plt.tight_layout()

if SAVE_RESULTS:
    plt.savefig(
        output_path / "stackloss_loadings_pc1_pc2.png",
        dpi=220,
        bbox_inches="tight",
    )

plt.show()


# ============================================================
# 9. Process-monitoring statistics
# ============================================================

q = N_COMPONENTS_MONITORING

Vq = eigenvectors[:, :q]
Tq = scores[:, :q]

# Hotelling's T^2
T2 = np.sum(
    (Tq ** 2) / eigenvalues[:q],
    axis=1,
)

# Reconstruct standardised observations
Z_hat = Tq @ Vq.T

# Residual matrix and Q statistic
E = Z - Z_hat
Q = np.sum(E ** 2, axis=1)

monitoring_results = pd.DataFrame({
    "Observation": np.arange(1, len(data) + 1),
    "T2": T2,
    "Q": Q,
})

print(f"\nProcess-monitoring statistics using {q} retained PCs:")
print(monitoring_results.round(4).to_string(index=False))


# ============================================================
# 10. T^2 plot
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    monitoring_results["Observation"],
    monitoring_results["T2"],
    marker="o",
)

plt.xlabel("Observation")
plt.ylabel(r"Hotelling's $T^2$")
plt.title(r"Stack Loss: Hotelling's $T^2$")
plt.xticks(monitoring_results["Observation"])
plt.tight_layout()

if SAVE_RESULTS:
    plt.savefig(
        output_path / "stackloss_T2.png",
        dpi=220,
        bbox_inches="tight",
    )

plt.show()


# ============================================================
# 11. Q plot
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    monitoring_results["Observation"],
    monitoring_results["Q"],
    marker="o",
)

plt.xlabel("Observation")
plt.ylabel(r"$Q$ statistic")
plt.title(r"Stack Loss: $Q$ Residual Statistic")
plt.xticks(monitoring_results["Observation"])
plt.tight_layout()

if SAVE_RESULTS:
    plt.savefig(
        output_path / "stackloss_Q.png",
        dpi=220,
        bbox_inches="tight",
    )

plt.show()


# ============================================================
# 12. Save numerical results
# ============================================================

if SAVE_RESULTS:
    data[variables].to_csv(
        output_path / "stackloss_data.csv",
        index=False,
    )

    standardised_data.to_csv(
        output_path / "stackloss_standardised_data.csv",
        index=False,
    )

    summary.to_csv(
        output_path / "pca_summary.csv",
        index=False,
    )

    pc12_eigenvectors.to_csv(
        output_path / "pc1_pc2_eigenvectors.csv",
    )

    pc12_loadings.to_csv(
        output_path / "pc1_pc2_loadings.csv",
    )

    pc12_scores.to_csv(
        output_path / "pc1_pc2_scores.csv",
        index=False,
    )

    monitoring_results.to_csv(
        output_path / "T2_Q_statistics.csv",
        index=False,
    )

    print(f"\nResults saved to: {output_path.resolve()}")
