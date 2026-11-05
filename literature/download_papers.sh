#!/usr/bin/env bash
# Download the open-access arXiv PDFs listed in literature/README.md
# into literature/pdfs/ (git-ignored). Run from the repository root.
set -euo pipefail

OUT="$(dirname "$0")/pdfs"
mkdir -p "$OUT"

papers=(
  "2306.16424 01_altman2023_aml_synthetic_dataset"
  "2306.11586 02_egressy2024_multigraph_gnn"
  "2402.08593 03_blanusa2024_graph_feature_preprocessor"
  "1908.02591 04_weber2019_elliptic_gcn"
  "2309.01127 05_innan2024_quantum_gnn_fraud"
  "2208.07963 06_grossi2022_qsvm_fraud"
  "2308.05237 07_innan2023_qml_fraud_comparison"
  "2508.09237 07b_damico2025_quantum_inspired_gnn_aml"
  "1909.12264 08_verdon2019_quantum_gnn"
  "2205.06109 09_skolik2023_equivariant_graph_circuits"
  "2011.01938 10_huang2021_power_of_data"
  "1804.11326 11_havlicek2019_quantum_feature_spaces"
  "1803.07128 12_schuld2019_feature_hilbert_spaces"
)

for entry in "${papers[@]}"; do
  id="${entry%% *}"
  name="${entry#* }"
  file="$OUT/${name}.pdf"
  if [[ -f "$file" ]]; then
    echo "skip  $name"
    continue
  fi
  echo "fetch $name ($id)"
  curl -fsSL "https://arxiv.org/pdf/${id}" -o "$file"
  sleep 3  # be polite to arXiv
done
