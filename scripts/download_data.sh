#!/usr/bin/env bash
# Download the Jigsaw Toxic Comment Classification Challenge data with the Kaggle API.
#
# Prerequisites (one-time):
#   1. pip install kaggle
#   2. Kaggle -> Settings -> API -> "Create New Token" -> save kaggle.json to ~/.kaggle/
#      (or export KAGGLE_USERNAME / KAGGLE_KEY)
#   3. Accept the competition rules once in the browser:
#      https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/rules
#
# The data is not redistributed in this repository.
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

if [[ -f "$DATA_DIR/train.csv" && -f "$DATA_DIR/test.csv" && -f "$DATA_DIR/test_labels.csv" ]]; then
  echo "Data already present in $DATA_DIR"
  exit 0
fi

kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p "$DATA_DIR"

# the competition archive contains one zip per CSV
python - "$DATA_DIR" <<'EOF'
import sys
from pathlib import Path
from zipfile import ZipFile
d = Path(sys.argv[1])
for _ in range(2):
    for z in d.glob("*.zip"):
        ZipFile(z).extractall(d)
missing = [f for f in ("train.csv", "test.csv", "test_labels.csv") if not (d / f).exists()]
if missing:
    sys.exit(f"Still missing {missing} in {d}")
print("Extracted:", sorted(p.name for p in d.glob("*.csv")))
EOF
