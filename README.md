# word2vec implementation in pure NumPy 

skip-gram with negative sampling

## Dataset 

https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish

The dataset was split in half to comply with GitHub's 100 MB file size limit.

## Setup

### Prerequisites

- Python 3.11+
- pip

### Install

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requerements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requerements.txt
```

### Run

```bash
python main.py
```