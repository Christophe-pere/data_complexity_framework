# data-complexity-framework
Framework to determine the separation between the classical and quantum regimes based on the papers data complexity. 

## Module layout (Python package style)

data_complexity_framework/
│
├── datasets/
│   ├── __init__.py
│   └── synthetic.py   # parity, Gaussian mixtures, spin models (start with parity)
│
├── metrics/
│   ├── __init__.py
│   ├── entropy.py     # entropy estimators
│   ├── interaction.py # mutual info, higher-order interactions
│   └── compression.py # compression-based complexity
│
├── quantum/
│   ├── __init__.py
│   ├── embeddings.py  # quantum feature maps
│   └── complexity.py  # induced quantum complexity metrics (QFI, entanglement, kernel spectrum)
│
├── classical/
│   ├── __init__.py
│   └── baselines.py   # logistic regression, kernel SVM, etc.
│
├── core.py            # orchestrates experiments, builds complexity cards
└── demo_parity.py     # simple demo script

---
