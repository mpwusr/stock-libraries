# Stock Libraries — Python Reference Implementations

Reference code examples for the 13 Python libraries used in the [StockLTSMTransformerQuantum](https://github.com/mpwusr/StockLTSMTransformerQuantum) project.

## Library Inventory

### Deep Learning & Quantum ML (5)
| Library | Category | Example File |
|---|---|---|
| TensorFlow/Keras | Deep Learning | `deep_learning_quantum/keras_lstm_training.py`, `keras_transformer.py` |
| PennyLane | Quantum ML | `deep_learning_quantum/pennylane_quantum.py` |
| scikit-learn | ML Preprocessing | `deep_learning_quantum/sklearn_preprocessing.py` |
| NumPy | Array Operations | `deep_learning_quantum/numpy_timeseries.py` |

### Financial Data & Technical Analysis (4)
| Library | Category | Example File |
|---|---|---|
| yfinance | Stock Data | `finance_data/yfinance_download.py` |
| ta | Technical Indicators | `finance_data/ta_indicators.py` |
| alpha-vantage | Financial API | `finance_data/alphavantage_api.py` |
| pandas | Data Analysis | `finance_data/pandas_finance.py` |

### Visualization & GUI (2)
| Library | Category | Example File |
|---|---|---|
| matplotlib | Charting | `visualization_gui/matplotlib_finance.py` |
| PyQt5 | Desktop GUI | `visualization_gui/pyqt5_app.py`, `pyqt5_matplotlib.py` |

### Utilities (2)
| Library | Category | Example File |
|---|---|---|
| python-dotenv | Env Management | `utilities/dotenv_validation.py` |
| requests + tenacity | HTTP + Retry | `utilities/requests_financial_api.py` |

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # Edit with your API keys
```

## Author
Michael P. Williams
