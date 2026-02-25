"""
pyqt5_app.py — PyQt5 Desktop Application
==========================================

Reference implementation showing PyQt5 patterns for building a desktop
stock analysis GUI: window setup, layout management, widget creation,
background threading with signals/slots, tab views, and progress bars.

Patterns from: StockLTSMTransformerQuantum project

Dependencies:
    pip install PyQt5

Usage:
    python pyqt5_app.py
"""

import sys
import time
import random

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QTabWidget,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QStatusBar,
    QMessageBox,
    QSpinBox,
)


# ---------------------------------------------------------------------------
# 1. Background Worker Thread — runs long tasks without freezing the GUI
# ---------------------------------------------------------------------------
class TrainingWorker(QThread):
    """
    QThread subclass that simulates a model training loop.

    In the real StockLTSM project this would run the PyTorch training
    loop.  Communication back to the GUI happens entirely through
    pyqtSignal — never modify GUI widgets directly from a QThread.

    Signals
    -------
    progress : int
        Emitted each epoch with the current progress percentage (0-100).
    epoch_result : dict
        Emitted each epoch with metrics (epoch, train_loss, val_loss).
    finished : str
        Emitted when training completes with a summary message.
    error : str
        Emitted if an exception occurs during training.
    """

    # Define custom signals (must be class-level attributes)
    progress = pyqtSignal(int)
    epoch_result = pyqtSignal(dict)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, ticker: str, epochs: int, parent=None):
        super().__init__(parent)
        self.ticker = ticker
        self.epochs = epochs
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of the training loop."""
        self._is_cancelled = True

    def run(self):
        """
        Execute the training loop in a background thread.

        This method is called automatically when you call worker.start().
        Never call run() directly — always use start().
        """
        try:
            for epoch in range(1, self.epochs + 1):
                if self._is_cancelled:
                    self.finished.emit("Training cancelled by user.")
                    return

                # Simulate training work (replace with actual model training)
                time.sleep(0.15)

                # Simulated metrics
                train_loss = 0.5 * (0.95 ** epoch) + random.uniform(-0.01, 0.01)
                val_loss = 0.55 * (0.93 ** epoch) + random.uniform(-0.02, 0.02)

                # Emit progress (0-100 percentage)
                pct = int((epoch / self.epochs) * 100)
                self.progress.emit(pct)

                # Emit epoch metrics
                self.epoch_result.emit({
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                })

            self.finished.emit(
                f"Training completed: {self.epochs} epochs on {self.ticker}"
            )

        except Exception as e:
            self.error.emit(f"Training error: {str(e)}")


# ---------------------------------------------------------------------------
# 2. Data Fetching Worker
# ---------------------------------------------------------------------------
class DataFetchWorker(QThread):
    """
    Background worker for downloading stock data.

    Keeps the GUI responsive during potentially slow network calls.
    """

    progress = pyqtSignal(int)
    data_ready = pyqtSignal(str)  # Emits a status message with results info
    error = pyqtSignal(str)

    def __init__(self, ticker: str, period: str, parent=None):
        super().__init__(parent)
        self.ticker = ticker
        self.period = period

    def run(self):
        """Simulate data download (replace with yfinance call)."""
        try:
            self.progress.emit(20)
            time.sleep(0.5)   # Simulate network latency

            self.progress.emit(60)
            time.sleep(0.3)

            self.progress.emit(100)
            self.data_ready.emit(
                f"Downloaded {self.ticker} data for period '{self.period}' "
                f"— 252 trading days"
            )

        except Exception as e:
            self.error.emit(f"Download error: {str(e)}")


# ---------------------------------------------------------------------------
# 3. Main Window
# ---------------------------------------------------------------------------
class StockAnalysisWindow(QMainWindow):
    """
    Main application window with tabbed interface.

    Tabs
    ----
    1. Data    — ticker input, period selection, download button
    2. Train   — model configuration, training controls, progress
    3. Results — training log output
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("StockLTSM Analyzer — PyQt5 Reference")
        self.setMinimumSize(800, 600)

        # Track active workers so we can cancel them
        self._training_worker: TrainingWorker | None = None
        self._fetch_worker: DataFetchWorker | None = None

        # Build the UI
        self._init_ui()

    # ---- UI Construction -----------------------------------------------

    def _init_ui(self):
        """Build the complete user interface."""
        # Central widget — QMainWindow requires a central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Root layout
        root_layout = QVBoxLayout(central)

        # --- Tab Widget (multiple views) ---
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        # Build each tab as a separate widget
        self.tabs.addTab(self._build_data_tab(), "Data")
        self.tabs.addTab(self._build_train_tab(), "Train")
        self.tabs.addTab(self._build_results_tab(), "Results")

        # --- Status Bar (bottom of window) ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _build_data_tab(self) -> QWidget:
        """Build the Data tab: ticker input, period selection, download."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ---- Input Group ----
        input_group = QGroupBox("Data Source")
        input_layout = QHBoxLayout(input_group)

        # Ticker input
        input_layout.addWidget(QLabel("Ticker:"))
        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setPlaceholderText("e.g. AAPL, MSFT, GOOGL")
        self.ticker_input.setMaximumWidth(200)
        input_layout.addWidget(self.ticker_input)

        # Period selector (QComboBox)
        input_layout.addWidget(QLabel("Period:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
        self.period_combo.setCurrentText("1y")
        input_layout.addWidget(self.period_combo)

        # Spacer to push button to the right
        input_layout.addStretch()

        # Download button
        self.download_btn = QPushButton("Download Data")
        self.download_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        # Signal/slot connection: clicked → handler
        self.download_btn.clicked.connect(self._on_download_clicked)
        input_layout.addWidget(self.download_btn)

        layout.addWidget(input_group)

        # ---- Progress Bar ----
        self.data_progress = QProgressBar()
        self.data_progress.setValue(0)
        self.data_progress.setTextVisible(True)
        layout.addWidget(self.data_progress)

        # ---- Status Label ----
        self.data_status = QLabel("No data loaded.")
        self.data_status.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.data_status)

        # Push everything to the top
        layout.addStretch()

        return tab

    def _build_train_tab(self) -> QWidget:
        """Build the Train tab: model config, training controls, progress."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ---- Model Configuration Group ----
        config_group = QGroupBox("Model Configuration")
        config_layout = QHBoxLayout(config_group)

        # Model type selector
        config_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "Transformer", "LSTM+Attention", "Hybrid"])
        config_layout.addWidget(self.model_combo)

        # Epochs spinner
        config_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        config_layout.addWidget(self.epochs_spin)

        config_layout.addStretch()

        layout.addWidget(config_group)

        # ---- Training Controls (horizontal layout) ----
        controls_layout = QHBoxLayout()

        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #388E3C; }"
        )
        self.train_btn.clicked.connect(self._on_train_clicked)
        controls_layout.addWidget(self.train_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #D32F2F; }"
        )
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        controls_layout.addWidget(self.cancel_btn)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # ---- Training Progress ----
        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_progress.setFormat("Epoch %v / %m")
        layout.addWidget(self.train_progress)

        # ---- Live Metrics Label ----
        self.metrics_label = QLabel("Awaiting training start...")
        self.metrics_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.metrics_label)

        layout.addStretch()

        return tab

    def _build_results_tab(self) -> QWidget:
        """Build the Results tab: scrollable training log output."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(QLabel("Training Log:"))

        # QTextEdit for scrollable, read-only log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "font-family: 'Courier New', monospace; font-size: 11px; "
            "background-color: #1E1E1E; color: #D4D4D4; padding: 8px;"
        )
        layout.addWidget(self.log_output)

        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_output.clear)
        layout.addWidget(clear_btn)

        return tab

    # ---- Slot Handlers --------------------------------------------------

    @pyqtSlot()
    def _on_download_clicked(self):
        """Handle the Download Data button click."""
        ticker = self.ticker_input.text().strip().upper()
        period = self.period_combo.currentText()

        if not ticker:
            QMessageBox.warning(self, "Input Error", "Please enter a ticker symbol.")
            return

        self.data_progress.setValue(0)
        self.data_status.setText(f"Downloading {ticker} ({period})...")
        self.download_btn.setEnabled(False)
        self.status_bar.showMessage(f"Fetching {ticker}...")

        # Start background download
        self._fetch_worker = DataFetchWorker(ticker, period)
        self._fetch_worker.progress.connect(self.data_progress.setValue)
        self._fetch_worker.data_ready.connect(self._on_data_ready)
        self._fetch_worker.error.connect(self._on_data_error)
        self._fetch_worker.start()

    @pyqtSlot(str)
    def _on_data_ready(self, message: str):
        """Handle successful data download."""
        self.data_status.setText(message)
        self.data_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.download_btn.setEnabled(True)
        self.status_bar.showMessage("Data download complete.", 5000)
        self._append_log(f"[DATA] {message}")

    @pyqtSlot(str)
    def _on_data_error(self, message: str):
        """Handle data download error."""
        self.data_status.setText(message)
        self.data_status.setStyleSheet("color: #F44336; font-weight: bold;")
        self.download_btn.setEnabled(True)
        self.status_bar.showMessage("Download failed.", 5000)
        self._append_log(f"[ERROR] {message}")

    @pyqtSlot()
    def _on_train_clicked(self):
        """Handle the Start Training button click."""
        ticker = self.ticker_input.text().strip().upper() or "AAPL"
        epochs = self.epochs_spin.value()
        model = self.model_combo.currentText()

        # Update UI state for training mode
        self.train_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.train_progress.setMaximum(100)
        self.train_progress.setValue(0)
        self.metrics_label.setText("Training started...")
        self.status_bar.showMessage(f"Training {model} on {ticker}...")

        self._append_log(
            f"[TRAIN] Starting {model} training: "
            f"{ticker}, {epochs} epochs"
        )

        # Create and start the training worker thread
        self._training_worker = TrainingWorker(ticker, epochs)
        self._training_worker.progress.connect(self.train_progress.setValue)
        self._training_worker.epoch_result.connect(self._on_epoch_result)
        self._training_worker.finished.connect(self._on_training_finished)
        self._training_worker.error.connect(self._on_training_error)
        self._training_worker.start()

    @pyqtSlot()
    def _on_cancel_clicked(self):
        """Handle the Cancel button click."""
        if self._training_worker and self._training_worker.isRunning():
            self._training_worker.cancel()
            self.status_bar.showMessage("Cancelling training...")

    @pyqtSlot(dict)
    def _on_epoch_result(self, metrics: dict):
        """Update the UI with per-epoch training metrics."""
        self.metrics_label.setText(
            f"Epoch {metrics['epoch']:>3d}  |  "
            f"Train Loss: {metrics['train_loss']:.6f}  |  "
            f"Val Loss: {metrics['val_loss']:.6f}"
        )

    @pyqtSlot(str)
    def _on_training_finished(self, message: str):
        """Handle training completion."""
        self.train_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_bar.showMessage(message, 10000)
        self._append_log(f"[DONE] {message}")

        # Switch to results tab to show the log
        self.tabs.setCurrentIndex(2)

    @pyqtSlot(str)
    def _on_training_error(self, message: str):
        """Handle training error."""
        self.train_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_bar.showMessage("Training failed.", 5000)
        self._append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "Training Error", message)

    # ---- Helpers --------------------------------------------------------

    def _append_log(self, text: str):
        """Append a timestamped line to the results log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {text}")


# ---------------------------------------------------------------------------
# 4. Application Entry Point
# ---------------------------------------------------------------------------
def main():
    """
    Standard PyQt5 application lifecycle:
      1. Create QApplication (must be first)
      2. Create main window
      3. Show window
      4. Enter event loop with sys.exit(app.exec_())
    """
    # QApplication must be created before any QWidget
    app = QApplication(sys.argv)

    # Optional: set application-wide stylesheet
    app.setStyle("Fusion")

    # Create and show the main window
    window = StockAnalysisWindow()
    window.show()

    # Enter the Qt event loop — this blocks until the window is closed.
    # sys.exit() ensures a clean exit code is returned to the OS.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
