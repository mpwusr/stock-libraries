"""
pyqt5_matplotlib.py — Embedding matplotlib in PyQt5
=====================================================

Reference implementation showing how to embed interactive matplotlib
charts inside a PyQt5 desktop application: FigureCanvasQTAgg as a
widget, NavigationToolbar2QT for zoom/pan, dynamic plot updates from
a background thread, and side-by-side comparison layouts.

Patterns from: StockLTSMTransformerQuantum project

Dependencies:
    pip install PyQt5 matplotlib numpy

Usage:
    python pyqt5_matplotlib.py
"""

import sys
import time
import random
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSplitter,
    QGroupBox,
    QStatusBar,
)

# matplotlib Qt5 backend imports
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# 1. Matplotlib Canvas Widget — wraps a Figure as a QWidget
# ---------------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    """
    A FigureCanvasQTAgg subclass that can be embedded as a QWidget.

    This is the bridge between matplotlib's Figure and PyQt5's widget
    system. You can add this canvas to any QLayout just like a QLabel
    or QPushButton.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.
    dpi : int
        Dots per inch (resolution).
    """

    def __init__(self, parent=None, width: float = 6, height: float = 4,
                 dpi: int = 100):
        # Create the matplotlib Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        # Initialize the canvas with our figure
        super().__init__(self.fig)
        self.setParent(parent)

    def clear_and_get_axes(self, nrows: int = 1, ncols: int = 1,
                           subplot_index: int = 1):
        """
        Clear the figure and return a fresh Axes.

        This is the standard update pattern:
          1. figure.clear()
          2. figure.add_subplot(...)
          3. ... draw on axes ...
          4. canvas.draw()

        Parameters
        ----------
        nrows, ncols : int
            Subplot grid dimensions.
        subplot_index : int
            Which subplot to return (1-indexed).

        Returns
        -------
        matplotlib.axes.Axes
        """
        self.fig.clear()
        ax = self.fig.add_subplot(nrows, ncols, subplot_index)
        return ax

    def refresh(self):
        """Redraw the canvas after modifying the figure."""
        self.fig.tight_layout()
        self.draw()


# ---------------------------------------------------------------------------
# 2. Background Worker — generates data and emits plot updates
# ---------------------------------------------------------------------------
class SimulationWorker(QThread):
    """
    Background thread that simulates a live training loop and emits
    plot data via signals.

    In the real StockLTSM project, this would be the actual model
    training loop emitting loss values each epoch.

    Signals
    -------
    new_data : dict
        Emitted each iteration with keys:
        'epoch', 'train_loss', 'val_loss', 'prices', 'predictions'
    finished : str
        Emitted when the simulation completes.
    """

    new_data = pyqtSignal(dict)
    finished = pyqtSignal(str)

    def __init__(self, n_epochs: int = 60, parent=None):
        super().__init__(parent)
        self.n_epochs = n_epochs
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation."""
        self._is_cancelled = True

    def run(self):
        """Run the simulation in a background thread."""
        train_losses = []
        val_losses = []

        # Generate a synthetic price series for the "prediction" chart
        np.random.seed(42)
        n_points = 200
        t = np.linspace(0, 4 * np.pi, n_points)
        base_prices = 150 + 20 * np.sin(t) + np.cumsum(
            np.random.normal(0, 0.5, n_points)
        )

        for epoch in range(1, self.n_epochs + 1):
            if self._is_cancelled:
                self.finished.emit("Simulation cancelled.")
                return

            time.sleep(0.1)  # Simulate computation time

            # Simulated loss values (exponential decay + noise)
            tl = 0.8 * np.exp(-0.04 * epoch) + 0.02 + random.gauss(0, 0.005)
            vl = 0.9 * np.exp(-0.035 * epoch) + 0.03 + random.gauss(0, 0.008)
            train_losses.append(max(tl, 0.01))
            val_losses.append(max(vl, 0.01))

            # Simulated predictions that improve over epochs
            noise_scale = 0.5 * np.exp(-0.03 * epoch) + 0.02
            predictions = base_prices + np.random.normal(0, noise_scale * 10,
                                                         n_points)

            self.new_data.emit({
                "epoch": epoch,
                "train_losses": list(train_losses),
                "val_losses": list(val_losses),
                "prices": base_prices.tolist(),
                "predictions": predictions.tolist(),
            })

        self.finished.emit(
            f"Simulation complete: {self.n_epochs} epochs."
        )


# ---------------------------------------------------------------------------
# 3. Chart Panel — canvas + toolbar in a group box
# ---------------------------------------------------------------------------
class ChartPanel(QWidget):
    """
    Reusable panel containing an MplCanvas and a NavigationToolbar.

    The NavigationToolbar2QT provides built-in zoom, pan, home, and
    save functionality without any extra code.

    Parameters
    ----------
    title : str
        Group box title displayed above the chart.
    parent : QWidget or None
        Parent widget.
    """

    def __init__(self, title: str = "Chart", parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)

        # Create the matplotlib canvas
        self.canvas = MplCanvas(self, width=6, height=4, dpi=100)

        # Create the navigation toolbar (zoom, pan, save, etc.)
        self.toolbar = NavigationToolbar(self.canvas, self)

        group_layout.addWidget(self.toolbar)
        group_layout.addWidget(self.canvas)

        layout.addWidget(group)


# ---------------------------------------------------------------------------
# 4. Main Window — side-by-side charts with dynamic updates
# ---------------------------------------------------------------------------
class MatplotlibApp(QMainWindow):
    """
    Main window with two side-by-side matplotlib charts:
      Left  — Training loss curves (updated each epoch)
      Right — Price vs Prediction comparison (updated each epoch)

    A background SimulationWorker feeds both charts with new data.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("StockLTSM — matplotlib + PyQt5 Reference")
        self.setMinimumSize(1200, 600)

        self._worker: SimulationWorker | None = None

        self._init_ui()

    def _init_ui(self):
        """Build the UI: controls bar + two side-by-side chart panels."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ---- Top Controls Bar ----
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Epochs:"))
        self.epochs_combo = QComboBox()
        self.epochs_combo.addItems(["30", "60", "100", "150"])
        self.epochs_combo.setCurrentText("60")
        controls_layout.addWidget(self.epochs_combo)

        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 6px 14px; border-radius: 4px; }"
        )
        self.start_btn.clicked.connect(self._on_start)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; "
            "padding: 6px 14px; border-radius: 4px; }"
        )
        self.stop_btn.clicked.connect(self._on_stop)
        controls_layout.addWidget(self.stop_btn)

        self.epoch_label = QLabel("Epoch: 0")
        self.epoch_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; font-weight: bold;"
        )
        controls_layout.addWidget(self.epoch_label)

        controls_layout.addStretch()
        root_layout.addLayout(controls_layout)

        # ---- Side-by-Side Charts (using QSplitter for resizable split) ----
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Training loss curves
        self.loss_panel = ChartPanel("Training Loss Curves")
        splitter.addWidget(self.loss_panel)

        # Right panel: Price vs Prediction
        self.pred_panel = ChartPanel("Price vs Prediction")
        splitter.addWidget(self.pred_panel)

        # Start with equal widths
        splitter.setSizes([600, 600])

        root_layout.addWidget(splitter)

        # ---- Status Bar ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — click Start Simulation")

        # Draw initial empty charts
        self._draw_empty_charts()

    def _draw_empty_charts(self):
        """Draw placeholder charts before simulation starts."""
        # Left: empty loss chart
        ax = self.loss_panel.canvas.clear_and_get_axes()
        ax.set_title("Training Loss", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.text(0.5, 0.5, "Awaiting simulation...",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")
        ax.grid(True, alpha=0.3)
        self.loss_panel.canvas.refresh()

        # Right: empty prediction chart
        ax = self.pred_panel.canvas.clear_and_get_axes()
        ax.set_title("Price vs Prediction", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Price ($)")
        ax.text(0.5, 0.5, "Awaiting simulation...",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")
        ax.grid(True, alpha=0.3)
        self.pred_panel.canvas.refresh()

    # ---- Slot: Start Simulation ----------------------------------------

    @pyqtSlot()
    def _on_start(self):
        """Launch the background simulation worker."""
        n_epochs = int(self.epochs_combo.currentText())

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_bar.showMessage(f"Running simulation ({n_epochs} epochs)...")

        self._worker = SimulationWorker(n_epochs=n_epochs)
        self._worker.new_data.connect(self._on_new_data)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    # ---- Slot: Stop Simulation -----------------------------------------

    @pyqtSlot()
    def _on_stop(self):
        """Cancel the running simulation."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()

    # ---- Slot: New Data from Worker ------------------------------------

    @pyqtSlot(dict)
    def _on_new_data(self, data: dict):
        """
        Update both charts with new data from the background thread.

        This is the core dynamic update pattern:
          1. figure.clear()
          2. add_subplot(111)
          3. Plot new data
          4. canvas.draw()
        """
        epoch = data["epoch"]
        self.epoch_label.setText(f"Epoch: {epoch}")

        # ---- Update Left Chart: Loss Curves ----
        ax = self.loss_panel.canvas.clear_and_get_axes()

        epochs_range = list(range(1, len(data["train_losses"]) + 1))
        ax.plot(epochs_range, data["train_losses"],
                label="Train Loss", color="#2196F3", linewidth=1.5)
        ax.plot(epochs_range, data["val_losses"],
                label="Val Loss", color="#F44336", linewidth=1.5)

        # Fill the generalization gap
        ax.fill_between(
            epochs_range,
            data["train_losses"], data["val_losses"],
            alpha=0.1, color="orange",
        )

        ax.set_title(f"Training Loss — Epoch {epoch}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        self.loss_panel.canvas.refresh()

        # ---- Update Right Chart: Price vs Prediction ----
        ax = self.pred_panel.canvas.clear_and_get_axes()

        prices = data["prices"]
        preds = data["predictions"]
        x = list(range(len(prices)))

        ax.plot(x, prices, label="Actual Price",
                color="#2196F3", linewidth=1.2, alpha=0.9)
        ax.plot(x, preds, label="Prediction",
                color="#FF9800", linewidth=1.0, alpha=0.7)

        # Show residual as a shaded area
        ax.fill_between(x, prices, preds, alpha=0.08, color="red")

        # Compute and display RMSE
        rmse = np.sqrt(np.mean((np.array(prices) - np.array(preds)) ** 2))
        ax.set_title(f"Price vs Prediction — RMSE: {rmse:.2f}", fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        self.pred_panel.canvas.refresh()

    # ---- Slot: Simulation Finished -------------------------------------

    @pyqtSlot(str)
    def _on_finished(self, message: str):
        """Handle simulation completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_bar.showMessage(message, 10000)


# ---------------------------------------------------------------------------
# 5. Application Entry Point
# ---------------------------------------------------------------------------
def main():
    """
    Standard PyQt5 + matplotlib application lifecycle.

    The key integration points are:
      - FigureCanvasQTAgg bridges matplotlib Figure → QWidget
      - NavigationToolbar2QT adds zoom/pan/save controls
      - QThread + pyqtSignal feeds data to canvas.draw() updates
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MatplotlibApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
