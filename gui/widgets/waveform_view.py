from pathlib import Path

import numpy as np
import soundfile as sf
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel


class WaveformView(QWidget):
    def __init__(self, title="Waveform"):
        super().__init__()

        self.title_label = QLabel(title)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.hideButtons()

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.plot)

    def load_file(self, file_path: str):
        self.plot.clear()

        path = Path(file_path)
        if not path.exists():
            self.title_label.setText(f"{self.title_label.text().split(' — ')[0]} — file not found")
            return

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)

        # Convert to mono for display only
        mono = np.mean(audio, axis=1)

        # Downsample for plotting if very long
        max_points = 12000
        if len(mono) > max_points:
            step = len(mono) // max_points
            mono = mono[::step]

        x = np.linspace(0, len(audio) / sr, len(mono))
        self.plot.plot(x, mono, pen=pg.mkPen(width=1))

        self.title_label.setText(f"{path.name}")