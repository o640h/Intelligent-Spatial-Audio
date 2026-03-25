from pathlib import Path

import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class StereoScopeView(QWidget):
    def __init__(self, title="Stereo Scope"):
        super().__init__()

        self.title_label = QLabel(title)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")
        self.plot.setXRange(-1.05, 1.05)
        self.plot.setYRange(-1.05, 1.05)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("left", "Right")
        self.plot.setLabel("bottom", "Left")
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setAspectLocked(True)

        self.scatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(220, 220, 220, 90))
        self.plot.addItem(self.scatter)

        self.info_label = QLabel("No file loaded")

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.plot)
        layout.addWidget(self.info_label)

        self.audio = None
        self.sample_rate = None
        self.playhead_seconds = 0.0
        self.window_ms = 80
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scope)

    def load_file(self, file_path: str):
        self.plot.clear()
        self.plot.addItem(self.scatter)

        path = Path(file_path)
        if not path.exists():
            self.audio = None
            self.title_label.setText("Stereo Scope — file not found")
            self.info_label.setText("No file loaded")
            self.scatter.setData([], [])
            return

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)

        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]

        self.audio = audio
        self.sample_rate = sr
        self.playhead_seconds = 0.0

        self.title_label.setText(path.name)
        self.info_label.setText(f"Loaded | {len(audio)/sr:.2f}s | {sr} Hz")
        self.update_scope()

    def set_playhead(self, seconds: float):
        self.playhead_seconds = max(0.0, seconds)

    def start(self):
        if self.audio is not None:
            self.timer.start(33)

    def stop(self):
        self.timer.stop()
        self.playhead_seconds = 0.0
        self.update_scope()

    def pause(self):
        self.timer.stop()

    def update_scope(self):
        if self.audio is None or self.sample_rate is None:
            self.scatter.setData([], [])
            return

        center = int(self.playhead_seconds * self.sample_rate)
        half_window = int((self.window_ms / 1000.0) * self.sample_rate / 2)

        start = max(0, center - half_window)
        end = min(len(self.audio), center + half_window)

        if end - start < 32:
            self.scatter.setData([], [])
            return

        segment = self.audio[start:end]
        left = segment[:, 0]
        right = segment[:, 1]

        max_points = 1500
        if len(left) > max_points:
            step = max(1, len(left) // max_points)
            left = left[::step]
            right = right[::step]

        self.scatter.setData(left, right)

        denom = (np.std(left) * np.std(right)) + 1e-8
        corr = float(np.mean((left - np.mean(left)) * (right - np.mean(right))) / denom)
        self.info_label.setText(
            f"Playhead: {self.playhead_seconds:.2f}s | Correlation: {corr:.3f}"
        )