from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from PySide6.QtCore import QThread, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from workers import PipelineWorker
from widgets.waveform_view import WaveformView
from widgets.stereo_scope import StereoScopeView


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def apply_width_trim_to_file(input_path: str, output_path: str, trim_value: int):
    """
    trim_value:
        -100 = narrower
         0   = unchanged
         100 = wider
    """
    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)

    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    left = audio[:, 0]
    right = audio[:, 1]

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    # Map slider range [-100,100] to side multiplier [0.0, 2.0]
    side_scale = 1.0 + (trim_value / 100.0)
    side_scale = clamp(side_scale, 0.0, 2.0)

    new_left = mid + side * side_scale
    new_right = mid - side * side_scale

    out = np.stack([new_left, new_right], axis=1)

    peak = np.max(np.abs(out)) + 1e-8
    if peak > 0.98:
        out *= 0.98 / peak

    sf.write(output_path, out.astype(np.float32), sr, subtype="PCM_16")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Spatial Audio")
        self.resize(1450, 850)

        self.thread = None
        self.worker = None

        self.current_input = None
        self.current_output_mix = None
        self.current_output_dir = None
        self.current_preview_mix = None

        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(0.8)

        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ================= LEFT PANEL =================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        settings_group = QGroupBox("Pipeline Settings")
        settings_layout = QGridLayout(settings_group)

        self.input_edit = QLineEdit()
        self.input_btn = QPushButton("Browse...")
        self.input_btn.clicked.connect(self.browse_input)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["mdx_extra", "htdemucs_ft", "htdemucs"])

        self.ml_model_edit = QLineEdit("models/spatial_refiner.joblib")

        self.ml_blend_slider = QSlider(Qt.Horizontal)
        self.ml_blend_slider.setMinimum(0)
        self.ml_blend_slider.setMaximum(100)
        self.ml_blend_slider.setValue(20)
        self.ml_blend_label = QLabel("0.20")
        self.ml_blend_slider.valueChanged.connect(self.update_blend_label)

        self.deap_checkbox = QCheckBox("Use DEAP optimisation")
        self.deap_checkbox.setChecked(True)

        settings_layout.addWidget(QLabel("Input Track"), 0, 0)
        settings_layout.addWidget(self.input_edit, 0, 1)
        settings_layout.addWidget(self.input_btn, 0, 2)

        settings_layout.addWidget(QLabel("Demucs Model"), 1, 0)
        settings_layout.addWidget(self.model_combo, 1, 1, 1, 2)

        settings_layout.addWidget(QLabel("ML Model"), 2, 0)
        settings_layout.addWidget(self.ml_model_edit, 2, 1, 1, 2)

        settings_layout.addWidget(QLabel("ML Blend"), 3, 0)
        settings_layout.addWidget(self.ml_blend_slider, 3, 1)
        settings_layout.addWidget(self.ml_blend_label, 3, 2)

        settings_layout.addWidget(self.deap_checkbox, 4, 0, 1, 2)

        left_layout.addWidget(settings_group)

        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Spatialise")
        self.run_btn.clicked.connect(self.run_pipeline)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        left_layout.addWidget(self.log_box)

        # ================= RIGHT PANEL =================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        compare_group = QGroupBox("Input Preview")
        compare_layout = QVBoxLayout(compare_group)
        self.input_waveform = WaveformView("Input")
        compare_layout.addWidget(self.input_waveform)
        right_layout.addWidget(compare_group, stretch=2)

        output_group = QGroupBox("Enhanced Output")
        output_layout = QVBoxLayout(output_group)

        self.stereo_scope = StereoScopeView("Enhanced Output Stereo Scope")
        output_layout.addWidget(self.stereo_scope, stretch=5)

        transport_layout = QHBoxLayout()
        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)

        self.open_mix_btn = QPushButton("Open Final Mix")
        self.open_mix_btn.clicked.connect(self.open_final_mix)
        self.open_mix_btn.setEnabled(False)

        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)

        transport_layout.addWidget(self.play_pause_btn)
        transport_layout.addWidget(self.stop_btn)
        transport_layout.addStretch()
        transport_layout.addWidget(self.open_mix_btn)
        transport_layout.addWidget(self.open_folder_btn)

        output_layout.addLayout(transport_layout)

        width_layout = QHBoxLayout()
        self.width_trim_label = QLabel("Width Trim")
        self.width_trim_slider = QSlider(Qt.Horizontal)
        self.width_trim_slider.setMinimum(-100)
        self.width_trim_slider.setMaximum(100)
        self.width_trim_slider.setValue(0)
        self.width_trim_value = QLabel("0")
        self.width_trim_slider.valueChanged.connect(self.update_width_trim_label)
        self.width_trim_slider.sliderReleased.connect(self.apply_width_trim_preview)

        width_layout.addWidget(self.width_trim_label)
        width_layout.addWidget(self.width_trim_slider)
        width_layout.addWidget(self.width_trim_value)

        output_layout.addLayout(width_layout)

        self.output_path_label = QLineEdit()
        self.output_path_label.setReadOnly(True)
        output_layout.addWidget(self.output_path_label)

        right_layout.addWidget(output_group, stretch=4)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([620, 830])

    def browse_input(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a);;All Files (*)",
        )
        if file_path:
            self.input_edit.setText(file_path)
            self.current_input = file_path
            self.input_waveform.load_file(file_path)

    def update_blend_label(self):
        value = self.ml_blend_slider.value() / 100.0
        self.ml_blend_label.setText(f"{value:.2f}")

    def update_width_trim_label(self):
        self.width_trim_value.setText(str(self.width_trim_slider.value()))

    def append_log(self, text):
        self.log_box.appendPlainText(text.rstrip())

    def run_pipeline(self):
        infile = self.input_edit.text().strip()
        if not infile:
            QMessageBox.warning(self, "Missing input", "Please select an input audio file.")
            return

        if not Path(infile).exists():
            QMessageBox.warning(self, "Invalid file", "Selected input file does not exist.")
            return

        self.run_btn.setEnabled(False)
        self.log_box.clear()

        model = self.model_combo.currentText()
        ml_model = self.ml_model_edit.text().strip()
        ml_blend = self.ml_blend_slider.value() / 100.0
        use_deap = self.deap_checkbox.isChecked()

        self.current_input = infile
        self.input_waveform.load_file(infile)

        self.thread = QThread()
        self.worker = PipelineWorker(infile, model, ml_model, ml_blend, use_deap)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.pipeline_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def handle_error(self, message):
        QMessageBox.critical(self, "Pipeline Error", message)

    def pipeline_finished(self, code):
        self.run_btn.setEnabled(True)

        if self.current_input:
            track_name = Path(self.current_input).stem
            output_dir = Path("outputs/rendered") / track_name
            final_mix = output_dir / f"{track_name}_enhanced.wav"

            self.current_output_dir = str(output_dir.resolve())
            self.current_output_mix = str(final_mix.resolve())
            self.current_preview_mix = self.current_output_mix

            self.output_path_label.setText(self.current_preview_mix)

            if final_mix.exists():
                self.stereo_scope.load_file(str(final_mix))
                self.open_mix_btn.setEnabled(True)
                self.open_folder_btn.setEnabled(True)
                self.play_pause_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.width_trim_slider.setValue(0)

        if code == 0:
            QMessageBox.information(self, "Done", "Spatialisation complete.")
        else:
            QMessageBox.warning(self, "Finished with errors", f"Exit code: {code}")

    def apply_width_trim_preview(self):
        if not self.current_output_mix or not Path(self.current_output_mix).exists():
            return

        trim_value = self.width_trim_slider.value()

        if trim_value == 0:
            self.current_preview_mix = self.current_output_mix
            self.output_path_label.setText(self.current_preview_mix)
            self.stereo_scope.load_file(self.current_preview_mix)
            return

        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / "isa_width_preview.wav"

        apply_width_trim_to_file(self.current_output_mix, str(temp_path), trim_value)

        self.current_preview_mix = str(temp_path)
        self.output_path_label.setText(self.current_preview_mix)
        self.stereo_scope.load_file(self.current_preview_mix)

        # If currently playing, restart playback on preview file
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(self.current_preview_mix))
            self.player.play()
            self.stereo_scope.start()

    def toggle_play_pause(self):
        if not self.current_preview_mix or not Path(self.current_preview_mix).exists():
            return

        from PySide6.QtMultimedia import QMediaPlayer

        current_source = self.player.source().toLocalFile()
        if current_source != self.current_preview_mix:
            self.player.setSource(QUrl.fromLocalFile(self.current_preview_mix))

        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.stereo_scope.pause()
        else:
            self.player.play()
            self.stereo_scope.start()

    def stop_playback(self):
        self.player.stop()
        self.stereo_scope.stop()

    def on_position_changed(self, position_ms):
        self.stereo_scope.set_playhead(position_ms / 1000.0)

    def on_playback_state_changed(self, state):
        from PySide6.QtMultimedia import QMediaPlayer

        if state == QMediaPlayer.PlayingState:
            self.play_pause_btn.setText("Pause")
        else:
            self.play_pause_btn.setText("Play")

    def open_final_mix(self):
        if self.current_output_mix and Path(self.current_output_mix).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_output_mix))

    def open_output_folder(self):
        if self.current_output_dir and Path(self.current_output_dir).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.current_output_dir))