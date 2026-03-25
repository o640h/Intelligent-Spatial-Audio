import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot


class PipelineWorker(QObject):
    finished = Signal(int)
    log = Signal(str)
    error = Signal(str)

    def __init__(self, infile, model, ml_model, ml_blend, use_deap):
        super().__init__()
        self.infile = infile
        self.model = model
        self.ml_model = ml_model
        self.ml_blend = ml_blend
        self.use_deap = use_deap

    @Slot()
    def run(self):
        try:
            cmd = [
                sys.executable,
                "src/run_full_pipeline.py",
                "--infile",
                self.infile,
                "--model",
                self.model,
                "--device",
                "cuda",
                "--ml_model",
                self.ml_model,
                "--ml_blend",
                str(self.ml_blend),
            ]

            if self.use_deap:
                cmd.append("--use_deap")

            self.log.emit("Running pipeline...\n")
            self.log.emit(" ".join(cmd) + "\n\n")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                self.log.emit(line)

            process.wait()

            if process.returncode != 0:
                self.error.emit(f"Pipeline failed with exit code {process.returncode}")
                self.finished.emit(process.returncode)
                return

            self.finished.emit(0)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(1)