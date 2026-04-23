# Intelligent Spatial Audio

An end-to-end stem-based stereo spatial enhancement system for music production.

This project takes a stereo music file, separates it into stems, extracts interpretable audio features from each stem, generates spatial placement decisions using a hybrid rule-based and machine-learning approach, optionally refines those decisions with evolutionary optimisation, and renders an enhanced stereo mix. The final version focuses on **stereo spatial enhancement** rather than binaural/HRTF rendering.

## Project Summary

The system was developed as a final-year Computer Science project exploring whether spatial mix decisions can be made more consistently and transparently using a structured audio pipeline. Instead of learning a full mixing system end-to-end, the project combines:

- pre-trained source separation
- per-stem feature extraction
- explainable rule-based spatial logic
- lightweight machine-learning refinement
- optional DEAP-based optimisation
- stereo rendering with safe low-end handling
- a desktop GUI for running the pipeline and previewing results

The output is an enhanced stereo mix designed to improve width, clarity, and spatial organisation while preserving stability and mono compatibility.

## Current Pipeline

The implemented pipeline is:

1. **Audio stem separation** using pre-trained Demucs models
2. **Per-stem feature extraction** from the separated stems
3. **Spatial decision generation** using rule-based logic
4. **Machine-learning refinement** of pan, width, and depth values
5. **Optional DEAP optimisation** of spatial parameters
6. **Stereo rendering** of the processed stems into a final enhanced mix
7. **GUI-based preview and playback** with waveform views, stereo scope, and width trim control

## Main Features

- Accepts a stereo music file as input
- Supports multiple Demucs separation models
- Extracts interpretable per-stem features such as:
  - RMS energy
  - spectral centroid
  - zero-crossing rate
  - spectral rolloff
  - low-frequency energy ratio
- Generates spatial parameters for each stem:
  - pan
  - stereo width
  - depth
- Blends rule-based decisions with ML-predicted refinements
- Includes optional DEAP optimisation for parameter refinement
- Applies stereo-safe processing with low-frequency narrowing for mono compatibility
- Provides a GUI with:
  - input file selection
  - model selection
  - ML blend control
  - optional optimisation toggle
  - waveform displays
  - stereo scope visualisation
  - playback controls
  - width trim preview

## Repository Structure

```text
Intelligent-Spatial-Audio-main/
├── data/
│   ├── raw/                  # input tracks
│   ├── processed/            # converted/normalised audio
│   └── training/             # labelled ML training data
├── gui/
│   ├── app.py                # GUI entry point
│   ├── main_window.py        # main desktop interface
│   ├── workers.py            # background pipeline worker
│   └── widgets/
│       ├── stereo_scope.py   # stereo scope visualisation
│       └── waveform_view.py  # waveform display
├── models/
│   └── spatial_refiner.joblib
├── outputs/
│   ├── features/
│   ├── rendered/
│   ├── spatial/
│   └── stems/
├── src/
│   ├── separate_demucs.py
│   ├── extract_features.py
│   ├── spatial_decision.py
│   ├── deap_optimise.py
│   ├── render_spatial_mix.py
│   ├── train_spatial_model.py
│   └── run_full_pipeline.py
├── requirements.txt
└── README.md
```

## Core Scripts

### `src/separate_demucs.py`
Separates an input track into stems using a pre-trained Demucs model.

### `src/extract_features.py`
Computes per-stem audio descriptors used for spatial decision-making.

### `src/spatial_decision.py`
Applies rule-based logic and optional ML refinement to produce pan, width, and depth values.

### `src/deap_optimise.py`
Optionally refines the generated spatial parameters using DEAP.

### `src/render_spatial_mix.py`
Processes the stems and renders the final enhanced stereo mix.

### `src/train_spatial_model.py`
Trains the lightweight ML model used to refine spatial parameters.

### `src/run_full_pipeline.py`
Runs the full pipeline from input track to rendered output.

### `gui/app.py`
Launches the desktop interface for running the pipeline visually.

## Installation

Create and activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Requirements

The project depends primarily on:

- Python 3.x
- Demucs
- librosa
- NumPy
- pandas
- soundfile
- scipy
- scikit-learn
- joblib
- DEAP
- PySide6
- pyqtgraph
- ffmpeg

## Usage

### Run the full pipeline

```bash
python src/run_full_pipeline.py --infile data/raw/your_track.wav --model mdx_extra --device cuda --ml_model models/spatial_refiner.joblib --ml_blend 0.2
```

### Run the full pipeline with DEAP optimisation

```bash
python src/run_full_pipeline.py --infile data/raw/your_track.wav --model mdx_extra --device cuda --ml_model models/spatial_refiner.joblib --ml_blend 0.2 --use_deap
```

### Launch the GUI

```bash
python gui/app.py
```

## Output Files

Typical outputs are written to:

- `outputs/stems/` for separated stems
- `outputs/features/` for extracted feature CSV files
- `outputs/spatial/` for generated spatial parameter CSV files
- `outputs/rendered/` for the final enhanced stereo mix

## Design Notes

This version of the project no longer centres on binaural rendering. Earlier binaural/HRTF exploration was not retained as the main rendering path because the final system moved toward a more practical **stereo mixing assistant** for conventional playback contexts.

The emphasis of the final implementation is therefore on:

- explainable spatial decisions
- practical stereo enhancement
- audible width and clarity improvement
- compatibility with normal stereo listening systems

## Known Limitations

- Separation quality is limited by the chosen Demucs model and may introduce artefacts
- The feature set is relatively lightweight and stem-level rather than deeply time-varying
- Depth processing is simpler than the pan and width stages
- Results depend on the quality and balance of the separated stems
- Already well-spatialised commercial mixes may benefit less than flatter or narrower inputs

## Possible Future Improvements

- time-varying spatial decisions rather than stem-level averages
- band-dependent spatial control for lows, mids, and highs
- stronger depth modelling
- transient-aware and harmonic-aware processing
- improved training targets for the ML refinement stage
- broader listening evaluation with more participants and structured scoring

## Academic Context

This repository contains the implementation for a final-year Computer Science dissertation project on intelligent music stem spatialisation. The work focuses on combining existing signal-processing and machine-learning methods into an interpretable system for stereo spatial enhancement.

## Status

This repository reflects the completed project version provided in the uploaded archive. If your GitHub copy came from this same archive, then yes, it looks like the README mainly needed updating to match the final stereo-focused implementation.
