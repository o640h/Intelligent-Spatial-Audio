# Intelligent Music Stem Spatialisation

This repository contains the implementation for the final‑year Computer
Science project **"Intelligent Music Stem Spatialisation"** at the
University of York.

The goal of the project is to build an automated system that can analyse
music stems and place them intelligently in stereo space using
rule‑based spatialisation informed by audio features.

## Project Pipeline

The system is designed as a modular pipeline consisting of five stages:

1.  Audio stem separation\
2.  Per‑stem feature extraction\
3.  Intelligent spatial decision logic\
4.  Binaural rendering using HRTFs\
5.  Evaluation of spatial quality

The current implementation focuses on **Stage 1: Stem Separation**.

------------------------------------------------------------------------

# Stem Separation

Audio source separation is performed using **pre‑trained Demucs
models**, which divide a stereo music track into four stems:

-   Vocals\
-   Drums\
-   Bass\
-   Other / instruments

GPU acceleration (CUDA) is supported and significantly reduces
processing time when available.

### Tested Models

During development several Demucs models were tested:

-   `htdemucs` -- good balance of speed and quality\
-   `htdemucs_ft` -- fine‑tuned version with improved separation\
-   `mdx_extra` -- often produces cleaner stems with fewer artefacts

The model can be selected via command‑line arguments.

------------------------------------------------------------------------

# Repository Structure

    src/
        separate_demucs.py     # Stem separation script

    data/
        raw/                   # Input audio files
        processed/             # Optional preprocessing outputs

    outputs/
        stems/                 # Generated stems

    requirements.txt           # Python dependencies

------------------------------------------------------------------------

# Setup

Create a virtual environment and install dependencies:

``` bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The project requires **PyTorch** and optionally **CUDA** for GPU
acceleration.

------------------------------------------------------------------------

# Running Stem Separation

Place an audio file inside:

    data/raw/

Then run:

``` bash
python src/separate_demucs.py --infile data/raw/test.wav --model mdx_extra --device cuda
```

Example output:

    outputs/stems/mdx_extra/test/
        vocals.wav
        drums.wav
        bass.wav
        other.wav

------------------------------------------------------------------------

# Notes

-   Input audio can be WAV or MP3.
-   Separation quality depends on both the model and the quality of the
    input audio.
-   Minor artefacts (e.g., high‑frequency buzzing) are a known
    limitation of modern source separation models.

------------------------------------------------------------------------

# Project Context

This repository forms part of the undergraduate dissertation:

**Oliver Heron**\
BSc Computer Science\
University of York

Supervisor: **Simon O'Keefe**
