# RNN Poetry Generation

A professional, modular, and production-ready Recurrent Neural Network (RNN) designed to generate Shakespeare-like poetry. This project uses a character-level Long Short-Term Memory (LSTM) network built with **TensorFlow** and **Keras** to model the structure of text and predict sequential character outputs.

Restructured with robust logging, custom exception handling, and a command-line interface (CLI) that separates training and generation workflows.

---

## Features

- **Unified CLI Orchestration**: Easily switch between model training (`train`) and text generation (`generate`) subcommands.
- **Robust Exception Handling**: User-friendly console error reporting with complete tracebacks logged behind the scenes.
- **Centralized Dual Logging**: Steps and debug info are printed to the console and saved to `poetry_generation.log`.
- **Modern Packaging**: Supports installing locally in editable mode via PEP 517 (`pyproject.toml`).
- **Flexible Parameters**: Tune epochs, batch size, learning rates, sequence lengths, temperatures, and custom seed texts directly from the command line.

---

## Repository Structure

```text
RNN_Poetry_Generation/
├── .gitignore               # Excludes python cache, models, datasets, and logs
├── pyproject.toml           # Package distribution and installation config
├── requirements.txt         # Pinned packages for deployment
├── README.md                # This documentation
├── main.py                  # Primary CLI entrypoint (routes to train/generate commands)
└── poetry_gen/              # Core logic package
    ├── __init__.py
    ├── dataset.py           # Preprocessing, data-download, and OHE vectors
    ├── exceptions.py        # Custom exceptions hierarchy
    ├── generator.py         # Softmax temperature sampling and generation loop
    ├── logger.py            # Console/file logger setup
    └── model.py             # Keras LSTM architecture build, save, and load helpers
```

---

## Setup & Installation

Follow these steps to run this project on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/KGupta171025/RNN_Poetry_Generation.git
cd RNN_Poetry_Generation
```

### 2. Set Up a Virtual Environment
It is recommended to run this project inside a clean virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
You can install dependencies using either the requirements file or directly as an editable package:

**Option A: Install via requirements file (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Install package in editable mode**
```bash
pip install -e .
```

---

## Usage Guide

Run the main file with `--help` to view all available commands:
```bash
python main.py --help
```

### 1. Training a Model
To train the model on the Shakespeare corpus:
```bash
python main.py train --epochs 4 --batch-size 256 --model-path textgenerator.keras
```

#### Available training arguments:
- `--epochs`: Number of epochs to train (default: `4`).
- `--batch-size`: Batch size for training (default: `256`).
- `--lr`: Learning rate for the RMSprop optimizer (default: `0.01`).
- `--model-path`: Path where the trained model should be saved (default: `textgenerator.keras`).
- `--seq-length`: Input sequence length in characters (default: `40`).
- `--step-size`: Sliding window step size for text chunking (default: `3`).
- `--corpus-path`: Path to a local text corpus. If omitted, downloads the Shakespeare dataset automatically.

---

### 2. Generating Poetry
To load a trained model and generate text:
```bash
python main.py generate --model-path textgenerator.keras --length 300 --temperature 0.5
```

#### Available generation arguments:
- `--model-path`: Path to the trained `.keras` or `.h5` model file.
- `--length`: Total number of characters to generate (default: `300`).
- `--temperature`: Creativity/sampling temperature (default: `0.5`). 
  - *Lower (e.g. 0.2)*: More deterministic, repetitive, and close to the source corpus.
  - *Higher (e.g. 1.0)*: Highly creative, random, but prone to grammatical mistakes or spelling slips.
- `--seed`: Optional seed text to prompt the model. Must be at least 40 characters (shorter seeds will be automatically padded). If not provided, a random chunk of the Shakespeare corpus will be selected.

#### Example with seed text:
```bash
python main.py generate --model-path textgenerator.keras --temperature 0.6 --seed "to be, or not to be, that is the question"
```

---

## Logging & Troubleshooting

All operation steps and debug logs are saved to a file named `poetry_generation.log` in the root directory.

- If the application crashes or behaves unexpectedly, check `poetry_generation.log` for the full traceback details.
- To enable verbose debug-level prints directly on the console, run any subcommand with the `--verbose` flag:
  ```bash
  python main.py --verbose generate --model-path textgenerator.keras
  ```
