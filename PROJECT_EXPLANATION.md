# RNN Poetry Generator - Explainer & Execution Guide

This document is designed to help you **understand the project**, **explain it to others (interviewers, classmates, or colleagues)**, and **run the code** step-by-step in any terminal shell.

---

## 1. How to Explain This Project to Others

If someone asks you, *"What is this project?"*, here is a simple and clear way to explain it:

### The Pitch (The 30-Second Summary)
> "This project is a character-level text generator that uses a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. It reads a corpus of text (like Shakespeare's plays), learns the statistical probability of which character comes next after a sequence, and then writes original, Shakespeare-style poetry by predicting characters recursively. I structured it into a professional, modular CLI tool complete with error handling and file logging."

### Key Concepts to Describe

1. **Character-Level Prediction**: 
   - Instead of predicting whole words, the model predicts the very next *character* (letters, spaces, punctuation). This means it learns not only vocabulary but also how to spell words, where to put spaces, and how to format play scripts.
2. **LSTM (Long Short-Term Memory)**:
   - Standard neural networks have no memory. An LSTM is a type of RNN that can remember long-term patterns in sequences (like remembering to close a quotation mark or maintaining the format of character dialogues).
3. **One-Hot Encoding**:
   - Computers don't understand text. We extract every unique character in the text (e.g., `a, b, c, !, ?, \n`). If there are 39 unique characters, we represent each character as a list of 39 numbers (zeros and a single one at the character's index).
4. **Temperature**:
   - This controls how "creative" or "random" the output is. 
   - **Low Temperature (e.g., 0.2)**: The model is conservative and chooses the highest probability character, leading to very safe, repetitive, and realistic text.
   - **High Temperature (e.g., 1.0)**: The model takes risks by selecting less likely characters. This makes the text creative and poetic but increases the chance of spelling errors.

---

## 2. File Structure: What does each file do?

- **`main.py`**: The main control room. It parses your command-line arguments and calls either the training pipeline or the text generator.
- **`poetry_gen/dataset.py`**: The data prep factory. It downloads the text, maps characters to numbers, and splits the text into sliding window chunks (e.g., sequences of 40 characters) for the model to train on.
- **`poetry_gen/model.py`**: The architect. It builds the LSTM model using Keras and handles saving and loading the files.
- **`poetry_gen/generator.py`**: The writer. It takes a "seed" sentence, passes it to the model to predict the next character, updates the seed, and repeats this recursively to write poetry.
- **`poetry_gen/logger.py`**: The reporter. It prints friendly progress reports on the screen and logs detailed tracebacks in `poetry_generation.log`.
- **`poetry_gen/exceptions.py`**: The safety net. Custom error categories that capture network issues or missing files without crashing the terminal with messy tracebacks.

---

## 3. How to Run This Project

You can run this project using **PowerShell**, **Command Prompt (CMD)**, or **Git Bash** on Windows. 

Here are the step-by-step guides for two scenarios:
- **Scenario A**: Starting from scratch (not cloned yet).
- **Scenario B**: Already cloned (you are already inside the folder).

---

### Scenario A: Starting From Scratch (Step-by-Step)

If you are on a new machine or haven't cloned the repository yet:

#### Step 1: Open your terminal
- Search for **PowerShell** or **Command Prompt (CMD)** in Windows and open it.

#### Step 2: Navigate to where you want to keep the code
Use the `cd` (change directory) command to go to your projects folder. For example:
```bash
cd D:\Code_Files\Projects
```

#### Step 3: Clone the project from GitHub
```bash
git clone https://github.com/KGupta171025/RNN_Poetry_Generation.git
cd RNN_Poetry_Generation
```

#### Step 4: Create a virtual environment (venv)
A virtual environment keeps the dependencies isolated.
```bash
python -m venv venv
```

#### Step 5: Activate the virtual environment
Depending on what shell you opened, run the corresponding command:

- **In PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Note: If PowerShell complains about Execution Policies, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, then run the activation script.)*

- **In Command Prompt (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

- **In Git Bash:**
  ```bash
  source venv/Scripts/activate
  ```

Once activated, you will see `(venv)` at the beginning of your command line.

#### Step 6: Install the requirements
```bash
pip install -r requirements.txt
```

#### Step 7: Run text generation!
```bash
python main.py generate --model-path textgenerator.keras --length 300 --temperature 0.5
```

---

### Scenario B: Running the Project (Already Cloned)

If the files are already on your computer and you want to run them again:

#### Step 1: Open your terminal
Open **PowerShell** or **Command Prompt (CMD)**.

#### Step 2: Navigate directly to the project folder
```bash
cd D:\Code_Files\Projects\RNN_Poetry_Generation
```

#### Step 3: Activate the virtual environment
- **In PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **In CMD:**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

#### Step 4: Run CLI commands

**To generate poetry (fast):**
```bash
python main.py generate --model-path textgenerator.keras --length 300 --temperature 0.5
```

*Try changing the temperature (e.g., `--temperature 0.2` for safe text, or `--temperature 0.8` for more creative text).*

**To provide your own seed prompt:**
```bash
python main.py generate --model-path textgenerator.keras --length 200 --temperature 0.6 --seed "shall i compare thee to a summer's day? thou art"
```

**To retrain the model (takes some minutes):**
```bash
python main.py train --epochs 4 --batch-size 256 --model-path my_new_model.keras
```
