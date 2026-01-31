# Authentication System (Face + Voice + Optional STT) — Privacy-Safe Repo

A practical authentication workflow that can combine:
- **Face verification** (InsightFace embeddings)
- **Speaker verification** (SpeechBrain speaker embeddings)
- **Optional** offline speech-to-text (STT) for capturing a name using **Vosk**

This repository is published in a **privacy-safe** way: the project structure is complete, but any sensitive datasets/logs/identities were removed or replaced with placeholders.

---

## Demo Run (How we run the full system)
We typically run the complete system using:

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.15 --name_seconds 6 --cooldown 10 --pc_ui_lang en
````

### What these flags mean

* `--stt_name`
  Enables offline speech-to-text for capturing/confirming a spoken name.
* `--vosk_model <folder>`
  Path (or folder name) of the Vosk model directory.
* `--name_seconds 6`
  How many seconds to listen for the name.
* `--cooldown 10`
  Cooldown time between attempts (helps avoid repeated triggers).
* `--pc_ui_lang en`
  UI / prompts language on PC side.

---

## Key Features

* ✅ Multi-factor verification (Face + Voice) with fusion
* ✅ Offline STT option (Vosk) for name capture
* ✅ Privacy-safe dataset/DB templates included (no real user data)
* ✅ Prompt assets for PC UI language selection (AR/EN)
* ✅ Modular scripts for building databases and running the pipeline

---

## Privacy & Data Handling Notes (Important)

This repo intentionally excludes any private or identifying data.

**What you will notice (and why it’s normal):**

* `dataset/` exists as **structure only** (no images/audio are shipped).
* `db/teachers.json` and `db/pending.json` are included as **templates/placeholders** to show the expected schema.
* `logs/attempts.jsonl` is **empty**.

> **Why are some files “empty”?**
> Because publishing real images, recordings, teacher identities, or attempt logs would violate privacy.
> This repository is meant to be safe to share while still being understandable and runnable once you provide your own data.

---

## Project Structure (high level)

```text
.
├─ run_system.py              # Main runner (full system)
├─ main.py                    # FastAPI server (if used by your flow)
├─ pc_client.py               # PC-side logic (STT + UI prompts)
├─ verify_fusion.py           # Fusion logic (face + voice)
├─ face_model_insightface.py  # Face embeddings using InsightFace
├─ voice_model.py             # Speaker embeddings using SpeechBrain
├─ db/
│  ├─ teachers.json           # Template (placeholder)
│  └─ pending.json            # Template (placeholder)
├─ dataset/                   # Template only (no private media)
├─ logs/
│  └─ attempts.jsonl          # Empty placeholder
└─ assets/prompts/            # Audio/text prompts (AR/EN)
```

---

## Requirements

* Python 3.9+ (recommended)
* `numpy`, `requests`, `tqdm`
* `opencv-python`
* `torch`, `torchaudio`
* `speechbrain`
* `insightface`
* `fastapi`, `uvicorn` (if you use `main.py`)
* **Optional**: `vosk` (only needed if you use `--stt_name`)

---

## Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies (typical)

```bash
pip install -U pip
pip install numpy requests tqdm opencv-python
pip install torch torchaudio
pip install speechbrain insightface
pip install fastapi uvicorn
# Optional (only if using STT):
pip install vosk
```

> Note: Torch/torchaudio installation can vary by OS/GPU.
> If you face install issues, prefer CPU-only builds or install from the official PyTorch instructions.

---

## Vosk Model (Not Included)

### Why the Vosk model folder is not in this repo

A folder like `vosk-model-small-en-us-0.15` is an external pre-trained model downloaded from a third-party source.
To avoid licensing/ownership issues (and reduce repo size), **the model is not committed** here.

### How to use Vosk with this project

1. Download any compatible Vosk model (English or other language).
2. Place the model folder inside the project directory, for example:

```text
Authentication-System/
└─ vosk-model-small-en-us-0.15/
```

### If your model has a different folder name

No problem — just pass its name/path via `--vosk_model`:

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.22 --name_seconds 6 --cooldown 10 --pc_ui_lang en
```

> Alternatively, you can change the default in the code where the argument is defined
> (search for `--vosk_model` in `run_system.py`).

---

## Running the Full System

From the project root:

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.15 --name_seconds 6 --cooldown 10 --pc_ui_lang en
```

### Example variations

* Run without STT (if supported by your setup):

```bash
python run_system.py --vosk_model vosk-model-small-en-us-0.15 --cooldown 10 --pc_ui_lang en
```

* Switch UI language (if you have prompts for Arabic):

```bash
python run_system.py --stt_name --vosk_model vosk-model-small-en-us-0.15 --name_seconds 6 --cooldown 10 --pc_ui_lang ar
```

---

## What you need to provide (because this repo is privacy-safe)

To actually enroll/verify identities you must provide your own data:

* Add your own images/audio into the expected `dataset/` structure.
* Fill your local teacher/user list (CSV/JSON templates) with your own IDs.
* Generate embeddings/databases according to the scripts you use in your workflow.

---

## Troubleshooting

* **The system can’t find the Vosk model**
  Make sure the folder exists and the path matches the `--vosk_model` value.
* **Empty dataset / missing identities**
  This repo ships without private media by design. Add your own data locally.
* **Repeated triggers / too many attempts**
  Increase `--cooldown` to reduce back-to-back attempts.

---

## Third-Party Models & Licensing

This repository contains project code and placeholder templates only.
Third-party models (e.g., Vosk) are governed by their original licenses and must be obtained separately.

---

## Contact

If you build on this project, feel free to open an issue or submit a pull request.
