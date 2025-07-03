# Speaker Diarization Clustering

This project performs unsupervised speaker clustering using speech embeddings from the ECAPA-TDNN model. It includes two clustering approaches: Agglomerative Hierarchical Clustering (AHC) and Spectral Clustering (SC). The workflow includes preprocessing raw files, extracting embeddings, and clustering them to group audio segments by speaker.

---

## Project Structure

```
├── preprocessing/
│   ├── convert_to_wav.py
│   └── rename_files.py
├── model_ahc/
│   └── agglomerative_clustering.py
├── model_sc/
│   └── spectral_clustering.py
├── utils/
│   └── embedding_extraction.py
├── data/
│   └── (input and processed audio files)
├── requirements.txt
├── README.md
```

---

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv diar_env
source diar_env/bin/activate  # On Windows: diar_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Preprocessing

### Step 1: Rename Files
```bash
python preprocessing/rename_files.py --input_dir path/to/input_audio
```

### Step 2: Convert to WAV
```bash
python preprocessing/convert_to_wav.py --input_dir path/to/input_audio --output_dir data/wav
```

---

## Run Clustering

### 1. Agglomerative Clustering
```bash
python model_ahc/agglomerative_clustering.py --input_dir data/wav --num_speakers 3
```

### 2. Spectral Clustering
```bash
python model_sc/spectral_clustering.py --input_dir data/wav --num_speakers 3 --epochs 1000
```

Adjust `--num_speakers` and `--epochs` as needed.

---

## Results (Sample)

| Clustering Method | Accuracy (%) |
|------------------|--------------|
| AHC              | 82.64        |
| SC (avg)         | 88.83        |
| SC (max)         | 93.06        |

---

## Notes
- Embeddings are extracted using ECAPA-TDNN.
- Cosine distance is used for affinity calculations.
- Spectral clustering outperforms AHC in most cases.
- Results may vary based on speaker diversity and utterance duration.
