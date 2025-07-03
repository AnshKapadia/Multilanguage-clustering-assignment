# Speaker Diarization Clustering

This project performs unsupervised speaker clustering using speech embeddings from the ECAPA-TDNN model. It includes two clustering approaches: Agglomerative Hierarchical Clustering (AHC) and Spectral Clustering (SC). The workflow includes preprocessing raw files, extracting embeddings, and clustering them to group audio segments by speaker.

---

## Dataset Overview

This dataset consists of 144 utterances collected from 36 children. Each child contributed 4 utterances — 2 in English and 2 in Hindi. For each language, the 2 utterances are different paragraphs taken from the same story, ensuring consistency in context and style within each language.

---

## Project Structure

```
├── data/
│   └── (input audio files and RecordingDetails.CSV)
├── .gitignore
├── README.md
├── eval.py
├── main.py
├── model_ahc.py
├── model_sc.py
├── preprocessing.py
├── requirements.txt
├── speaker_embedding.py
```

---

## Setup Instructions

### 1. Create Virtual Environment
```bash
mkdir data
#Paste data in this folder along with RecordingDetails.csv
python -m venv diar_env
source diar_env/bin/activate  # On Windows: diar_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Preprocessing - Rename Files and save as wav

```bash
python preprocessing.py
```
---

### Run Clustering

```bash
python main.py --project_dir '<path_to_project>'  --n_speakers 36 --clust_name 'sc'/'ahc' --force_emb True/False
```

---

## Results (Sample)

| Clustering Method| Accuracy (%) |
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
