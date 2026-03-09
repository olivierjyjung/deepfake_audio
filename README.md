# Visual Priming & Deepfake Audio Perception Study

**Unexpected Finding: Visual priming *reduces* false memory for deepfake audio**

## Key Result

| Condition | False Memory Rate | p-value |
|-----------|-------------------|---------|
| Treatment (Idol Priming) | 7.1% | |
| Neutral (Control) | 17.8% | **.030*** |

Visual priming with idol-related content significantly **reduced** false memory rates, contrary to predictions from traditional priming theory.

## Project Structure

```
audio_ai/
├── README.md                          # This file
├── experiment_overview.md             # Full experiment documentation
│
├── Study Experimental Plan.pdf        # Original protocol
├── stimuli.xlsx                       # Group × stimuli assignment
├── participant_group for Audio Task.csv
├── Group #1-8 (Responses).xlsx        # Survey data (Groups 1-8)
├── Group #9-16 (Responses).xlsx       # Survey data (Groups 9-16)
├── Errors from Audio task.xlsx        # Error log
│
├── Final Vids/                        # Video stimuli (30 MB)
├── gdrive_download/                   # EEG data (2.4 GB)
│
├── scripts/                           # Analysis code
│   ├── analyze_real_data.py           # Main analysis (actual data)
│   ├── behavioral_analysis.py         # General behavioral analysis
│   ├── eeg_analysis.py                # EEG processing pipeline
│   └── requirements.txt               # Python dependencies
│
├── paper/                             # Manuscript
│   └── main.tex                       # LaTeX paper with results
│
└── analysis_results/                  # Output (generated)
    ├── processed_real_data.csv
    ├── real_data_analysis_report.txt
    └── figures/
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r scripts/requirements.txt
```

### 2. Run Analysis

```bash
python3 scripts/analyze_real_data.py
```

### 3. Compile Paper

```bash
cd paper
pdflatex main.tex
```

## Experimental Design

### Conditions
- **Treatment (Groups 1-8):** Idol-related visual cues (Hanni/Jennie)
- **Neutral (Groups 9-16):** Unrelated visual content

### Audio Stimuli
- **Real:** "Best Part" by Hanni/Jennie
- **AI-generated:** "Tango" with voice conversion

### Measures
1. False memory ("Have you seen this idol perform this song?")
2. Audio source judgment (AI vs Real)
3. Confidence ratings
4. Perceptual factors (audio quality, breathing, intonation, etc.)

## Main Findings

### 1. Protective Priming Effect
Visual priming **reduced** false memory (7.1% vs 17.8%, p = .030)

### 2. AI Detection
- Treatment: 47.3% correctly identified AI
- Neutral: 42.2% correctly identified AI

### 3. Key Perceptual Cues
1. Audio quality (M = 3.87)
2. Breathing naturalness (M = 3.65)
3. Intonation (M = 3.23)

## Theoretical Interpretation

Three possible explanations for the protective effect:

1. **Reference Standard Hypothesis:** Priming established a mental baseline for authentic content
2. **Enhanced Source Monitoring:** Priming activated analytical processing
3. **Expectation Violation Detection:** Mismatch detection between primed expectations and actual audio

## Paper

**Title:** "Visual Priming Reduces False Memory for Deepfake Audio: An Unexpected Protective Effect of Contextual Cues"

**Abstract:** Contrary to predictions from priming and false memory literature, exposure to idol-related visual content significantly reduced false memory rates for deepfake audio. These findings suggest that contextual priming may serve a protective rather than vulnerability-inducing function in synthetic media evaluation.

## Data

- **Participants:** 54 (216 observations)
- **EEG Recordings:** 56 participants, XDF format
- **Collection Period:** June-July 2025

---

*Last updated: 2026-03-09*
