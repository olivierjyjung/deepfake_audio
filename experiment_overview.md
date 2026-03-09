# Visual Priming and Deepfake Audio Perception Study
## Experiment Overview

---

## 1. Research Objectives

**Main Research Question:** Does visual priming (idol-related video content) affect false memory formation and deepfake audio detection?

### Theoretical Background
- **Priming Theory:** Visual cues activate mental schemas that influence subsequent information processing
- **False Memory & Source Monitoring:** Potential for confusing imagined exposure with actual memory
- **Unexpected Finding:** Priming appears to have a *protective* rather than vulnerability-inducing effect

---

## 2. Experimental Design

### Participants
- **N = 54** participants (216 total observations)
- Female undergraduates at Ewha Womans University
- Quiet laboratory environment
- Duration: ~20 minutes per participant

### Conditions (Between-Subjects Design)
| Condition | Groups | Visual Cue | Description |
|-----------|--------|------------|-------------|
| **Treatment** | 1-8 | Idol-related (Hanni/Jennie) | Visual priming present |
| **Neutral** | 9-16 | Unrelated content | Control (no priming) |

### Stimuli
- **Idols:** Hanni (NewJeans), Jennie (BLACKPINK) - high brand recognition in Korea
- **Audio Types:**
  - Real: "Best Part" by Hanni/Jennie (authentic recordings)
  - AI-generated: "Tango" with voice conversion technology
- **Clip Duration:** ~30 seconds each

---

## 3. Procedure

```
Consent → Demographics → Hearing Check → Random Assignment (Groups 1-16)
    → Visual Cue Exposure → Audio Playback (4 clips) → Survey
```

### Survey Items (11 questions)
1. Have you seen this idol perform this song in a video? (Yes/No)
2. Confidence in Q1 response (1-5 scale)
3. How was the audio created? (6 options)
4. Confidence in Q3 response (1-5 scale)
5. Factors influencing judgment (pronunciation, intonation, emotion, quality, rhythm, breathing)
6. Overall likeability (1-5 scale)
7. Why did you think it was this idol?
8. Listening frequency (1-5 scale)
9. Voice familiarity (1-5 scale)
10. Where have you encountered this idol's voice?
11. Other factors (open-ended)

---

## 4. Key Results

### False Memory (Main Finding)
| Condition | False Memory Rate | n |
|-----------|-------------------|---|
| **Treatment** (Priming) | **7.1%** | 112 |
| **Neutral** (Control) | **17.8%** | 101 |

**Statistics:** χ²(1) = 4.70, p = .030, Cramér's V = .149

**Interpretation:** Visual priming *reduced* false memory rates (opposite to hypothesis)

### Audio Source Judgment
| Condition | AI | Other/Cover | Real |
|-----------|-----|-------------|------|
| Treatment | 47.3% | 19.6% | 33.0% |
| Neutral | 42.2% | 26.5% | 31.4% |

### Factors Influencing Judgment (Importance Ranking)
1. **Audio Quality** (M = 3.87)
2. **Breathing Naturalness** (M = 3.65)
3. Intonation (M = 3.23)
4. Pronunciation (M = 3.20)
5. Emotional Expression (M = 3.00)
6. Rhythm (M = 2.89)

### Confidence Ratings
- False Memory Confidence: M = 3.70, SD = 1.11
- Audio Judgment Confidence: M = 2.92, SD = 0.92
- No significant difference between conditions (p = .40)

### Familiarity Effect
- Voice familiarity × False memory: r = -.098, p = .157 (not significant)

---

## 5. Theoretical Implications

### Why Did Priming *Reduce* False Memory?

Three possible explanations:

1. **Reference Standard Hypothesis**
   - Idol-related visual cues established a mental baseline for "authentic" content
   - Participants could better detect when audio deviated from expectations

2. **Enhanced Source Monitoring**
   - Priming activated analytical processing mode
   - Increased scrutiny of audio authenticity markers

3. **Expectation Violation Detection**
   - Mismatch between primed expectations and actual audio characteristics
   - More likely to recognize discrepancies

---

## 6. Data Structure

### Directory Structure
```
/Users/user/Desktop/audio_ai/
├── README.md                          # Project guide
├── experiment_overview.md             # This file
├── Study Experimental Plan.pdf        # Original protocol
├── stimuli.xlsx                       # Stimuli assignment by group
├── participant_group for Audio Task.csv
├── Group #1-8 (Responses).xlsx        # Survey responses
├── Group #9-16 (Responses).xlsx
├── Errors from Audio task.xlsx
├── Final Vids/                        # Video stimuli (30 MB)
├── gdrive_download/                   # EEG data (2.4 GB)
├── scripts/                           # Analysis scripts
│   ├── analyze_real_data.py
│   ├── behavioral_analysis.py
│   ├── eeg_analysis.py
│   └── requirements.txt
├── paper/                             # Manuscript
│   └── main.tex
└── analysis_results/                  # Output
    ├── processed_real_data.csv
    ├── real_data_analysis_report.txt
    └── figures/
```

### EEG Data Collection
- **Period:** June 25 - July 13, 2025
- **Participants:** 56 (61 recordings including backups)
- **Format:** XDF (Lab Streaming Layer)
- **File Size:** ~61 MB per recording

---

## 7. Publications

### Paper Title
**"Visual Priming Reduces False Memory for Deepfake Audio: An Unexpected Protective Effect of Contextual Cues"**

### Key Contributions
- First evidence of protective priming effect in deepfake detection context
- Challenges conventional assumptions about priming-induced vulnerability
- Practical implications for media literacy interventions

---

*Last Updated: 2026-03-09*
