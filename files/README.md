# 🛡️ DeepGuard — AI-Generated Image Detector

> A Python-based cybersecurity tool for detecting AI-generated and deepfake images using multi-signal frequency and texture analysis.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-9%20passed-brightgreen)
![Domain](https://img.shields.io/badge/Domain-AI%20Security-red)

---

## 🎯 Overview

As AI-generated imagery becomes indistinguishable from real photos, detecting synthetic content is a critical cybersecurity challenge. DeepGuard tackles this using **five independent detection signals** derived from signal processing and computer vision — no large model downloads required, runs fully offline.

**Use cases:**
- Detecting AI-generated profile pictures (social engineering / fraud)
- Verifying authenticity of digital evidence
- Content moderation pipelines
- Security research and red team operations

---

## 🔬 Detection Approach

DeepGuard implements a **multi-signal ensemble** approach. Each signal targets a different artifact class produced by GAN/Diffusion model generation:

| Signal | What it detects | Weight |
|---|---|---|
| **Frequency Anomaly** | Unnatural DCT spectrum — AI images lack the 1/f "pink noise" characteristic of real cameras | 30% |
| **Noise Residual** | Spatially uniform synthetic noise vs. structured camera sensor noise (PRNU) | 25% |
| **Texture Analysis** | LBP histogram entropy — AI images produce unnaturally smooth or repetitive textures | 20% |
| **Channel Mismatch** | R/G/B correlation deviation — AI generation paths can introduce subtle channel inconsistencies | 15% |
| **Edge Artifact** | Periodic edge patterns from latent space upsampling in Diffusion/GAN models | 10% |

A weighted ensemble produces a final **confidence score (0.0–1.0)** and risk classification.

---

## 🚀 Installation

```bash
git clone https://github.com/yourhandle/deepguard.git
cd deepguard
pip install -r requirements.txt
pip install -e .
```

**Dependencies:** `numpy`, `Pillow`, `opencv-python` — no GPU required.

---

## 💻 Usage

### Single image scan

```bash
deepguard scan photo.jpg
```

```
  ██████╗ ███████╗███████╗██████╗  ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
  ...

────────────────────────────────────────────────────────────
  File:     photo.jpg
  Verdict:  🤖 AI-GENERATED
  Score:    [██████████████░░░░░░░░░░░░░░░░] 46.3%
  Risk:     MEDIUM
────────────────────────────────────────────────────────────
```

### Verbose mode (per-signal breakdown)

```bash
deepguard scan photo.jpg --verbose
```

### Save anomaly heatmap

```bash
deepguard scan photo.jpg --heatmap
# → saves photo_heatmap.png highlighting suspicious regions
```

### Batch scan a folder

```bash
deepguard scan images/ --batch --verbose
```

### JSON output (for pipeline integration)

```bash
deepguard scan photo.jpg --json
```
```json
{
  "filepath": "photo.jpg",
  "is_ai_generated": true,
  "confidence": 0.634,
  "risk_level": "HIGH",
  "signals": {
    "frequency_anomaly": 0.81,
    "noise_residual": 0.72,
    "texture_score": 0.44,
    "channel_mismatch": 0.51,
    "edge_artifact": 0.38
  }
}
```

### Adjust detection threshold

```bash
deepguard scan photo.jpg --threshold 0.65   # stricter (fewer false positives)
deepguard scan photo.jpg --threshold 0.35   # looser  (fewer false negatives)
```

---

## 🐍 Python API

```python
from deepguard import DeepfakeDetector

detector = DeepfakeDetector(threshold=0.5)
result = detector.detect("suspicious_image.jpg", save_heatmap=True)

print(result.confidence)      # 0.734
print(result.risk_level)      # "HIGH"
print(result.is_ai_generated) # True
print(result.signals)         # {'frequency_anomaly': 0.81, ...}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

```
tests/test_detector.py::TestDetector::test_returns_detection_result PASSED
tests/test_detector.py::TestDetector::test_confidence_in_range      PASSED
tests/test_detector.py::TestDetector::test_risk_level_values        PASSED
tests/test_detector.py::TestDetector::test_all_signals_present      PASSED
tests/test_detector.py::TestDetector::test_all_signals_in_range     PASSED
tests/test_detector.py::TestDetector::test_file_not_found_raises    PASSED
tests/test_detector.py::TestDetector::test_heatmap_saved            PASSED
tests/test_detector.py::TestDetector::test_custom_threshold         PASSED
tests/test_detector.py::TestDetector::test_str_representation       PASSED
9 passed in 1.28s
```

---

## 📁 Project Structure

```
deepguard/
├── deepguard/
│   ├── __init__.py       # Public API
│   ├── detector.py       # Core detection engine (5 signals)
│   └── cli.py            # Command-line interface
├── tests/
│   └── test_detector.py  # Full test suite (9 tests)
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔮 Roadmap

- [ ] Integrate pretrained CNN classifier (CNNDetection / UniversalFakeDetect weights)
- [ ] Video frame analysis (deepfake video detection)
- [ ] REST API wrapper (Flask/FastAPI)
- [ ] Docker container for deployment
- [ ] Benchmark against DFDC, FaceForensics++ datasets

---

## 🔗 Background & References

- [CNNDetection — Wang et al. 2020](https://github.com/peterwang512/CNNDetection)
- [UniversalFakeDetect — Ojha et al. 2023](https://github.com/Yuheng-Li/UniversalFakeDetect)
- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
- [NIST Media Forensics Challenge](https://www.nist.gov/programs-projects/media-forensics)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of a cybersecurity AI research portfolio. Targeting ICT Security Specialist roles in NZ/AU tech sector.*
