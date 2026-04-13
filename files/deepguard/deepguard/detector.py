"""
DeepGuard - AI-Generated Image Detector
Core detection engine using frequency analysis + texture features.
No heavy model downloads required - works out of the box.
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


@dataclass
class DetectionResult:
    """Result of a deepfake/AI-generated image detection."""
    filepath: str
    is_ai_generated: bool
    confidence: float          # 0.0 = definitely real, 1.0 = definitely AI
    risk_level: str            # LOW / MEDIUM / HIGH
    signals: dict              # Individual feature scores
    heatmap_path: Optional[str] = None

    def __str__(self):
        verdict = "🤖 AI-GENERATED" if self.is_ai_generated else "✅ LIKELY REAL"
        return (
            f"File     : {self.filepath}\n"
            f"Verdict  : {verdict}\n"
            f"Confidence: {self.confidence:.1%}\n"
            f"Risk     : {self.risk_level}\n"
        )


class DeepfakeDetector:
    """
    Multi-signal AI image detector.

    Detection approach:
    1. Frequency domain analysis (GAN/Diffusion artifacts in DCT spectrum)
    2. Noise residual analysis (camera sensor noise vs synthetic noise)
    3. Texture inconsistency (unnatural smoothness in AI-generated regions)
    4. Color channel correlation (AI models often produce subtle channel mismatches)
    5. Edge artifact detection (upsampling/latent space boundaries)
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Confidence threshold above which image is flagged as AI. Default 0.5.
        """
        self.threshold = threshold

    def detect(self, image_path: str, save_heatmap: bool = False) -> DetectionResult:
        """
        Analyze an image and return a DetectionResult.

        Args:
            image_path: Path to image file (JPG, PNG, WEBP supported)
            save_heatmap: If True, saves a GradCAM-style heatmap alongside the image

        Returns:
            DetectionResult with verdict and individual signal scores
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        signals = {
            "frequency_anomaly": self._frequency_analysis(img_gray),
            "noise_residual":    self._noise_residual_analysis(img_np),
            "texture_score":     self._texture_analysis(img_gray),
            "channel_mismatch":  self._channel_correlation(img_np),
            "edge_artifact":     self._edge_artifact_score(img_gray),
        }

        # Weighted ensemble (tuned empirically)
        weights = {
            "frequency_anomaly": 0.30,
            "noise_residual":    0.25,
            "texture_score":     0.20,
            "channel_mismatch":  0.15,
            "edge_artifact":     0.10,
        }
        confidence = sum(signals[k] * weights[k] for k in weights)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        is_ai = confidence >= self.threshold
        risk_level = self._risk_level(confidence)

        heatmap_path = None
        if save_heatmap:
            heatmap_path = self._generate_heatmap(img_np, signals, path)

        return DetectionResult(
            filepath=str(path),
            is_ai_generated=is_ai,
            confidence=confidence,
            risk_level=risk_level,
            signals=signals,
            heatmap_path=heatmap_path,
        )

    # ------------------------------------------------------------------ #
    #  Detection signals                                                   #
    # ------------------------------------------------------------------ #

    def _frequency_analysis(self, gray: np.ndarray) -> float:
        """
        Analyze DCT frequency spectrum.
        AI-generated images (GANs, Diffusion) often have characteristic
        high-frequency artifacts or unusual spectral distribution.
        """
        # Resize to standard size for consistent FFT
        resized = cv2.resize(gray, (256, 256)).astype(np.float32)

        # 2D DFT
        dft = np.fft.fft2(resized)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.log1p(np.abs(dft_shift))

        # Real camera images have 1/f noise characteristic (pink noise)
        # AI images often deviate from this
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2

        # Compare energy in different frequency bands
        low_freq_mask  = self._radial_mask(h, w, 0, 20)
        mid_freq_mask  = self._radial_mask(h, w, 20, 60)
        high_freq_mask = self._radial_mask(h, w, 60, 128)

        low_energy  = np.mean(magnitude[low_freq_mask])
        mid_energy  = np.mean(magnitude[mid_freq_mask])
        high_energy = np.mean(magnitude[high_freq_mask])

        total = low_energy + mid_energy + high_energy + 1e-8
        # Natural images: high low-freq ratio, steep falloff
        # AI images: flatter spectrum or unusual mid-freq bump
        falloff_ratio = (low_energy - high_energy) / total

        # Lower falloff ratio = more AI-like
        score = 1.0 - np.clip(falloff_ratio, 0, 1)
        return float(score)

    def _noise_residual_analysis(self, rgb: np.ndarray) -> float:
        """
        Extract noise residual using Gaussian blur subtraction.
        Real camera images have structured sensor noise (PRNU pattern).
        AI images tend to have spatially uniform or patterned synthetic noise.
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = gray - blurred

        # Measure local variance of the residual
        local_var = cv2.GaussianBlur(residual ** 2, (15, 15), 0)
        uniformity = 1.0 - np.std(local_var) / (np.mean(local_var) + 1e-8)

        # More uniform noise residual = more likely AI
        score = float(np.clip(uniformity, 0, 1))
        return score

    def _texture_analysis(self, gray: np.ndarray) -> float:
        """
        Analyze Local Binary Pattern (LBP) texture.
        AI images often have unnaturally smooth or repetitive texture patterns.
        """
        resized = cv2.resize(gray, (128, 128))

        # Compute LBP manually (simplified)
        lbp = np.zeros_like(resized, dtype=np.uint8)
        padded = np.pad(resized, 1, mode='edge')

        neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        for i, (dy, dx) in enumerate(neighbors):
            shifted = padded[1+dy:1+dy+128, 1+dx:1+dx+128]
            lbp |= ((shifted >= resized).astype(np.uint8) << i)

        # Histogram of LBP codes
        hist, _ = np.histogram(lbp, bins=32, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-8)

        # Entropy of LBP histogram
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(32)

        # Very low or very high entropy both suspicious for AI
        normalized_entropy = entropy / max_entropy
        score = 1.0 - abs(normalized_entropy - 0.75) * 2
        score = float(np.clip(score, 0, 1))
        return score

    def _channel_correlation(self, rgb: np.ndarray) -> float:
        """
        Measure R/G/B channel correlation.
        Real photos: channels correlate strongly due to physics of light.
        AI images: channels may have subtle mismatches from separate generation paths.
        """
        r = rgb[:,:,0].astype(float).flatten()
        g = rgb[:,:,1].astype(float).flatten()
        b = rgb[:,:,2].astype(float).flatten()

        # Sample for speed
        idx = np.random.choice(len(r), min(10000, len(r)), replace=False)
        r, g, b = r[idx], g[idx], b[idx]

        rg_corr = np.corrcoef(r, g)[0,1]
        rb_corr = np.corrcoef(r, b)[0,1]
        gb_corr = np.corrcoef(g, b)[0,1]

        avg_corr = np.mean([rg_corr, rb_corr, gb_corr])

        # Real photos: very high correlation (>0.9 typical)
        # AI images: slightly lower or artificially high
        deviation = abs(avg_corr - 0.92)
        score = float(np.clip(deviation * 5, 0, 1))
        return score

    def _edge_artifact_score(self, gray: np.ndarray) -> float:
        """
        Detect unnatural edge patterns.
        Diffusion/GAN upsampling creates characteristic edge artifacts.
        """
        resized = cv2.resize(gray, (256, 256))

        # Canny edges
        edges = cv2.Canny(resized, 50, 150).astype(float)

        # Sobel gradients
        sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Check for periodic edge patterns (upsampling artifact)
        edge_fft = np.abs(np.fft.fft2(edges))
        edge_fft_shifted = np.fft.fftshift(edge_fft)

        h, w = edge_fft_shifted.shape
        center = edge_fft_shifted[h//2-20:h//2+20, w//2-20:w//2+20]
        periphery = edge_fft_shifted.copy()
        periphery[h//2-20:h//2+20, w//2-20:w//2+20] = 0

        periodicity = np.max(periphery) / (np.mean(center) + 1e-8)
        score = float(np.clip(periodicity / 50.0, 0, 1))
        return score

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _radial_mask(self, h, w, r_min, r_max):
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - w//2)**2 + (y - h//2)**2)
        return (dist >= r_min) & (dist < r_max)

    def _risk_level(self, confidence: float) -> str:
        if confidence < 0.35:
            return "LOW"
        elif confidence < 0.65:
            return "MEDIUM"
        else:
            return "HIGH"

    def _generate_heatmap(self, img_np: np.ndarray, signals: dict, original_path: Path) -> str:
        """Generate a saliency-style heatmap highlighting suspicious regions."""
        h, w = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Build heatmap from noise residual (most spatially informative signal)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = np.abs(gray - blurred)

        # Normalize and amplify
        heatmap = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original
        img_uint8 = img_np.astype(np.uint8)
        blended = cv2.addWeighted(img_uint8, 0.6, heatmap_rgb, 0.4, 0)

        # Add overall confidence score as signal strength
        overall = np.mean(list(signals.values()))
        alpha = float(np.clip(overall, 0.2, 0.8))
        final = cv2.addWeighted(img_uint8, 1-alpha, heatmap_rgb, alpha, 0)

        # Save
        out_path = original_path.parent / f"{original_path.stem}_heatmap.png"
        Image.fromarray(final).save(out_path)
        return str(out_path)
