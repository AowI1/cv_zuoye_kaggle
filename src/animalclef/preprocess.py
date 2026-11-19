"""Species-aware image preprocessing.

Ported and extended from the AnimalCLEF26 winning solution (MIA-AI-Team) and
public top solutions:

- lynx: unsharp mask + CLAHE + bilateral filter, with optional IR/visible
  domain normalisation (IR frames are near-grayscale; we neutralise their
  chroma and align luminance so IR and colour crops live in the same space).
- turtle: red-channel compensation + gamma correction, matching the
  underwater colour cast of SeaTurtleID2022.
- salamander: Sobel edge overlay + gamma, emphasising the spot pattern.
- lizard: glare inpainting + luminance alignment (Texas horned lizards have
  glossy ventral patterns; flash glare destroys identity signal).
"""
import cv2
import numpy as np


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _is_ir(img_rgb: np.ndarray, sat_threshold: float = 22.0) -> bool:
    """IR camera-trap frames are (near) grayscale: low mean HSV saturation."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean()) < sat_threshold


def _lab_align(img_rgb: np.ndarray, target_mean: float = 128.0,
               target_std: float = 64.0) -> np.ndarray:
    """Align the lightness channel to a common mean/std and neutralise chroma."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    l = (l - l.mean()) / (l.std() + 1e-6) * target_std + target_mean
    a[:] = 128
    b[:] = 128
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(_to_uint8(lab), cv2.COLOR_LAB2RGB)


def lynx_preprocessing(img_rgb: np.ndarray, normalise_ir: bool = True) -> np.ndarray:
    """Sharpen, equalise and denoise lynx camera-trap images."""
    img = img_rgb
    if normalise_ir and _is_ir(img_rgb):
        # Neutral chroma + luminance alignment bridges IR and visible domains.
        img = _lab_align(img)

    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clip = 1.5 if _is_ir(img_rgb) else 2.0
    v = cv2.createCLAHE(clipLimit=clip, tileGridSize=(3, 3)).apply(v)
    img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    return cv2.bilateralFilter(img, 9, 50, 60)


def turtle_preprocessing(img_rgb: np.ndarray) -> np.ndarray:
    """Compensate underwater red loss and apply gamma correction."""
    img = img_rgb.astype(np.float32) / 255.0
    img[:, :, 0] *= 1.3
    img = np.power(np.clip(img, 0, 1), 1.5)
    return _to_uint8(img * 255)


def salamander_preprocessing(img_rgb: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Overlay Sobel edges on the salamander image, then gamma-enhance."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sx, sy)
    edges = np.uint8(edges / (edges.max() + 1e-6) * 255)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.7, edges_rgb, 0.3, 0)
    out = np.power(overlay.astype(np.float32) / 255.0, gamma)
    return _to_uint8(out * 255)


def lizard_preprocessing(img_rgb: np.ndarray) -> np.ndarray:
    """Remove flash glare and align luminance for ventral spot patterns."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    glare = (l > 245).astype(np.uint8) * 255
    if glare.sum() > 0:
        glare = cv2.dilate(glare, np.ones((5, 5), np.uint8), iterations=1)
        img_rgb = cv2.inpaint(img_rgb, glare, 3, cv2.INPAINT_TELEA)
    lum = img_rgb.astype(np.float32).mean(axis=2, keepdims=True)
    scale = np.clip(128.0 / (lum.mean() + 1e-6), 0.6, 1.6)
    out = np.clip(img_rgb.astype(np.float32) * scale, 0, 255)
    return _to_uint8(out)


def preprocess(img_rgb: np.ndarray, dataset: str, enabled: bool = True) -> np.ndarray:
    """Route an RGB uint8 image through the species-specific recipe."""
    if not enabled:
        return img_rgb
    if dataset == "LynxID2025":
        return lynx_preprocessing(img_rgb)
    if dataset == "SeaTurtleID2022":
        return turtle_preprocessing(img_rgb)
    if dataset == "SalamanderID2025":
        return salamander_preprocessing(img_rgb)
    if dataset == "TexasHornedLizards":
        return lizard_preprocessing(img_rgb)
    return img_rgb
