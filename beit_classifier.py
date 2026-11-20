"""Clasificador de dibujos para el Pictionary con BEiT-sketch (345 categorías de Quick, Draw!).

Modelo preentrenado: ``kmewhort/beit-sketch-classifier`` (Hugging Face, licencia Apache-2.0,
BEiT-base, ~87 M parámetros, 348 MB). NO lo hemos entrenado nosotros: solo lo usamos.
La primera vez se descarga solo a la caché de Hugging Face (~348 MB); después funciona sin red.

Uso desde el bucle del juego::

    from beit_classifier import BeitClassifier

    clf = BeitClassifier()                       # cargar UNA vez al arrancar (tarda unos segundos)

    # 1) Mejor opción: los trazos del dedo, en píxeles del lienzo
    #    trazos = [[(x, y), (x, y), ...], [(x, y), ...], ...]
    top3 = clf.predict_strokes(trazos)           # [("cat", 0.71), ("dog", 0.12), ("lion", 0.05)]

    #    Si los trazos vienen normalizados 0-1 (Lienzo.trazos de hand_detection.py),
    #    indica el tamaño del lienzo para no deformar el dibujo:
    top3 = clf.predict_strokes(lienzo.trazos, image_size=(ancho, alto))

    # 2) Alternativa: solo la imagen que devuelve capturar_dibujo() (a tamaño completo,
    #    NO con tamano=(28, 28): con BEiT la precisión cae del 84 % al 38 %)
    top3 = clf.predict(imagen)

Ambos devuelven una lista de ``(categoría_en_inglés, probabilidad)`` ordenada de mayor a menor,
o una lista vacía si el dibujo está en blanco. Las categorías son las 345 oficiales.

Resultados medidos (CPU, 89 dibujos reales hechos con ratón, 345 clases): top-1 0,90 con
trazos y 0,85 con la imagen; ~160 ms por dibujo. Con 4.500 dibujos oficiales: 0,87 / 0,79.

Preparación del dibujo (igual que la del autor del modelo, ``binToPIL``): los trazos se suavizan
y simplifican como los dibujos oficiales (``preprocess.simplify_strokes``: coordenadas 0..255,
RDP eps 2) y se dibujan en 224x224, fondo blanco, margen 8, línea negra de grosor 7. Si solo hay
imagen, las líneas se adelgazan a 1 píxel (esqueleto) y se vuelven a dibujar con ese grosor.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as P  # noqa: E402

DEFAULT_MODEL = "kmewhort/beit-sketch-classifier"
DEFAULT_REVISION = "e0f58aa3b774e6eadf9c179ca7fc553d97bc1620"  # versión exacta que evaluamos
ENV_MODEL = "BEIT_SKETCH_MODEL"  # opcional: ruta local a una carpeta con los pesos

IMG_SIZE = 224
PADDING = 8
LINE_WIDTH = 7
SCALE = (IMG_SIZE - 2 * PADDING) / 256.0


# ----------------------------------------------------------------------------------------------
# Preparación de la entrada
# ----------------------------------------------------------------------------------------------
def render_strokes(strokes: Sequence[np.ndarray]) -> np.ndarray:
    """Trazos ya simplificados (0..255) -> imagen 224x224 uint8, fondo 255 y línea 0."""
    img = np.full((IMG_SIZE, IMG_SIZE), 255, np.uint8)
    for s in strokes:
        q = np.round(np.asarray(s, np.float32)[:, :2] * SCALE).astype(np.int32) + PADDING
        if len(q) == 1:
            cv2.line(img, tuple(map(int, q[0])), tuple(map(int, q[0])), 0, LINE_WIDTH)
        for a, b in zip(q[:-1], q[1:]):
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 0, LINE_WIDTH)
    return img


def render_points(pts: np.ndarray) -> np.ndarray:
    """Puntos de un esqueleto (0..255) -> imagen 224x224 con líneas de grosor 7."""
    img = np.full((IMG_SIZE, IMG_SIZE), 255, np.uint8)
    if len(pts):
        q = np.clip(np.round(pts * SCALE).astype(np.int32) + PADDING, 0, IMG_SIZE - 1)
        mask = np.zeros_like(img)
        mask[q[:, 1], q[:, 0]] = 255
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LINE_WIDTH, LINE_WIDTH)))
        img[mask > 0] = 0
    return img


def _zhang_suen_thinning(mask: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Adelgazamiento Zhang-Suen de una imagen booleana -> esqueleto de 1 píxel."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = img
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(n.astype(np.int32) for n in nb)
            seq = nb + [P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            if step == 0:
                c1, c2 = P2 * P4 * P6, P4 * P6 * P8
            else:
                c1, c2 = P2 * P4 * P8, P2 * P6 * P8
            rm = (p[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if rm.any():
                img = img.copy()
                img[1:-1, 1:-1][rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def canvas_to_skeleton_points(canvas: np.ndarray, work_size: int = 256) -> np.ndarray:
    """Imagen del lienzo (gris o BGR, fondo claro u oscuro) -> puntos del esqueleto (K, 2) en
    0..255. Lienzo vacío -> array (0, 2)."""
    img = np.asarray(canvas)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = gray.astype(np.float32)
    if gray.size and gray.max() <= 1.0:
        gray = gray * 255.0
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    ink = np.abs(gray - float(np.median(border))) > 60
    if ink.sum() < 3:
        return np.zeros((0, 2), np.float32)
    ys, xs = np.nonzero(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    h, w = crop.shape
    s = (work_size - 1) / max(h, w)
    crop = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) > 127
    yy, xx = np.nonzero(_zhang_suen_thinning(crop))
    pts = np.stack([xx, yy], 1).astype(np.float32)
    if len(pts) == 0:
        return pts
    pts -= pts.min(0)
    if pts.max() > 0:
        pts *= 255.0 / pts.max()
    return pts


def _to_pixel_strokes(strokes: Any, image_size: tuple[int, int] | None) -> list[np.ndarray]:
    """Lista de trazos [(x, y), ...] -> arrays (n, 2) float32; si ``image_size=(ancho, alto)``,
    las coordenadas 0-1 se pasan a píxeles (evita estirar el dibujo en lienzos 4:3)."""
    out = []
    for s in strokes or []:
        a = np.asarray(s, np.float32)
        if a.size == 0:
            continue
        a = a.reshape(-1, a.shape[-1])[:, :2]
        if image_size is not None:
            a = a * np.asarray(image_size, np.float32)
        out.append(a)
    return out


# ----------------------------------------------------------------------------------------------
# Clasificador
# ----------------------------------------------------------------------------------------------
class BeitClassifier:
    """BEiT-sketch con las 345 categorías de Quick, Draw!.

    Args:
        model: id de Hugging Face o carpeta local con los pesos. Por defecto, la variable de
            entorno ``BEIT_SKETCH_MODEL`` si existe, y si no ``kmewhort/beit-sketch-classifier``.
        revision: versión exacta del modelo en Hugging Face (ignorada para carpetas locales).
        device: "cuda", "cpu" o None (GPU si hay).
        smooth_window: suavizado de los trazos antes de simplificarlos (5 = recomendado webcam).
    """

    def __init__(self, model: str | None = None, revision: str = DEFAULT_REVISION,
                 device: str | None = None, smooth_window: int = 5):
        from transformers import BeitForImageClassification

        model = model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        kwargs = {} if os.path.isdir(model) else {"revision": revision}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BeitForImageClassification.from_pretrained(model, **kwargs).eval().to(self.device)
        cfg = self.model.config
        self.classes: list[str] = [cfg.id2label[i] for i in range(cfg.num_labels)]
        self.smooth_window = int(smooth_window)

    # -- imágenes de entrada ------------------------------------------------------------------
    def strokes_to_image(self, strokes: Any, image_size: tuple[int, int] | None = None) -> np.ndarray | None:
        """Trazos -> imagen 224x224 tal como la ve el modelo (None si no hay trazos)."""
        px = _to_pixel_strokes(strokes, image_size)
        if not px:
            return None
        simp = P.simplify_strokes(px, epsilon=2.0, smooth_window=self.smooth_window)
        return render_strokes(simp) if simp else None

    def canvas_to_image(self, image: np.ndarray) -> np.ndarray | None:
        """Imagen del lienzo -> imagen 224x224 tal como la ve el modelo (None si está en blanco)."""
        if image is None:
            return None
        pts = canvas_to_skeleton_points(image)
        return render_points(pts) if len(pts) else None

    # -- probabilidades -----------------------------------------------------------------------
    @torch.inference_mode()
    def _proba(self, img224: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(img224.astype(np.float32) / 255.0)
        x = ((x - 0.5) / 0.5)[None, None].expand(1, 3, IMG_SIZE, IMG_SIZE).contiguous().to(self.device)
        return F.softmax(self.model(pixel_values=x).logits.float(), -1)[0].cpu().numpy()

    def predict_proba_strokes(self, strokes: Any, image_size: tuple[int, int] | None = None) -> np.ndarray | None:
        """Probabilidades de las 345 clases (orden de ``self.classes``), o None si está vacío."""
        img = self.strokes_to_image(strokes, image_size)
        return None if img is None else self._proba(img)

    def predict_proba(self, image: np.ndarray) -> np.ndarray | None:
        img = self.canvas_to_image(image)
        return None if img is None else self._proba(img)

    # -- predicción -----------------------------------------------------------------------------
    def _top(self, p: np.ndarray | None, top_k: int) -> list[tuple[str, float]]:
        if p is None:
            return []
        idx = np.argsort(-p)[:top_k]
        return [(self.classes[i], float(p[i])) for i in idx]

    def predict_strokes(self, strokes: Any, top_k: int = 3,
                        image_size: tuple[int, int] | None = None) -> list[tuple[str, float]]:
        """Trazos en orden ([[(x, y), ...], ...]) -> [(categoría, probabilidad), ...]."""
        return self._top(self.predict_proba_strokes(strokes, image_size), top_k)

    def predict(self, image: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """Imagen del lienzo (np.ndarray gris o BGR) -> [(categoría, probabilidad), ...]."""
        return self._top(self.predict_proba(image), top_k)


# ----------------------------------------------------------------------------------------------
# Atajos a nivel de módulo (el modelo se carga una sola vez)
# ----------------------------------------------------------------------------------------------
_CLF: BeitClassifier | None = None


def get_classifier(**kwargs) -> BeitClassifier:
    global _CLF
    if _CLF is None:
        _CLF = BeitClassifier(**kwargs)
    return _CLF


def predict_strokes(strokes: Any, top_k: int = 3, image_size: tuple[int, int] | None = None):
    return get_classifier().predict_strokes(strokes, top_k, image_size)


def predict(image: np.ndarray, top_k: int = 3):
    return get_classifier().predict(image, top_k)


if __name__ == "__main__":
    # Prueba rápida: un círculo dibujado como trazo
    t = np.linspace(0, 2 * np.pi, 60)
    circulo = [list(zip(320 + 120 * np.cos(t), 240 + 120 * np.sin(t)))]
    print(BeitClassifier().predict_strokes(circulo))
