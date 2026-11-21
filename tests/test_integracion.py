"""Integración hand_detection.capturar_dibujo() + beit_classifier sin webcam ni mano real.

Se sustituyen la cámara (fotogramas negros), MediaPipe (manos "programadas" fotograma a
fotograma) y las ventanas de OpenCV; el resto de capturar_dibujo() se ejecuta tal cual:
gestos, filtro One-Euro, cortes de trazo, borrado y pulgar arriba.
"""
import math
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip("mediapipe")
hd = pytest.importorskip("hand_detection", reason="hand_detection.py está en la rama feature/vision-canvas")
import test_webcam_beit as twb  # noqa: E402
import beit_classifier as BC

W, H = 640, 480


# ---------------------------------------------------------------- manos sintéticas (21 puntos)
def _lm(x, y, z=0.0):
    return NS(x=x, y=y, z=z)


def _world(dedos_rectos):
    pts = [_lm(0, 0)] + [_lm(-1, 0.5 * k) for k in range(1, 5)]  # muñeca + pulgar
    for i in range(4):
        if dedos_rectos:
            pts += [_lm(i, 1), _lm(i, 2), _lm(i, 3), _lm(i, 4)]
        else:  # dedo doblado: la punta vuelve hacia la palma
            pts += [_lm(i, 1), _lm(i, 2), _lm(i, 2, 1), _lm(i, 1, 1)]
    return pts


def _img(wrist_x, pulgar_arriba=False, punta_indice=None):
    pts = [_lm(wrist_x, 0.8)] + [_lm(wrist_x, 0.7) for _ in range(20)]
    pts[5] = _lm(wrist_x, 0.7)
    pts[4] = _lm(wrist_x, 0.3 if pulgar_arriba else 0.75)
    if punta_indice is not None:
        pts[8] = _lm(*punta_indice)
    return pts


def izquierda(gesto):  # "puno" o "ok"
    return _img(0.02, pulgar_arriba=(gesto == "ok")), _world(False)


def derecha(punto, palma_abierta=False):
    return _img(0.98, punta_indice=punto), _world(palma_abierta)


def frame(izq=None, der=None):
    manos = [m for m in (izq, der) if m is not None]
    return NS(hand_landmarks=[m[0] for m in manos], hand_world_landmarks=[m[1] for m in manos],
              handedness=[[NS(category_name="Left")] for _ in manos])  # solo se usa con 1 mano


def circulo(n=40, cx=0.5, cy=0.5, r_px=120):
    return [(cx + r_px / W * math.cos(t), cy + r_px / H * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, n)]


def guion():
    """Dibujar circulo + linea, borrar, dibujar circulo, pulgar arriba."""
    f = []
    trazo = lambda pts: f.extend(frame(izquierda("puno"), derecha(p)) for p in pts)
    levantar = lambda n=8: f.extend(frame(None, derecha((0.5, 0.5))) for _ in range(n))
    trazo(circulo()); levantar()
    trazo([(0.2 + 0.01 * i, 0.8) for i in range(20)]); levantar()
    f.extend(frame(izquierda("puno"), derecha((0.5, 0.5), palma_abierta=True)) for _ in range(32))
    levantar()
    trazo(circulo(cx=0.45, cy=0.4)); f.extend(frame(izquierda("ok"), derecha((0.5, 0.5))) for _ in range(12))
    return f


@pytest.fixture
def simular(monkeypatch):
    def _activar(frames):
        it = iter(frames)
        estado = {"fin": False}

        class Cam:
            def isOpened(self): return True
            def read(self):
                return (True, np.zeros((H, W, 3), np.uint8)) if not estado["fin"] else (False, None)
            def release(self): pass

        class Landmarker:
            def detect_for_video(self, img, ts):
                try:
                    return next(it)
                except StopIteration:
                    estado["fin"] = True
                    return frame()
            def close(self): pass

        reloj = {"t": 1000.0}
        def monotonic():
            reloj["t"] += 1 / 30  # webcam a 30 fps
            return reloj["t"]

        monkeypatch.setattr(hd.cv2, "VideoCapture", lambda i: Cam())
        monkeypatch.setattr(hd.vision.HandLandmarker, "create_from_options", lambda o: Landmarker())
        monkeypatch.setattr(hd, "time", NS(monotonic=monotonic))
        for nombre, valor in (("imshow", None), ("waitKey", -1), ("getWindowProperty", 1),
                              ("destroyAllWindows", None)):
            monkeypatch.setattr(hd.cv2, nombre, lambda *a, v=valor, **k: v)
    return _activar


# ---------------------------------------------------------------- tests
def test_modelo_mediapipe_carga_y_procesa_un_frame():
    import mediapipe as mp
    opts = hd.vision.HandLandmarkerOptions(base_options=hd.python.BaseOptions(model_asset_path=hd.RUTA_MODELO),
                                           running_mode=hd.vision.RunningMode.VIDEO, num_hands=2)
    lm = hd.vision.HandLandmarker.create_from_options(opts)
    try:
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((H, W, 3), np.uint8)), 1)
        assert r.hand_landmarks == []
    finally:
        lm.close()


def test_devuelve_imagen_y_trazos_en_pixeles_tras_borrar(simular):
    simular(guion())
    imagen, trazos = hd.capturar_dibujo(devolver_trazos=True)
    assert imagen.shape == (H, W) and imagen.dtype == np.uint8
    assert len(trazos) == 1, "el borrado debe quitar los dos primeros trazos"
    pts = np.array(trazos[0])
    assert pts.shape[1] == 2 and len(pts) >= 20
    assert 0 <= pts[:, 0].min() and pts[:, 0].max() <= W and 0 <= pts[:, 1].min() and pts[:, 1].max() <= H
    # en píxeles (no 0-1) y redondo en píxeles: mismo ancho que alto (~240 px)
    ancho, alto = np.ptp(pts, 0)
    assert 180 < ancho < 260 and abs(ancho - alto) < 0.15 * ancho
    # la imagen se ha pintado exactamente con esos trazos
    tinta = [imagen[min(H - 1, int(y)), min(W - 1, int(x))] < 128 for x, y in pts]
    assert np.mean(tinta) > 0.9
    assert (imagen < 128).sum() > 500 and imagen[int(0.8 * H), int(0.25 * W)] == 255  # línea borrada


def test_sin_devolver_trazos_se_comporta_como_antes(simular):
    simular(guion())
    imagen = hd.capturar_dibujo()
    assert isinstance(imagen, np.ndarray) and imagen.shape == (H, W)


def test_tamano_y_recortar_no_afectan_a_los_trazos(simular):
    simular(guion())
    imagen, trazos = hd.capturar_dibujo(tamano=(28, 28), recortar=True, devolver_trazos=True)
    assert imagen.shape == (28, 28)
    assert np.array(trazos[0]).max() > 28  # siguen en píxeles del lienzo completo


def test_cancelar_devuelve_none_y_lista_vacia(simular):
    simular([])  # la cámara deja de dar imágenes = como cerrar
    assert hd.capturar_dibujo(devolver_trazos=True) == (None, [])
    simular([])
    assert hd.capturar_dibujo() is None


@pytest.fixture(scope="module")
def clf_real():
    ruta = os.environ.get(BC.ENV_MODEL) or twb.PESOS_LOCALES
    if not os.path.isdir(ruta):
        pytest.skip("no están los pesos de BEiT en local")
    return BC.BeitClassifier(ruta, device="cpu")


def test_beit_real_clasifica_el_dibujo_por_trazos_y_por_imagen(simular, clf_real):
    assert len(clf_real.classes) == 345
    simular(guion())
    imagen, trazos = hd.capturar_dibujo(devolver_trazos=True)
    res = twb.clasificar(clf_real, imagen, trazos, top_k=3)
    for metodo in ("trazos", "imagen"):
        top, ms = res[metodo]
        assert len(top) == 3 and ms > 0
        assert all(0 <= p <= 1 for _, p in top) and top[0][1] >= top[1][1] >= top[2][1]
    assert "circle" in [c for c, _ in res["trazos"][0]]
    assert "circle" in [c for c, _ in res["imagen"][0]]
