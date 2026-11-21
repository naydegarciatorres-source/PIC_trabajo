"""Prueba de integración: dibujo con el dedo (hand_detection.py) + clasificador BEiT (345 clases).

    .venv\\Scripts\\python.exe test_webcam_beit.py

1. Carga BEiT una sola vez (pesos locales si existen; si no, los descarga de Hugging Face).
2. Abre la webcam con capturar_dibujo(devolver_trazos=True):
     - puño izquierdo cerrado + índice derecho      -> dibujar
     - puño izquierdo + palma derecha abierta ~1 s  -> borrar todo
     - pulgar arriba con la izquierda               -> dibujo terminado
     - 'q' o cerrar la ventana                      -> salir
3. Clasifica el dibujo de dos formas y muestra el top-3 (345 categorías) con su porcentaje y
   el tiempo de inferencia de cada una:
     - TRAZOS: los puntos originales del dedo en píxeles  -> clf.predict_strokes(trazos)
     - IMAGEN: la imagen del lienzo a tamaño completo      -> clf.predict(imagen)
4. Ventana "Resultado BEiT": cualquier tecla = dibujar otra vez; 'q' o Esc = salir.

Cada dibujo se guarda (trazos + predicciones) en dibujos_webcam.ndjson para poder evaluarlo
después (usa --no-guardar para desactivarlo).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# Pesos en local (opcional, no se suben a GitHub): carpeta weights/ junto a este archivo, o la
# variable de entorno BEIT_SKETCH_MODEL. Si no hay ninguna, se descargan de Hugging Face (~348 MB).
PESOS_LOCALES = os.path.join(HERE, "weights", "beit-sketch-classifier")

import beit_classifier as BC  # noqa: E402
from hand_detection import capturar_dibujo  # noqa: E402


def clasificar(clf, imagen, trazos, top_k=3):
    """Devuelve {"trazos": (top, ms), "imagen": (top, ms)} con top = [(clase, prob), ...]."""
    out = {}
    for metodo, f in (("trazos", lambda: clf.predict_strokes(trazos, top_k)),
                      ("imagen", lambda: clf.predict(imagen, top_k))):
        t0 = time.perf_counter()
        top = f()
        out[metodo] = (top, (time.perf_counter() - t0) * 1000)
    return out


def imprimir(res):
    print("\n" + "=" * 62)
    for metodo, (top, ms) in res.items():
        print(f"  {metodo.upper():7s} ({ms:6.1f} ms)")
        for i, (c, p) in enumerate(top, 1):
            print(f"     {i}. {c:25s} {100 * p:5.1f} %")
    print("=" * 62)


def ventana_resultado(clf, imagen, trazos, res):
    """Lienzo + lo que ve BEiT con cada método + top-3 (sin tildes: putText no las admite)."""
    lienzo = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR) if imagen.ndim == 2 else imagen.copy()
    h = 480
    lienzo = cv2.resize(lienzo, (int(lienzo.shape[1] * h / lienzo.shape[0]), h))
    panel = np.full((h, 520, 3), 255, np.uint8)
    for col, (metodo, img224) in enumerate((("trazos", clf.strokes_to_image(trazos)),
                                            ("imagen", clf.canvas_to_image(imagen)))):
        x0 = 20 + col * 250
        if img224 is not None:
            panel[40:264, x0:x0 + 224] = cv2.cvtColor(img224, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(panel, (x0, 40), (x0 + 224, 264), (180, 180, 180), 1)
        top, ms = res[metodo]
        cv2.putText(panel, f"{metodo.upper()}  {ms:.0f} ms", (x0, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2)
        for i, (c, p) in enumerate(top):
            color = (0, 130, 0) if i == 0 else (60, 60, 60)
            cv2.putText(panel, f"{i + 1}. {c[:16]}", (x0, 300 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2 if i == 0 else 1)
            cv2.putText(panel, f"{100 * p:.1f} %", (x0 + 20, 322 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1)
    cv2.putText(panel, "Tecla: otro dibujo   q/Esc: salir", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (120, 0, 0), 1)
    cv2.imshow("Resultado BEiT", np.hstack([lienzo, panel]))
    k = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Resultado BEiT")
    cv2.waitKey(1)
    return k not in (ord("q"), 27)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--modelo", default=None, help="carpeta local con los pesos o id de Hugging Face")
    ap.add_argument("--device", default=None, help="cpu / cuda (por defecto: GPU si hay)")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--guardar", default=os.path.join(HERE, "dibujos_webcam.ndjson"))
    ap.add_argument("--no-guardar", action="store_true")
    ap.add_argument("--etiqueta", default=None, help="lo que vas a dibujar (se guarda para evaluar)")
    a = ap.parse_args(argv)

    modelo = a.modelo or os.environ.get(BC.ENV_MODEL) or (PESOS_LOCALES if os.path.isdir(PESOS_LOCALES) else None)
    print("Cargando BEiT desde:", modelo or BC.DEFAULT_MODEL)
    t0 = time.perf_counter()
    clf = BC.BeitClassifier(modelo, device=a.device)
    print(f"  OK: {len(clf.classes)} categorías, dispositivo {clf.device}, "
          f"{sum(p.numel() for p in clf.model.parameters()) / 1e6:.1f} M parámetros, "
          f"{time.perf_counter() - t0:.1f} s; torch {torch.__version__}")
    clf.predict_strokes([[(0, 0), (100, 100)]])  # calentamiento: la 1.ª inferencia siempre es más lenta

    n = 0
    while True:
        print("\nAbriendo la webcam... (puño izq. + índice der. = dibujar; palma der. 1 s = borrar; "
              "pulgar arriba izq. = terminar; 'q' = salir)")
        imagen, trazos = capturar_dibujo(camara=a.camara, devolver_trazos=True)
        if imagen is None:
            print("Cancelado: se cierra la prueba.")
            break
        if not trazos:
            print("El lienzo está vacío: no hay nada que clasificar. Vuelve a intentarlo.")
            continue
        n += 1
        print(f"Dibujo {n}: {len(trazos)} trazos, {sum(len(t) for t in trazos)} puntos, "
              f"imagen {imagen.shape} {imagen.dtype}")
        res = clasificar(clf, imagen, trazos, a.top)
        imprimir(res)
        if not a.no_guardar:
            rec = {"etiqueta": a.etiqueta, "lienzo": [int(imagen.shape[1]), int(imagen.shape[0])],
                   "trazos": [[[round(x, 2), round(y, 2)] for x, y in t] for t in trazos],
                   **{f"pred_{m}": [[c, round(p, 4)] for c, p in top] for m, (top, _) in res.items()},
                   **{f"ms_{m}": round(ms, 1) for m, (_, ms) in res.items()}}
            with open(a.guardar, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not ventana_resultado(clf, imagen, trazos, res):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
