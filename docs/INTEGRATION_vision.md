# Integración con `vision.py` (juego de Pictionary por webcam)

Este documento explica cómo el código de visión (OpenCV + seguimiento de la yema del dedo índice)
debe llamar a los modelos entrenados. Toda la inferencia está en `classifier.py`; `vision.py`
no tiene que saber nada de PyTorch ni del preprocesado.

```python
import classifier

top3 = classifier.predict(canvas_img)        # CNN (imagen del lienzo)
top3 = classifier.predict_strokes(strokes)   # BiLSTM (trazos grabados)
# -> [("cat", 0.81), ("dog", 0.07), ("bird", 0.03)]   (probabilidades softmax, orden descendente)
```

- Los modelos se cargan **una sola vez** (singleton perezoso) en la primera llamada.
- Checkpoints por defecto: `checkpoints/cnn/best.pt` y `checkpoints/bilstm/best.pt`.
  Se pueden cambiar con las variables de entorno `QUICKDRAW_CNN_CKPT` y `QUICKDRAW_BILSTM_CKPT`.
- Si falta el checkpoint se lanza `FileNotFoundError` con instrucciones (entrenar con
  `python train.py --model cnn|bilstm` o descargar `best.pt`).
- Tiempo objetivo: < 50 ms por predicción en CPU. Aun así, conviene **no predecir en cada
  frame**: por ejemplo cada 0,5 s o cuando el jugador levanta el dedo.
- También se puede usar la clase directamente:
  `clf = classifier.QuickDrawClassifier("checkpoints/cnn/best.pt")`, con `clf.classes`, `clf.mode`.
- Prueba rápida desde terminal:
  `python classifier.py --checkpoint checkpoints/cnn/best.pt --image dibujo.png`

## 1. Camino CNN (imagen): no hay que cambiar nada salvo la llamada

```python
guesses = classifier.predict(canvas_img, top_k=3)
```

Qué acepta `canvas_img` (se procesa con `preprocess.preprocess_canvas`, la misma función usada
para alinear con el entrenamiento):

- Imagen de OpenCV `(H, W)`, `(H, W, 1)`, `(H, W, 3)` en BGR o `(H, W, 4)`; `uint8` o `float`.
- **Cualquier fondo** (blanco, negro, de color): el fondo se detecta en el borde de la imagen y se
  invierte si hace falta.
- **Cualquier color de trazo** y **líneas gruesas**: no hay problema; el dibujo se recorta a su
  caja, se centra y se reescala a 64×64 como en el entrenamiento.
- Pasad **solo el lienzo** (la capa de dibujo), no el frame de la cámara con la mano encima.
  Si el lienzo está superpuesto al vídeo, guardad una copia limpia del lienzo y pasad esa.
- Lienzo vacío: no falla; devuelve resultados con probabilidades bajas y casi uniformes.
  `classifier.is_blank(canvas_img)` permite saltarse la predicción si no hay tinta.

## 2. Camino BiLSTM (trazos): lo que `vision.py` debe añadir

**El orden de los trazos NO se puede recuperar a partir de una imagen.** Una imagen no dice en qué
orden se dibujó cada línea, en qué dirección, ni dónde se levantó el "lápiz". El BiLSTM se entrenó
con secuencias de puntos `(dx, dy, fin_de_trazo)`, así que `predict_strokes` necesita que
`vision.py` **grabe la trayectoria del dedo mientras se dibuja**. Por eso `predict(imagen)` sobre
un modelo BiLSTM lanza un error explicativo. (Al revés sí es posible: `predict_strokes` sobre un
modelo CNN dibuja los trazos en una imagen y la clasifica.)

Qué hay que grabar:

1. Una **lista de trazos**; cada trazo es una lista de puntos `(x, y)`.
2. Coordenadas en **píxeles del lienzo** (las mismas con las que dibujáis en el canvas; si el
   frame se voltea en espejo, usad las coordenadas ya volteadas). No hace falta normalizarlas:
   `classifier` las escala a la caja 0..255 y las simplifica (RDP) igual que los datos oficiales.
   Antes de simplificar aplica un suavizado de media móvil de 5 puntos por trazo para quitar el
   temblor de la yema del dedo (en una simulación con ~1.5 px de ruido, la BiLSTM pasaba de 0.03
   a 0.26 de top-1 al suavizar). No hace falta suavizar en `vision.py`; sí conviene **no** enviar
   ya los puntos simplificados por otro método.
3. **Empezar un trazo nuevo cada vez que el "lápiz" se levanta**: dedo levantado / cambio de gesto
   (p. ej. de "dibujar" a "mover" o "borrar") / se pierde el seguimiento de la mano.
4. Opcional: descartar puntos a menos de ~2 px del anterior (ruido de la detección).
5. **Vaciar la lista cuando se borra el lienzo** (y al empezar una ronda nueva).
6. Llamar a `classifier.predict_strokes(strokes)`.

También se acepta el formato oficial de Quick, Draw! `[[[x0, x1, ...], [y0, y1, ...]], ...]`.
Trazos vacíos se ignoran, un trazo de un solo punto es válido y una lista vacía devuelve
resultados de baja confianza sin fallar.

### Esbozo de código (documentación; no tenemos vuestro `vision.py`)

```python
import math
import classifier


class StrokeRecorder:
    """Graba la trayectoria de la yema del índice como lista de trazos."""

    def __init__(self, min_dist=2.0):
        self.min_dist = min_dist
        self.strokes = []        # list[list[tuple[int, int]]]
        self._pen_down = False

    def update(self, point, drawing):
        """Llamar en cada frame.
        point: (x, y) de la yema en píxeles del lienzo, o None si no hay mano detectada.
        drawing: True si el gesto actual es "dibujar"."""
        if point is None or not drawing:
            self._pen_down = False              # lápiz arriba -> el próximo punto abre trazo
            return
        x, y = int(point[0]), int(point[1])
        if not self._pen_down:
            self.strokes.append([(x, y)])       # nuevo trazo
            self._pen_down = True
            return
        last = self.strokes[-1][-1]
        if math.hypot(x - last[0], y - last[1]) >= self.min_dist:
            self.strokes[-1].append((x, y))

    def clear(self):
        """Llamar cuando se borra el lienzo / empieza una ronda."""
        self.strokes = []
        self._pen_down = False

    def is_empty(self):
        return not any(self.strokes)


# --- dentro del bucle de vision.py --------------------------------------------
recorder = StrokeRecorder()

# en cada frame:
#   tip = (x, y) de la yema del índice (landmark 8 de MediaPipe) en coords del lienzo, o None
#   dibujando = gesto_actual == "dibujar"
recorder.update(tip, dibujando)

# al borrar el lienzo:
#   canvas[:] = color_fondo
#   recorder.clear()

# para adivinar (p. ej. cada 0,5 s o al levantar el dedo):
if not recorder.is_empty():
    guesses_lstm = classifier.predict_strokes(recorder.strokes, top_k=3)
    guesses_cnn = classifier.predict(canvas, top_k=3)
```

Notas:

- `recorder.strokes` es una lista de Python; si la predicción se hace en otro hilo, pasad una
  copia (`[list(s) for s in recorder.strokes]`).
- Si se implementa una "goma de borrar" parcial, el BiLSTM no puede saber qué se borró: lo más
  sencillo es que borrar = vaciar el lienzo y `recorder.clear()`, o usar solo la CNN en ese caso.
- Ambos modelos usan la misma lista de clases (`classes.json`); se pueden combinar, por ejemplo
  promediando las probabilidades de ambos para la misma clase.
