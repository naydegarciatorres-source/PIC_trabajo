# QuickDraw_IA — Clasificador de dibujos para el Pictionary con webcam

Módulo de IA del proyecto: una persona dibuja con el dedo índice delante de la webcam y una red
neuronal propia reconoce el dibujo entre **45 categorías** del dataset oficial
[Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset). Todo está preparado
para ampliarse a las **345 categorías** cambiando solo `classes.json`.

Se incluyen dos modelos en PyTorch:

| Modelo | Entrada | Uso en el juego |
|---|---|---|
| `QuickDrawCNN` (referencia) | imagen 64×64 renderizada (`canvas_img` de OpenCV) | `classifier.predict(canvas_img)` — no requiere cambios en `vision.py` |
| `StrokeBiLSTM` | secuencia de puntos `(dx, dy, fin_de_trazo)` | `classifier.predict_strokes(strokes)` — requiere que `vision.py` grabe los trazos |

La BiLSTM está **inspirada conceptualmente** en el tutorial oficial de TensorFlow
*Recurrent Neural Networks for Drawing Classification* (Conv1D → LSTM bidireccional → suma
temporal → capa lineal). **No** es una reproducción del modelo del juego original de Google ni
del tutorial exacto.

## Estructura

```
classes.json            45 categorías (etiqueta = índice en la lista)
preprocess.py           transformaciones compartidas (train + inferencia) y CLI de descarga/preparación
data_loader.py          Dataset con memmap, render/secuencias al vuelo, aumentos, DataLoaders
model.py                QuickDrawCNN, StrokeBiLSTM, build_model, guardar/cargar checkpoints
train.py                entrenamiento (AdamW, warmup+coseno, early stopping, checkpoints, reanudación, AMP)
evaluate.py             top-1/top-3, métricas por clase, matriz de confusión, pares más confundidos
classifier.py           API para el juego: predict(canvas_img), predict_strokes(strokes)
notebooks/train_colab.ipynb   entrenamiento completo en Google Colab (GPU)
tests/                  99 tests automáticos (pytest)
results/                métricas, curvas y matrices de confusión por ejecución
docs/CONTRACT.md        contrato de interfaces entre módulos
docs/INTEGRATION_vision.md    qué tiene que hacer vision.py (en especial para la BiLSTM)
```

## Datos

- Fuente: ficheros oficiales **simplified ndjson**
  (`https://storage.googleapis.com/quickdraw_dataset/full/simplified/{clase}.ndjson`).
- Solo dibujos con `recognized == true`. Duplicados eliminados por `key_id` y por contenido.
- Configuración objetivo: **20.000 dibujos por categoría** (900.000 en total).
- Para no descargar los ficheros completos (50–300 MB cada uno) se hace *streaming* del inicio de
  cada fichero y se para al reunir `20.000 × 1.25` candidatos; de ahí se eligen 20.000 de forma
  determinista. Estimación: ~600 MB descargados para 45 clases (~4,5 GB para 345).
- División **80/10/10** determinista por hash de `key_id` → un mismo dibujo nunca puede aparecer
  en dos conjuntos; `verify_processed_dataset` lo comprueba al terminar y los tests también.
- Almacenamiento compacto: coordenadas `uint8` + offsets (`.npy` cargados con `mmap_mode="r"`).
  Las imágenes para la CNN se **renderizan al vuelo**, así que el mismo dataset sirve para ambos
  modelos y la RAM no se satura (20k/clase ocupan del orden de ~100 MB en disco).

## Instalación local (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
(Para usar la GPU local, instalar la rueda CUDA de PyTorch desde https://pytorch.org.)
Nota: el `.venv` actual tiene `opencv-python-headless`; el juego con ventana necesita
`opencv-python`. No instaléis los dos a la vez en el mismo entorno.

## Uso

```powershell
# 0) comprobar que los nombres de classes.json existen en categories.txt oficial
python preprocess.py --categories-check

# 1) datos (prueba pequeña: 200/clase, ~40 s)   |  real: --per-class 20000
python preprocess.py --per-class 200 --out-dir data/processed_smoke --raw-dir data/raw_smoke

# 2) entrenamiento
python train.py --model cnn    --data-dir data/processed --epochs 30 --amp --resume auto
python train.py --model bilstm --data-dir data/processed --epochs 30 --amp --resume auto

# 3) evaluación (test)
python evaluate.py --checkpoint checkpoints/cnn/best.pt --data-dir data/processed --split test

# 4) tests
python -m pytest -q
```

Salidas: `checkpoints/{run}/last.pt` (cada época) y `best.pt` (mejor top-1 de validación);
`results/{run}/history.json`, `curves.png`, `config.json`, `metrics_test.json`,
`per_class_test.csv`, `confusion_test.png`.
`--resume auto` continúa desde `last.pt` (modelo, optimizador, scheduler, scaler AMP, historial,
early stopping y estado aleatorio).

## Integración con el juego

```python
import classifier

top3 = classifier.predict(canvas_img)           # lista de (categoría, probabilidad), p. ej. [('cat', p1), ('dog', p2), ('bird', p3)]
top3 = classifier.predict_strokes(strokes)      # strokes = [[(x, y), ...], [(x, y), ...], ...]
```

- `predict` acepta el lienzo de OpenCV tal cual: BGR/gris/BGRA, fondo claro u oscuro, cualquier
  color y grosor de línea. Recorta y centra el dibujo igual que en el entrenamiento.
  Pasad el **lienzo**, no el frame de la cámara.
- `predict_strokes` necesita los trazos en orden, con separación entre trazos. **El orden de los
  trazos no se puede recuperar a partir de una imagen**, así que `vision.py` debe grabarlos
  mientras se dibuja: ver [`docs/INTEGRATION_vision.md`](docs/INTEGRATION_vision.md).
- Checkpoints por defecto: `checkpoints/cnn/best.pt` y `checkpoints/bilstm/best.pt`
  (o variables de entorno `QUICKDRAW_CNN_CKPT`, `QUICKDRAW_BILSTM_CKPT`).
- Latencia medida en CPU (portátil): ~11 ms `predict` (lienzo 640×480), ~5 ms `predict_strokes`.

## Resultados

**Todavía no hay resultados del entrenamiento real.** Lo único ejecutado hasta ahora es una
**prueba de humo** para comprobar que todo funciona: 200 dibujos/clase, 3–4 épocas en CPU.
Esos números (en `results/smoke_*`) **no** representan el rendimiento final:

| Prueba de humo (200/clase, test n=965) | top-1 | top-3 |
|---|---|---|
| CNN, 4 épocas | 0.809 | 0.937 |
| BiLSTM, 3 épocas | 0.269 | 0.490 |

La BiLSTM converge más despacio y necesita muchos más datos y épocas; con tan pocos datos estos
números no permiten compararla con la CNN.

## Ampliar a 345 categorías

1. Sustituir la lista de `classes.json` por los nombres exactos de `categories.txt` oficial.
2. `python preprocess.py --categories-check`
3. Volver a ejecutar `preprocess.py` y `train.py` (`num_classes` se deduce de `classes.json`;
   no hay ningún 45 escrito en el código). Los checkpoints guardan su propia lista de clases.
