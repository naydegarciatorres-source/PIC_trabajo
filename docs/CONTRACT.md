# Internal contract between modules

This file is the single source of truth for interfaces. Every module must follow it.
Python: use `.venv\Scripts\python.exe` (Windows, local venv, CPU torch). Code must also run on
Linux/Colab with CUDA. Python >= 3.10. Only dependencies: torch, numpy, opencv-python(-headless),
matplotlib, tqdm, requests, pytest, nbformat.

## Classes
`classes.json` -> `{"classes": [...]}`. Label = index in that list. `num_classes = len(classes)`.
Never hard-code 45 anywhere; everything must work for 345.

## File ownership (only the owner edits a file)
| File | Owner |
|---|---|
| `preprocess.py`, `data_loader.py`, `tests/test_preprocess.py`, `tests/test_data_loader.py` | Data agent |
| `model.py`, `tests/test_model.py` | Architecture agent |
| `train.py`, `notebooks/train_colab.ipynb`, `requirements.txt`, `tests/test_train.py` | Training agent |
| `evaluate.py`, `classifier.py`, `docs/INTEGRATION_vision.md`, `tests/test_classifier.py`, `tests/test_evaluate.py` | Evaluation/integration agent |
| `classes.json`, `docs/CONTRACT.md`, `README.md`, `.gitignore`, `tests/conftest.py` | Coordinator |

## 1. Processed data format (written by `preprocess.py`, read by `data_loader.py`)
Directory `data/processed/` containing, for each split in `train`, `val`, `test`:
- `{split}_points.npy`  uint8 array (P, 3): columns `x, y, end_of_stroke` (x,y in 0..255, the
  official *simplified* coordinates; `end_of_stroke` = 1 on the last point of each stroke, else 0).
- `{split}_offsets.npy` int64 array (N+1,): drawing i = `points[offsets[i]:offsets[i+1]]`.
- `{split}_labels.npy`  int16 array (N,).
- `{split}_key_ids.npy` int64 array (N,) (official `key_id`, used to prove no leakage).
- `meta.json`: `{"classes": [...], "counts": {split: {class: n}}, "source": url, "per_class": int,
  "split_ratios": [0.8, 0.1, 0.1], "seed": int}`.
Files are loaded with `np.load(..., mmap_mode="r")` so RAM stays low.
Split assignment is deterministic from a hash of `key_id` (not random order) and duplicates
(same key_id or identical stroke content) are removed before splitting -> no leakage.
Only `recognized == true` drawings are kept.

## 2. Shared transforms (`preprocess.py`, pure functions, used at train AND inference time)
- `load_classes(path="classes.json") -> list[str]`
- `ndjson_strokes_to_points(strokes) -> np.ndarray (P,3) float32` : official format
  `[[[x0,x1..],[y0,y1..]], ...]` -> rows `x, y, end_of_stroke`.
- `normalize_strokes_input(strokes) -> list[np.ndarray (n_i,2)]`: accepts official format OR list of
  strokes as lists of `(x, y)` points / arrays. Empty strokes dropped.
- `smooth_stroke(stroke, window=5)` / `simplify_strokes(..., smooth_window=0)`: optional moving-average
  smoothing before RDP; `classifier.predict_strokes` uses window 5 for live fingertip jitter.
- `simplify_strokes(strokes, epsilon=2.0) -> same`: scale to 0..255 box (aspect preserved, like the
  official simplified data) then Ramer-Douglas-Peucker. Used by `predict_strokes` so live webcam
  strokes look like training data.
- `points_to_sequence(points, max_len=250) -> np.ndarray (T,3) float32` : follows the TF RNN
  tutorial: normalise x,y to [0,1] using the bounding box (per axis, zero range -> 1), take deltas
  between consecutive points, drop first row; third column = end_of_stroke (stroke separation).
  Truncate to `max_len`. T >= 1 always.
- `render_points(points, image_size=64, thickness=None, margin=0.1) -> np.ndarray (H,W) uint8`:
  white strokes (255) on black (0); drawing centred, aspect preserved, scaled to fit in
  `image_size*(1-2*margin)`. Default thickness = max(1, round(image_size/32)).
- `preprocess_canvas(canvas_img, image_size=64, margin=0.1) -> np.ndarray (H,W) float32 in [0,1]`:
  accepts OpenCV images: (H,W), (H,W,1), (H,W,3) BGR or (H,W,4); uint8 or float. Converts to gray,
  detects background from the image border (if background is light, invert) so strokes end up
  bright on dark, thresholds lightly, crops the bounding box of the ink, pads to square with the
  same margin as `render_points`, resizes with INTER_AREA. Blank canvas -> zeros (no crash).

## 3. Data loading (`data_loader.py`)
- `QuickDrawDataset(data_dir, split, mode="image"|"strokes", image_size=64, max_len=250,
  augment=False, max_per_class=None, seed=0)`
  - mode "image": item = `(tensor float32 (1,H,W) in [0,1], label int)` rendered on the fly with
    `render_points` (augment: random thickness, small rotation/scale/shift, done on the points).
  - mode "strokes": item = `(tensor float32 (T,3), label int)` via `points_to_sequence`
    (augment: small random scale/shear of x,y before normalisation).
- `collate_strokes(batch) -> (padded float32 (B,Tmax,3), lengths int64 (B,), labels int64 (B,))`
- `make_dataloaders(data_dir, mode, batch_size=256, num_workers=2, image_size=64, max_len=250,
  augment_train=True, max_per_class=None, seed=0) -> dict{"train","val","test"}` (missing split ->
  key absent). Uses `collate_strokes` for strokes mode. Picklable for Windows workers.
- `make_synthetic_dataset(out_dir, classes, n_per_class=20, seed=0)` writes a tiny valid
  dataset in the format of section 1 (random strokes) for tests.

## 4. Models (`model.py`)
- `QuickDrawCNN(num_classes, in_channels=1, dropout=0.3)`: input (B,1,H,W) float in [0,1],
  any H=W >= 28 (uses adaptive pooling); output logits (B,num_classes).
- `StrokeBiLSTM(num_classes, input_dim=3, conv_channels=(48,64,96), conv_kernels=(5,5,3),
  hidden_size=128, num_layers=3, dropout=0.3)`: TF-tutorial-inspired: Conv1d stack -> BiLSTM
  (packed sequences) -> masked sum over time -> Linear. `forward(x (B,T,3), lengths (B,) long)`
  -> logits (B,num_classes). `lengths` optional (None = all full length).
- `build_model(name: "cnn"|"bilstm", num_classes, **kwargs) -> nn.Module`
- `save_checkpoint(path, model, model_name, model_kwargs, classes, preprocessing: dict,
  optimizer=None, scheduler=None, epoch=0, best_metric=None, history=None, extra=None)`:
  `torch.save` of dict with keys `format_version=1, model_name, model_kwargs, num_classes, classes,
  preprocessing, state_dict, optimizer_state, scheduler_state, epoch, best_metric, history, extra`.
  Writes atomically (tmp file + `os.replace`).
- `load_checkpoint(path, map_location="cpu") -> dict` (uses `weights_only=False`, trusted local files).
- `load_model_from_checkpoint(path, device="cpu") -> (model.eval(), ckpt_dict)`.
- `preprocessing` dict: `{"mode": "image"|"strokes", "image_size": 64, "max_len": 250, "margin": 0.1}`.

## 5. Training (`train.py`)
CLI: `python train.py --model cnn|bilstm --data-dir data/processed --classes classes.json
 --epochs 30 --batch-size 256 --lr 1e-3 --weight-decay 1e-4 --patience 5 --num-workers 2
 --image-size 64 --max-len 250 --max-per-class N --seed 42 --run-name NAME --out-dir checkpoints
 --results-dir results --resume [path|auto] --amp --no-augment --device auto|cpu|cuda`.
Writes `checkpoints/{run}/last.pt` (every epoch) and `best.pt` (best val top-1),
`results/{run}/history.json`, `results/{run}/curves.png`. AdamW, cosine or ReduceLROnPlateau,
early stopping on val top-1, top-1/top-3 logged, seeds set, gradient clipping for LSTM.
Expose `main(argv=None)` for tests. Default run-name = model name.

## 6. Evaluation (`evaluate.py`)
CLI: `python evaluate.py --checkpoint checkpoints/cnn/best.pt --data-dir data/processed
 --split test --results-dir results`. Computes top-1, top-3, loss, per-class accuracy,
confusion matrix; writes `results/{run}/metrics_{split}.json`, `per_class_{split}.csv`,
`confusion_{split}.png`. Expose `main(argv=None)` and `evaluate_model(model, loader, num_classes,
device, mode) -> dict`, plus `topk_accuracy(logits, targets, ks=(1,3))`.

## 7. Game integration (`classifier.py`)
- `class QuickDrawClassifier(checkpoint_path, device=None)`; `.classes`, `.mode`.
  - `.predict(canvas_img, top_k=3) -> list[tuple[str, float]]` (image models).
  - `.predict_strokes(strokes, top_k=3) -> list[tuple[str, float]]` (stroke models).
- Module-level helpers with lazy singletons: `predict(canvas_img, top_k=3)` (default checkpoint
  `checkpoints/cnn/best.pt`), `predict_strokes(strokes, top_k=3)` (default
  `checkpoints/bilstm/best.pt`), `load_classifier(path)`. Override with env vars
  `QUICKDRAW_CNN_CKPT`, `QUICKDRAW_BILSTM_CKPT`.
- Probabilities = softmax, sorted descending. Empty drawing -> returns list with low-confidence
  results rather than crashing (document behaviour).
- Never try to recover stroke order from an image.
