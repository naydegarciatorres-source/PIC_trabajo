"""Neural network architectures and checkpoint utilities for Quick, Draw! recognition.

This module only depends on PyTorch (it must not import the data/training modules) and
implements section 4 of ``docs/CONTRACT.md``:

* :class:`QuickDrawCNN` - image-based baseline CNN, input ``(B, 1, H, W)`` in ``[0, 1]``.
* :class:`StrokeBiLSTM` - stroke-sequence model, input ``(B, T, 3)`` = ``(dx, dy, end_of_stroke)``.
* :func:`build_model`, :data:`MODEL_REGISTRY`, :func:`count_parameters`.
* :func:`save_checkpoint`, :func:`load_checkpoint`, :func:`load_model_from_checkpoint`.

The number of classes is always a constructor argument (never hard-coded), so moving from
45 to all 345 Quick, Draw! categories only requires changing ``num_classes``.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = [
    "QuickDrawCNN",
    "StrokeBiLSTM",
    "MODEL_REGISTRY",
    "CHECKPOINT_FORMAT_VERSION",
    "build_model",
    "count_parameters",
    "unwrap_model",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
]

CHECKPOINT_FORMAT_VERSION = 1


# --------------------------------------------------------------------------------------
# Image model
# --------------------------------------------------------------------------------------
class _ConvBlock(nn.Sequential):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2 followed by 2x2 max pooling (halves H and W)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class QuickDrawCNN(nn.Module):
    """Baseline CNN for rasterised Quick, Draw! sketches.

    Architecture: 4 convolutional blocks (each ``Conv-BN-ReLU x2 + MaxPool``) with
    ``channels`` (default 32-64-128-256), then ``AdaptiveAvgPool2d(1)`` -> dropout -> Linear.
    Thanks to the global adaptive pooling, any square input with ``H = W >= 28`` works
    (28 -> 14 -> 7 -> 3 -> 1 spatial size at the deepest block for the smallest input).
    About 1.2M parameters: trainable on a Colab T4 and real time on a laptop CPU.

    Args:
        num_classes: number of output classes (``len(classes)``).
        in_channels: input channels (1 for grayscale canvases).
        dropout: dropout probability before the classifier.
        channels: output channels of the four conv blocks.

    Input:  float tensor ``(B, in_channels, H, W)`` with values in ``[0, 1]``
            (white strokes on black background, as produced by ``preprocess``).
    Output: logits ``(B, num_classes)`` (no softmax).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        dropout: float = 0.3,
        channels: Sequence[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        channels = tuple(int(c) for c in channels)
        if len(channels) == 0:
            raise ValueError("channels must be non-empty")

        blocks = []
        prev = self.in_channels
        for ch in channels:
            blocks.append(_ConvBlock(prev, ch))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev, self.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits ``(B, num_classes)`` for images ``(B, C, H, W)``."""
        if x.dim() == 3:  # (B, H, W) -> (B, 1, H, W) convenience
            x = x.unsqueeze(1)
        if x.dim() != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                f"expected input of shape (B, {self.in_channels}, H, W), got {tuple(x.shape)}"
            )
        x = self.features(x.float())
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


# --------------------------------------------------------------------------------------
# Stroke model
# --------------------------------------------------------------------------------------
class StrokeBiLSTM(nn.Module):
    """Stroke-sequence classifier: Conv1d stack -> bidirectional LSTM -> masked sum -> Linear.

    Conceptually *inspired by* the official TensorFlow tutorial "Recurrent Neural Networks
    for Drawing Classification" (Quick, Draw!): it is a re-implementation in PyTorch with
    our own choices, **not** a reproduction of Google's tutorial code nor of the model used
    in the original Quick, Draw! game.

    Input features per timestep are ``(dx, dy, end_of_stroke)``: deltas between consecutive
    points of the bounding-box-normalised drawing, and a flag that is 1 on the last point of
    each stroke - this third channel is what encodes the separation between strokes.

    Pipeline:
        1. 1-D convolutions over time (default 48, 64, 96 channels with kernels 5, 5, 3,
           "same" padding, ReLU, optional BatchNorm, dropout).
        2. ``num_layers`` stacked bidirectional LSTM layers fed with
           ``pack_padded_sequence(..., enforce_sorted=False)`` so padding is never processed.
        3. Sum of the LSTM outputs over the *valid* timesteps only (masked sum).
        4. Dropout -> Linear -> logits.

    Padding invariance: padded positions are zeroed before every convolution, so the conv
    output at valid timesteps equals the output for the unpadded sequence (zero "same"
    padding); the LSTM never sees the padding (packed) and the sum is masked. Hence, in eval
    mode, logits do not depend on the padding content or the padded length.
    Note: if ``use_batchnorm=True``, training-mode batch statistics include padded zeros
    (eval mode stays padding invariant); it is therefore disabled by default.

    Args:
        num_classes: number of output classes.
        input_dim: features per timestep (3).
        conv_channels: output channels of the Conv1d layers.
        conv_kernels: kernel sizes (odd) of the Conv1d layers, same length as conv_channels.
        hidden_size: LSTM hidden size per direction (output feature size is 2*hidden_size).
        num_layers: number of stacked bidirectional LSTM layers.
        dropout: dropout used after convs, between LSTM layers and before the classifier.
        use_batchnorm: add BatchNorm1d after each convolution.

    ``forward(x, lengths=None)``: ``x`` float ``(B, T, input_dim)``, ``lengths`` long ``(B,)``
    with values in ``[1, T]`` (``None`` = all sequences have full length ``T``).
    Returns logits ``(B, num_classes)``.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 3,
        conv_channels: Sequence[int] = (48, 64, 96),
        conv_kernels: Sequence[int] = (5, 5, 3),
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        conv_channels = tuple(int(c) for c in conv_channels)
        conv_kernels = tuple(int(k) for k in conv_kernels)
        if len(conv_channels) != len(conv_kernels):
            raise ValueError("conv_channels and conv_kernels must have the same length")
        if any(k % 2 == 0 for k in conv_kernels):
            raise ValueError("conv_kernels must be odd for 'same' padding")

        self.num_classes = int(num_classes)
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.convs = nn.ModuleList()
        prev = self.input_dim
        for ch, k in zip(conv_channels, conv_kernels):
            layers: list[nn.Module] = [nn.Conv1d(prev, ch, k, padding=k // 2, bias=not use_batchnorm)]
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(ch))
            layers += [nn.ReLU(inplace=True), nn.Dropout(dropout)]
            self.convs.append(nn.Sequential(*layers))
            prev = ch

        self.lstm = nn.LSTM(
            input_size=prev,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * self.hidden_size, self.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                for i in range(4):  # orthogonal init per gate block (i, f, g, o)
                    nn.init.orthogonal_(p.data[i * self.hidden_size:(i + 1) * self.hidden_size])
            elif "bias" in name:
                nn.init.zeros_(p)
                # forget-gate bias = 1 (split between bias_ih and bias_hh -> 0.5 each)
                p.data[self.hidden_size:2 * self.hidden_size].fill_(0.5)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return logits ``(B, num_classes)`` for padded sequences ``(B, T, input_dim)``."""
        if x.dim() != 3 or x.size(2) != self.input_dim:
            raise ValueError(f"expected input (B, T, {self.input_dim}), got {tuple(x.shape)}")
        x = x.float()
        batch, total_t, _ = x.shape
        if total_t < 1:
            raise ValueError("sequences must have at least one timestep")

        if lengths is None:
            lengths_cpu = torch.full((batch,), total_t, dtype=torch.long)
        else:
            lengths_cpu = torch.as_tensor(lengths).detach().to("cpu", torch.long).reshape(-1)
            if lengths_cpu.numel() != batch:
                raise ValueError(f"lengths has {lengths_cpu.numel()} entries, batch is {batch}")
            lengths_cpu = lengths_cpu.clamp(1, total_t)

        # (B, T) mask of valid timesteps.
        mask = torch.arange(total_t).unsqueeze(0) < lengths_cpu.unsqueeze(1)
        mask = mask.to(x.device)
        mask_c = mask.unsqueeze(1).to(x.dtype)  # (B, 1, T) for channel-first tensors

        h = x.transpose(1, 2)  # (B, C, T)
        for conv in self.convs:
            h = conv(h * mask_c)  # zero padding before each conv -> padding invariant
        h = (h * mask_c).transpose(1, 2).contiguous()  # (B, T, C)

        packed = pack_padded_sequence(h, lengths_cpu, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=total_t)  # (B, T, 2H)

        pooled = (out * mask.unsqueeze(2).to(out.dtype)).sum(dim=1)  # masked sum over time
        return self.classifier(self.dropout(pooled))


# --------------------------------------------------------------------------------------
# Registry / helpers
# --------------------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "cnn": QuickDrawCNN,
    "bilstm": StrokeBiLSTM,
}


def build_model(name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    """Instantiate a model from :data:`MODEL_REGISTRY` (``"cnn"`` or ``"bilstm"``).

    Raises:
        ValueError: if ``name`` is not a registered model.
    """
    key = str(name).lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name {name!r}; available: {sorted(MODEL_REGISTRY)}")
    kwargs.pop("num_classes", None)
    return MODEL_REGISTRY[key](num_classes=int(num_classes), **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of (trainable by default) parameters of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip ``torch.compile`` / ``DataParallel`` / ``DistributedDataParallel`` wrappers."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod  # type: ignore[assignment]
        elif isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module
        else:
            return model


# --------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------
def save_checkpoint(
    path: str,
    model: nn.Module,
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]],
    classes: Sequence[str],
    preprocessing: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    history: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save everything needed to rebuild and resume ``model`` into ``path``.

    The file is a ``torch.save`` dict with keys ``format_version, model_name, model_kwargs,
    num_classes, classes, preprocessing, state_dict, optimizer_state, scheduler_state, epoch,
    best_metric, history, extra``. ``model_kwargs`` are the constructor kwargs other than
    ``num_classes`` so that :func:`load_model_from_checkpoint` can rebuild the model with
    no extra information. The write is atomic (temporary file in the same directory, then
    ``os.replace``), so an interrupted save never corrupts an existing checkpoint.
    """
    base = unwrap_model(model)
    classes = list(classes)
    num_classes = int(getattr(base, "num_classes", len(classes)))
    if len(classes) != num_classes:
        raise ValueError(f"len(classes)={len(classes)} != model num_classes={num_classes}")
    kwargs = dict(model_kwargs or {})
    kwargs.pop("num_classes", None)

    ckpt: Dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": str(model_name).lower(),
        "model_kwargs": kwargs,
        "num_classes": num_classes,
        "classes": classes,
        "preprocessing": dict(preprocessing or {}),
        "state_dict": base.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_metric": best_metric,
        "history": history,
        "extra": extra,
    }

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(ckpt, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint dict saved by :func:`save_checkpoint`.

    Uses ``weights_only=False`` because checkpoints contain plain Python metadata
    (classes, history...). Only load trusted, locally produced files.
    """
    ckpt = torch.load(os.fspath(path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "model_name" not in ckpt:
        raise ValueError(f"{path} is not a valid QuickDraw checkpoint")
    return ckpt


def load_model_from_checkpoint(
    path: str, device: Any = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the model stored in ``path``, load its weights, move it to ``device``.

    Returns:
        ``(model.eval(), ckpt_dict)``.
    """
    ckpt = load_checkpoint(path, map_location="cpu")
    kwargs = dict(ckpt.get("model_kwargs") or {})
    num_classes = int(ckpt.get("num_classes", len(ckpt.get("classes", []))))
    model = build_model(ckpt["model_name"], num_classes, **kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch.device(device) if not isinstance(device, torch.device) else device)
    model.eval()
    return model, ckpt
