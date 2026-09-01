"""Stream pipeline diagnostics to disk without retaining full frame stacks."""

import re
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


RGBColor = Tuple[int, int, int]

NAN_COLOR: RGBColor = (255, 0, 255)
POS_INF_COLOR: RGBColor = (255, 0, 0)
NEG_INF_COLOR: RGBColor = (0, 0, 255)


class DebugWriter:
    """Render and save numbered debug images, one frame at a time."""

    _FRAME_PATTERN = re.compile(r"frame_(\d+)\.png")
    _CATEGORY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

    def __init__(self, root: Union[str, Path] = "debug") -> None:
        self.root = Path(root)
        self._next_ids: Dict[str, int] = {}

    def write_rgb(self, category: str, frame: NDArray) -> Path:
        """Save a one- or three-channel image whose finite range is [0, 1]."""
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = frame[..., None]
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
            raise ValueError(
                "RGB debug frames must have shape (height, width), "
                "(height, width, 1), or (height, width, 3)"
            )

        nan_mask, pos_inf_mask, neg_inf_mask = self._invalid_masks(frame)
        rgb = self._to_uint8(frame, lower=0, upper=1)
        if rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        self._paint_invalid(rgb, nan_mask, pos_inf_mask, neg_inf_mask)
        return self._write(category, rgb)

    def write_scalar(
        self,
        category: str,
        values: NDArray,
        value_range: Tuple[float, float] = (0, 1),
        colormap: int = cv2.COLORMAP_VIRIDIS,
    ) -> Path:
        """Render a scalar field with a colormap and save it as RGB."""
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("Scalar debug frames must have shape (height, width)")

        nan_mask, pos_inf_mask, neg_inf_mask = self._invalid_masks(values)
        grayscale = self._to_uint8(values, *value_range)
        bgr = cv2.applyColorMap(grayscale, colormap)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._paint_invalid(rgb, nan_mask, pos_inf_mask, neg_inf_mask)
        return self._write(category, rgb)

    def write_flow(self, category: str, flow: NDArray) -> Path:
        """Render a ``(dx, dy)`` flow field with the conventional HSV map.

        Direction is encoded as hue and per-frame normalized magnitude as
        brightness. Zero motion is black.
        """
        flow = np.asarray(flow)
        if flow.ndim != 3 or flow.shape[-1] != 2:
            raise ValueError("Optical flow must have shape (height, width, 2)")

        nan_mask, pos_inf_mask, neg_inf_mask = self._invalid_masks(flow)
        finite_flow = np.nan_to_num(flow, nan=0, posinf=0, neginf=0)
        dx = finite_flow[..., 0].astype(np.float32, copy=False)
        dy = finite_flow[..., 1].astype(np.float32, copy=False)
        magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = np.floor(np.mod(angle, 360) / 2).astype(np.uint8)
        hsv[..., 1] = 255
        max_magnitude = float(np.max(magnitude)) if magnitude.size else 0
        if max_magnitude > 0:
            hsv[..., 2] = np.round(
                np.clip(magnitude / max_magnitude, 0, 1) * 255
            ).astype(np.uint8)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        self._paint_invalid(rgb, nan_mask, pos_inf_mask, neg_inf_mask)
        return self._write(category, rgb)

    @staticmethod
    def _invalid_masks(values: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        nan_mask = np.isnan(values)
        pos_inf_mask = np.isposinf(values)
        neg_inf_mask = np.isneginf(values)
        if values.ndim == 3:
            nan_mask = nan_mask.any(axis=-1)
            pos_inf_mask = pos_inf_mask.any(axis=-1)
            neg_inf_mask = neg_inf_mask.any(axis=-1)

        # Give NaN precedence, followed by positive and then negative infinity.
        pos_inf_mask &= ~nan_mask
        neg_inf_mask &= ~nan_mask & ~pos_inf_mask
        return nan_mask, pos_inf_mask, neg_inf_mask

    @staticmethod
    def _to_uint8(values: NDArray, lower: float, upper: float) -> NDArray:
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("The debug value range must contain two finite increasing values")
        values = np.nan_to_num(values, nan=lower, posinf=upper, neginf=lower)
        normalized = np.clip((values - lower) / (upper - lower), 0, 1)
        return np.round(normalized * 255).astype(np.uint8)

    @staticmethod
    def _paint_invalid(
        rgb: NDArray,
        nan_mask: NDArray,
        pos_inf_mask: NDArray,
        neg_inf_mask: NDArray,
    ) -> None:
        rgb[nan_mask] = NAN_COLOR
        rgb[pos_inf_mask] = POS_INF_COLOR
        rgb[neg_inf_mask] = NEG_INF_COLOR

    def _write(self, category: str, rgb: NDArray) -> Path:
        path = self._next_path(category)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise OSError(f"Could not write debug frame to {path}")
        return path

    def _next_path(self, category: str) -> Path:
        if not self._CATEGORY_PATTERN.fullmatch(category):
            raise ValueError(f"Invalid debug category: {category!r}")

        directory = self.root / category
        directory.mkdir(parents=True, exist_ok=True)

        if category not in self._next_ids:
            frame_ids = []
            for path in directory.iterdir():
                match = self._FRAME_PATTERN.fullmatch(path.name)
                if match:
                    frame_ids.append(int(match.group(1)))
            self._next_ids[category] = max(frame_ids, default=-1) + 1

        frame_id = self._next_ids[category]
        self._next_ids[category] += 1
        return directory / f"frame_{frame_id:04d}.png"
