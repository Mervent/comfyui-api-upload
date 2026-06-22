from __future__ import annotations

import io
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from torch import Tensor

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xsession-upload")
_ordered_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xsession-upload-seq")


class APIUpload:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "host": ("STRING", {"default": "http://localhost:8000"}),
                "collection": ("STRING",),
                "api_key": ("STRING",),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "model_name": ("STRING", {"default": ""}),
                "use_async": (
                    "BOOLEAN",
                    {"default": True, "forceInput": False},
                ),
                "keep_order": (
                    "BOOLEAN",
                    {"default": True, "forceInput": False},
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("upload_count",)
    FUNCTION = "run"
    CATEGORY = "api/upload"
    OUTPUT_NODE = True

    def run(
        self,
        host: str,
        collection: str,
        api_key: str,
        image_1: Optional[List[Tensor]] = None,
        image_2: Optional[List[Tensor]] = None,
        image_3: Optional[List[Tensor]] = None,
        image_4: Optional[List[Tensor]] = None,
        image_5: Optional[List[Tensor]] = None,
        model_name: str = "",
        use_async: bool = True,
        keep_order: bool = True,
    ) -> Tuple[int]:
        tensors = [t for batch in (image_1, image_2, image_3, image_4, image_5) if batch is not None for t in batch]
        if not tensors:
            return (0,)

        url = f"{host.rstrip('/')}/api/collections/by-name/{collection}/upload/"
        headers = {"Authorization": f"Bearer {api_key}"}
        form_data = {"model_name": model_name} if model_name else {}

        if use_async:
            arrays = [self._tensor_to_array(t) for t in tensors]
            pool = _ordered_executor if keep_order else _executor
            pool.submit(self._encode_and_upload, url, headers, arrays, keep_order, form_data)
            return (len(tensors),)

        arrays = [self._tensor_to_array(t) for t in tensors]
        if keep_order:
            return (self._upload_sequential(url, headers, arrays, form_data),)
        files = self._encode_arrays(arrays)
        return (self._do_upload(url, headers, files, form_data),)

    def _tensor_to_array(self, t: Tensor) -> np.ndarray:
        if t.ndim == 4:
            t = t.squeeze(0)
        u8 = t.mul(255).clamp_(0, 255).to(torch.uint8)
        if u8.shape[0] not in (1, 3):
            u8 = u8.permute(2, 0, 1).contiguous()
        h = torch.empty_like(u8, device="cpu", pin_memory=True)
        h.copy_(u8, non_blocking=True)
        return h.permute(1, 2, 0).contiguous().numpy()

    def _encode_arrays(self, arrays: List[np.ndarray]) -> list:
        files = []
        for arr in arrays:
            ok, enc = cv2.imencode(".png", arr[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 2])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            buf = io.BytesIO(enc.tobytes())
            buf.seek(0)
            fname = f"{uuid.uuid4().hex[:12]}.png"
            files.append(("files", (fname, buf, "image/png")))
        return files

    def _upload_sequential(
        self, url: str, headers: Dict[str, str], arrays: List[np.ndarray],
        form_data: Optional[Dict[str, str]] = None,
    ) -> int:
        count = 0
        for arr in arrays:
            files = self._encode_arrays([arr])
            count += self._do_upload(url, headers, files, form_data)
        return count

    def _encode_and_upload(
        self, url: str, headers: Dict[str, str], arrays: List[np.ndarray],
        keep_order: bool = False,
        form_data: Optional[Dict[str, str]] = None,
    ) -> int:
        if keep_order:
            return self._upload_sequential(url, headers, arrays, form_data)
        files = self._encode_arrays(arrays)
        return self._do_upload(url, headers, files, form_data)

    def _do_upload(
        self,
        url: str,
        headers: Dict[str, str],
        files: list,
        form_data: Optional[Dict[str, str]] = None,
    ) -> int:
        base_delay = 1.0
        max_delay = 60.0
        attempt = 0

        while True:
            attempt += 1
            try:
                for _, (_, buf, _) in files:
                    buf.seek(0)
                resp = requests.post(url, headers=headers, files=files, data=form_data or {}, timeout=120)
                resp.raise_for_status()
                if attempt > 1:
                    log.info("Upload succeeded on attempt %d", attempt)
                return len(resp.json())
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and 400 <= exc.response.status_code < 500:
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 1)
                log.warning("Upload attempt %d failed (%s), retrying in %.1fs", attempt, exc, delay)
                time.sleep(delay)
            except requests.exceptions.RequestException as exc:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 1)
                log.warning("Upload attempt %d failed (%s), retrying in %.1fs", attempt, exc, delay)
                time.sleep(delay)

    def _tensor_to_png(self, t: Tensor) -> io.BytesIO:
        if t.ndim == 4:
            t = t.squeeze(0)
        arr = t.cpu().mul(255).clamp(0, 255).to(torch.uint8).numpy()
        img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return buf

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()
