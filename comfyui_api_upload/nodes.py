from __future__ import annotations

import io
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from torch import Tensor

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xsession-upload")


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
                "use_async": (
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
        use_async: bool = True,
    ) -> Tuple[int]:
        tensors = [t for batch in (image_1, image_2, image_3, image_4, image_5) if batch is not None for t in batch]
        if not tensors:
            return (0,)

        files = []
        for tensor in tensors:
            buf = self._tensor_to_png(tensor)
            fname = f"{uuid.uuid4().hex[:12]}.png"
            files.append(("files", (fname, buf, "image/png")))

        url = f"{host.rstrip('/')}/api/collections/by-name/{collection}/upload/"
        headers = {"Authorization": f"Bearer {api_key}"}

        if use_async:
            count = len(files)
            _executor.submit(self._do_upload, url, headers, files)
            return (count,)

        return (self._do_upload(url, headers, files),)

    def _do_upload(
        self,
        url: str,
        headers: Dict[str, str],
        files: list,
    ) -> int:
        resp = requests.post(url, headers=headers, files=files, timeout=120)
        resp.raise_for_status()
        return len(resp.json())

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
