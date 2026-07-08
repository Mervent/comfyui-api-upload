from __future__ import annotations

import io
import logging
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import cv2
import numpy as np
import requests
import torch
from torch import Tensor

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="api-upload")
_ordered_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="api-upload-seq")


class UploadImage:
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

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()


class DownloadImage:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "host": ("STRING", {"default": "http://localhost:8000"}),
                "api_key": ("STRING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "item_id", "filename", "model_name")
    FUNCTION = "run"
    CATEGORY = "api/download"

    def run(self, host: str, api_key: str) -> Tuple[Tensor, int, str, str]:
        url = f"{host.rstrip('/')}/api/collections/video-render/next/"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = self._fetch(url, headers)

        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("failed to decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        item_id = int(resp.headers.get("X-Item-Id", "0"))
        filename = unquote(resp.headers.get("X-Original-Filename", ""))
        model_name = unquote(resp.headers.get("X-Model-Name", ""))
        return (image, item_id, filename, model_name)

    def _fetch(self, url: str, headers: Dict[str, str]) -> requests.Response:
        base_delay = 1.0
        max_delay = 60.0
        max_attempts = 5
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.get(url, headers=headers, timeout=120)
                if resp.status_code == 404:
                    raise RuntimeError("No images marked for video render")
                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and 400 <= exc.response.status_code < 500:
                    raise
                if attempt >= max_attempts:
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 1)
                log.warning("video-render fetch attempt %d failed (%s), retrying in %.1fs", attempt, exc, delay)
                time.sleep(delay)
            except requests.exceptions.RequestException as exc:
                if attempt >= max_attempts:
                    raise
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 1)
                log.warning("video-render fetch attempt %d failed (%s), retrying in %.1fs", attempt, exc, delay)
                time.sleep(delay)

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()


_VIDEO_CONTENT_TYPES = {"mp4": "video/mp4", "gif": "image/gif"}


class UploadVideo(UploadImage):
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "host": ("STRING", {"default": "http://localhost:8000"}),
                "collection": ("STRING",),
                "api_key": ("STRING",),
                "video": ("VHS_FILENAMES",),
            },
            "optional": {
                "source_image_id": ("INT", {"default": 0, "forceInput": True}),
                "model_name": ("STRING", {"default": ""}),
                "use_async": ("BOOLEAN", {"default": True, "forceInput": False}),
                "keep_order": ("BOOLEAN", {"default": True, "forceInput": False}),
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
        video: Tuple[bool, List[str]],
        source_image_id: int = 0,
        model_name: str = "",
        use_async: bool = True,
        keep_order: bool = True,
    ) -> Tuple[int]:
        _save_output, file_paths = video
        if not file_paths:
            return (0,)

        video_path = file_paths[-1]
        fname = os.path.basename(video_path)
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext not in _VIDEO_CONTENT_TYPES:
            raise ValueError(
                f"unsupported video format '.{ext}'; the API accepts {sorted(_VIDEO_CONTENT_TYPES)}"
            )

        url = f"{host.rstrip('/')}/api/collections/by-name/{collection}/upload/"
        headers = {"Authorization": f"Bearer {api_key}"}
        form_data = {}
        if model_name:
            form_data["model_name"] = model_name
        if source_image_id:
            form_data["source_image_id"] = str(source_image_id)

        if use_async:
            pool = _ordered_executor if keep_order else _executor
            pool.submit(self._upload_video, url, headers, video_path, form_data)
            return (1,)

        return (self._upload_video(url, headers, video_path, form_data),)

    def _upload_video(
        self,
        url: str,
        headers: Dict[str, str],
        video_path: str,
        form_data: Optional[Dict[str, str]] = None,
    ) -> int:
        fname = os.path.basename(video_path)
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        content_type = _VIDEO_CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(video_path, "rb") as f:
            buf = io.BytesIO(f.read())
        buf.seek(0)
        files = [("files", (fname, buf, content_type))]
        return self._do_upload(url, headers, files, form_data)

    @classmethod
    def IS_CHANGED(cls, *_: Any, **__: Any) -> float:
        return time.time()
