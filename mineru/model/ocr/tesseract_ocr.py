# Copyright (c) Opendatalab. All rights reserved.
"""Tesseract OCR engine integration for Sinhala (and multilingual) text recognition.

Provides a drop-in replacement for ``PytorchPaddleOCR`` / ``PPOCRv6ONNX`` when ``lang="sin"``.
Decouples text detection (via DBNet) and text recognition (via Tesseract with tessdata_best).
"""

from __future__ import annotations

import copy
import ctypes
import ctypes.util
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from .geometry import merge_det_boxes, sorted_boxes, update_det_boxes
from .image import check_img, get_rotate_crop_image_for_text_rec, preprocess_image

TESSDATA_BEST_URLS: dict[str, str] = {
    "sin": "https://github.com/tesseract-ocr/tessdata_best/raw/main/sin.traineddata",
    "eng": "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata",
}

DEFAULT_SYSTEM_TESSDATA_PATHS: tuple[str, ...] = (
    "/usr/share/tessdata",
    "/usr/share/tesseract/tessdata",
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/tessdata",
    "/usr/local/share/tessdata",
)


def _default_mineru_tessdata_dir() -> Path:
    """返回 MinerU 默认的 tessdata 本地缓存目录。"""
    cache_root = Path(os.getenv("MINERU_MODEL_DIR", Path.home() / ".cache" / "mineru" / "models"))
    tessdata_dir = cache_root / "tessdata"
    return tessdata_dir


def ensure_tessdata(
    languages: list[str] | tuple[str, ...],
    target_dir: Path | str | None = None,
) -> Path:
    """确保指定的 tesseract 语言模型存在，必要时从 tessdata_best 自动下载。"""
    # 1. 检查 TESSDATA_PREFIX 环境变量
    env_tessdata = os.getenv("TESSDATA_PREFIX")
    if env_tessdata:
        env_dir = Path(env_tessdata)
        if all((env_dir / f"{lang}.traineddata").is_file() for lang in languages):
            return env_dir

    # 2. 检查系统目录
    for sys_path in DEFAULT_SYSTEM_TESSDATA_PATHS:
        sys_dir = Path(sys_path)
        if sys_dir.is_dir() and all((sys_dir / f"{lang}.traineddata").is_file() for lang in languages):
            return sys_dir

    # 3. 本地缓存目录
    resolved_dir = Path(target_dir) if target_dir else _default_mineru_tessdata_dir()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        model_file = resolved_dir / f"{lang}.traineddata"
        if not model_file.is_file():
            url = TESSDATA_BEST_URLS.get(lang)
            if not url:
                continue
            logger.info("Downloading Tesseract best model for '{}' from {}", lang, url)
            tmp_file = resolved_dir / f"{lang}.traineddata.tmp"
            try:
                import urllib.request

                urllib.request.urlretrieve(url, tmp_file)
                tmp_file.replace(model_file)
                logger.info("Successfully downloaded {}", model_file)
            except Exception as exc:
                if tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to download tessdata_best model for '{lang}' from {url}: {exc}. "
                    f"Please manually place {lang}.traineddata into {resolved_dir}"
                ) from exc

    return resolved_dir


class _TesseractCtypesRunner:
    """基于 ctypes 的高性能在内存 Tesseract C-API 包装器。"""

    def __init__(self, tessdata_dir: str | Path, lang: str = "sin+eng") -> None:
        self.tessdata_dir = str(tessdata_dir)
        self.lang = lang
        self._lib = self._load_libtesseract()
        self._setup_c_api()
        self._api = self._init_api()

    @staticmethod
    def _load_libtesseract() -> ctypes.CDLL:
        lib_names = [
            ctypes.util.find_library("tesseract"),
            "/lib64/libtesseract.so.5.5",
            "/usr/lib/x86_64-linux-gnu/libtesseract.so.5",
            "/usr/lib64/libtesseract.so.5",
            "/usr/local/lib/libtesseract.so",
            "libtesseract.so",
            "libtesseract.so.5",
        ]
        for name in lib_names:
            if not name:
                continue
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise RuntimeError("libtesseract shared library could not be loaded via ctypes.")

    def _setup_c_api(self) -> None:
        lib = self._lib
        lib.TessBaseAPICreate.restype = ctypes.c_void_p
        lib.TessBaseAPICreate.argtypes = []

        lib.TessBaseAPIInit3.restype = ctypes.c_int
        lib.TessBaseAPIInit3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]

        lib.TessBaseAPISetImage.restype = None
        lib.TessBaseAPISetImage.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]

        lib.TessBaseAPISetVariable.restype = ctypes.c_int
        lib.TessBaseAPISetVariable.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]

        lib.TessBaseAPIGetUTF8Text.restype = ctypes.c_void_p
        lib.TessBaseAPIGetUTF8Text.argtypes = [ctypes.c_void_p]

        lib.TessBaseAPIMeanTextConf.restype = ctypes.c_int
        lib.TessBaseAPIMeanTextConf.argtypes = [ctypes.c_void_p]

        lib.TessBaseAPIDelete.restype = None
        lib.TessBaseAPIDelete.argtypes = [ctypes.c_void_p]

        if hasattr(lib, "TessBaseAPISetSourceResolution"):
            lib.TessBaseAPISetSourceResolution.restype = None
            lib.TessBaseAPISetSourceResolution.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
            ]

        if hasattr(lib, "TessBaseAPIClear"):
            lib.TessBaseAPIClear.restype = None
            lib.TessBaseAPIClear.argtypes = [ctypes.c_void_p]

        if hasattr(lib, "TessDeleteText"):
            lib.TessDeleteText.restype = None
            lib.TessDeleteText.argtypes = [ctypes.c_void_p]

    def _init_api(self) -> ctypes.c_void_p:
        api = self._lib.TessBaseAPICreate()
        if not api:
            raise RuntimeError("TessBaseAPICreate returned NULL")
        datapath_bytes = self.tessdata_dir.encode("utf-8")
        lang_bytes = self.lang.encode("utf-8")
        ret = self._lib.TessBaseAPIInit3(api, datapath_bytes, lang_bytes)
        if ret != 0:
            self._lib.TessBaseAPIDelete(api)
            raise RuntimeError(
                f"TessBaseAPIInit3 failed with code {ret} for datapath='{self.tessdata_dir}', lang='{self.lang}'"
            )
        # 默认使用单行文本模式 (PSM_SINGLE_LINE = 7) 进行文字行识别
        self._lib.TessBaseAPISetVariable(api, b"tessedit_pageseg_mode", b"7")
        return api

    def recognize(self, img: np.ndarray) -> tuple[str, float]:
        """识别单个文本行切片并返回 (text, confidence)。"""
        if img is None or img.size == 0 or min(img.shape[:2]) == 0:
            return "", 0.0

        # 为切片边缘补齐 5 像素白色边距，防止僧伽罗语元音上下标被 Leptonica 边缘二值化切除
        img_padded = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        if len(img_padded.shape) == 2:
            height, width = img_padded.shape
            bytes_per_pixel = 1
            contiguous_data = np.ascontiguousarray(img_padded, dtype=np.uint8)
        else:
            height, width = img_padded.shape[:2]
            # OpenCV 为 BGR，Tesseract 原生要求 RGB
            rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
            bytes_per_pixel = 3
            contiguous_data = np.ascontiguousarray(rgb, dtype=np.uint8)

        bytes_per_line = width * bytes_per_pixel
        raw_ptr = contiguous_data.ctypes.data_as(ctypes.c_char_p)

        self._lib.TessBaseAPISetImage(self._api, raw_ptr, width, height, bytes_per_pixel, bytes_per_line)
        if hasattr(self._lib, "TessBaseAPISetSourceResolution"):
            self._lib.TessBaseAPISetSourceResolution(self._api, 300)
        text_ptr = self._lib.TessBaseAPIGetUTF8Text(self._api)
        conf_int = self._lib.TessBaseAPIMeanTextConf(self._api)

        text = ""
        if text_ptr:
            raw_bytes = ctypes.string_at(text_ptr)
            text = raw_bytes.decode("utf-8", errors="replace").strip()
            if hasattr(self._lib, "TessDeleteText"):
                self._lib.TessDeleteText(text_ptr)

        if hasattr(self._lib, "TessBaseAPIClear"):
            self._lib.TessBaseAPIClear(self._api)

        score = max(0.0, min(1.0, float(conf_int) / 100.0))
        return text, score

    def close(self) -> None:
        if getattr(self, "_api", None):
            self._lib.TessBaseAPIDelete(self._api)
            self._api = None

    def __del__(self) -> None:
        self.close()


class _TesseractSubprocessRunner:
    """基于 Tesseract CLI 的稳定回退包装器。"""

    def __init__(self, tessdata_dir: str | Path, lang: str = "sin+eng") -> None:
        self.tessdata_dir = str(tessdata_dir)
        self.lang = lang
        self._executable = shutil.which("tesseract") or "/usr/bin/tesseract"
        if not os.path.isfile(self._executable):
            raise RuntimeError(f"Tesseract executable '{self._executable}' not found.")

    def recognize(self, img: np.ndarray) -> tuple[str, float]:
        if img is None or img.size == 0 or min(img.shape[:2]) == 0:
            return "", 0.0

        img_padded = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        success, encoded = cv2.imencode(".png", img_padded)
        if not success:
            return "", 0.0

        cmd = [
            self._executable,
            "stdin",
            "stdout",
            "-l",
            self.lang,
            "--tessdata-dir",
            self.tessdata_dir,
            "--dpi",
            "300",
            "--psm",
            "7",
            "tsv",
        ]
        proc = subprocess.run(cmd, input=encoded.tobytes(), capture_output=True, check=False)
        if proc.returncode != 0:
            return "", 0.0

        output = proc.stdout.decode("utf-8", errors="replace")
        words: list[str] = []
        confs: list[float] = []
        for line in output.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 12:
                word = parts[11].strip()
                conf_val = float(parts[10])
                if word and conf_val >= 0:
                    words.append(word)
                    confs.append(conf_val)

        text = " ".join(words).strip()
        score = (sum(confs) / (len(confs) * 100.0)) if confs else 0.0
        return text, min(1.0, max(0.0, score))


class TesseractRecognizer:
    """与 PaddleOCR ``TextRecognizer`` 接口对齐的 Tesseract 识别器。"""

    def __init__(self, runner: _TesseractCtypesRunner | _TesseractSubprocessRunner) -> None:
        self.runner = runner

    def __call__(
        self,
        img_crop_list: list[np.ndarray],
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
        tqdm_progress_bar: Any = None,
    ) -> tuple[list[tuple[str, float]], float]:
        t0 = time.perf_counter()
        results: list[tuple[str, float]] = []

        iterator = enumerate(img_crop_list)
        if tqdm_enable and not tqdm_progress_bar:
            iterator = enumerate(tqdm(img_crop_list, desc=tqdm_desc))

        for _, crop in iterator:
            text, score = self.runner.recognize(crop)
            results.append((text, score))

        elapse = time.perf_counter() - t0
        return results, elapse


class TesseractOCR:
    """针对僧伽罗语 (sin) 的 Tesseract OCR 封装。

    对外公开接口与 ``PytorchPaddleOCR`` / ``PPOCRv6ONNX`` 保持一致：
    - ``ocr(img, det=..., rec=...)``
    - ``__call__(img, mfd_res=None)``
    - ``self.text_detector``: 复用 DBNet 检测器以确保精确的文本行多边形定位
    - ``self.text_recognizer``: 使用 Tesseract (sin+eng tessdata_best) 进行行切片转写
    """

    def __init__(
        self,
        *args: Any,
        lang: str = "sin+eng",
        device: str | None = None,
        drop_score: float = 0.0,
        enable_merge_det_boxes: bool = True,
        tessdata_dir: Path | str | None = None,
        text_detector: Any = None,
        **kwargs: Any,
    ) -> None:
        self.lang = lang
        self.device = device or "cpu"
        self.drop_score = drop_score
        self.enable_merge_det_boxes = enable_merge_det_boxes
        self.is_seal = False

        # 确保 sin 和 eng 模型可用
        req_langs = [part.strip() for part in lang.split("+") if part.strip()]
        self.resolved_tessdata_dir = ensure_tessdata(req_langs, target_dir=tessdata_dir)

        # 优先使用高性能 ctypes 绑定，不可用时回退到 CLI
        try:
            self.runner: _TesseractCtypesRunner | _TesseractSubprocessRunner = _TesseractCtypesRunner(
                self.resolved_tessdata_dir, lang=lang
            )
            logger.debug("TesseractOCR initialized via libtesseract ctypes runner.")
        except Exception as exc:
            logger.warning(
                "libtesseract ctypes initialization failed ({}). Falling back to subprocess CLI.",
                exc,
            )
            self.runner = _TesseractSubprocessRunner(self.resolved_tessdata_dir, lang=lang)

        self.text_recognizer = TesseractRecognizer(self.runner)

        # 初始化文本检测器 (DBNet)
        if text_detector is not None:
            self.text_detector = text_detector
        else:
            self.text_detector = self._init_default_text_detector(device=self.device, **kwargs)

    def _init_default_text_detector(self, device: str | None = None, **kwargs: Any) -> Any:
        """根据小模型后端选择 DBNet 检测器，不依赖 Paddle 识别权重。"""
        small_backend = kwargs.get("small_backend")
        if small_backend == "onnx":
            from .pp_ocr_v6_onnx import TextDetectorONNX
            from ..registry import MINERU_4_MODELS_ONNX

            det_model_path = str(MINERU_4_MODELS_ONNX.ocr_det.ensure())
            return TextDetectorONNX(
                model_path=det_model_path,
                device=device,
                box_thresh=kwargs.get("det_db_box_thresh", 0.5),
                unclip_ratio=kwargs.get("det_db_unclip_ratio", 1.5),
            )

        from .._internal.pytorchocr.infer import (
            predict_det,
            pytorchocr_utility as utility,
        )
        from ..registry import MINERU_4_MODELS_TORCH
        import argparse

        parser = utility.init_args()
        parsed_args = parser.parse_args([])
        arg_dict = vars(parsed_args)
        arg_dict["device"] = device or "cpu"
        det_path = str(MINERU_4_MODELS_TORCH.pytorch_paddle.path("ch_PP-OCRv6_tiny_det_infer.safetensors").ensure())
        arg_dict["det_model_path"] = det_path
        arg_dict["det_db_box_thresh"] = kwargs.get("det_db_box_thresh", 0.5)
        arg_dict["det_db_unclip_ratio"] = kwargs.get("det_db_unclip_ratio", 1.5)
        arg_dict.update(kwargs)
        return predict_det.TextDetector(argparse.Namespace(**arg_dict))

    def ocr(
        self,
        img: np.ndarray | list[np.ndarray] | str | bytes,
        det: bool = True,
        rec: bool = True,
        mfd_res: list[Any] | None = None,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
        tqdm_progress_bar: Any = None,
    ) -> list[Any]:
        """与 PytorchPaddleOCR.ocr 兼容的统一推理入口。"""
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det:
            logger.error("When input a list of images, det must be false")
            return [None]

        if isinstance(img, list) and not det and rec:
            # 批量切片识别 (如 _apply_ocr_rec_results, span post-ocr)
            rec_res, _elapse = self.text_recognizer(
                img,
                tqdm_enable=tqdm_enable,
                tqdm_desc=tqdm_desc,
                tqdm_progress_bar=tqdm_progress_bar,
            )
            return [rec_res]

        img_checked = check_img(img)
        imgs = [img_checked]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if det and rec:
                # 完整 det + crop + rec (如 table_ocr_model)
                ocr_res: list[Any] = []
                for cur_img in imgs:
                    cur_img = preprocess_image(cur_img)
                    dt_boxes, rec_res = self.__call__(cur_img, mfd_res=mfd_res)
                    if not dt_boxes and not rec_res:
                        ocr_res.append(None)
                        continue
                    tmp_res = [[box.tolist() if hasattr(box, "tolist") else box, res] for box, res in zip(dt_boxes, rec_res)]
                    ocr_res.append(tmp_res)
                return ocr_res

            elif det and not rec:
                ocr_res = []
                for cur_img in imgs:
                    cur_img = preprocess_image(cur_img)
                    dt_boxes, _elapse = self.text_detector(cur_img)
                    if dt_boxes is None:
                        ocr_res.append(None)
                        continue
                    dt_boxes = sorted_boxes(dt_boxes)
                    if self.enable_merge_det_boxes:
                        dt_boxes = merge_det_boxes(dt_boxes)
                    if mfd_res:
                        dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                    tmp_res = [box.tolist() if hasattr(box, "tolist") else box for box in dt_boxes]
                    ocr_res.append(tmp_res)
                return ocr_res

            elif not det and rec:
                ocr_res = []
                for cur_img in imgs:
                    if not isinstance(cur_img, list):
                        cur_img = preprocess_image(cur_img)
                        cur_img = [cur_img]
                    rec_res, _elapse = self.text_recognizer(
                        cur_img,
                        tqdm_enable=tqdm_enable,
                        tqdm_desc=tqdm_desc,
                        tqdm_progress_bar=tqdm_progress_bar,
                    )
                    ocr_res.append(rec_res)
                return ocr_res

        return []

    def __call__(self, img: np.ndarray, mfd_res: list[Any] | None = None) -> tuple[list[Any], list[Any]]:
        if img is None:
            return [], []

        ori_im = img
        dt_boxes, _elapse = self.text_detector(img)
        if dt_boxes is None or len(dt_boxes) == 0:
            return [], []

        dt_boxes = sorted_boxes(dt_boxes)
        if self.enable_merge_det_boxes:
            dt_boxes = merge_det_boxes(dt_boxes)
        if mfd_res:
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)

        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image_for_text_rec(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, _elapse = self.text_recognizer(img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


__all__ = ["TesseractOCR", "TesseractRecognizer", "ensure_tessdata"]
