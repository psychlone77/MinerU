# Copyright (c) Opendatalab. All rights reserved.
"""Unit tests for Sinhala text extraction with Tesseract OCR integration."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np

from mineru.model.ocr.language import normalize_ocr_model_lang
from mineru.model.ocr.tesseract_ocr import TesseractOCR, TesseractRecognizer
from mineru.model.runtime.contracts import AtomicModelName
from mineru.model.runtime import hybrid
from mineru.parser.api_server import CreateJobRequest
from mineru.parser.api_client import MinerUApiParser
from mineru.parser.mineru_parser import MinerUParser


def test_normalize_ocr_model_lang_sinhala() -> None:
    """验证僧伽罗语各种别名正确归一化为 'sin'。"""
    assert normalize_ocr_model_lang("sin") == "sin"
    assert normalize_ocr_model_lang("sinhala") == "sin"
    assert normalize_ocr_model_lang("si") == "sin"
    assert normalize_ocr_model_lang("SIN") == "sin"
    assert normalize_ocr_model_lang(" Sinhala ") == "sin"


def test_normalize_ocr_model_lang_defaults_and_others() -> None:
    """验证默认值及其他语言别名行为不受影响。"""
    assert normalize_ocr_model_lang(None) == "ch"
    assert normalize_ocr_model_lang("en") == "ch"
    assert normalize_ocr_model_lang("ch") == "ch"
    assert normalize_ocr_model_lang("ru") == "east_slavic"


def test_tesseract_recognizer_calls_runner() -> None:
    """验证 TesseractRecognizer 正确包装 runner 并返回 (text, score) 列表。"""
    mock_runner = Mock()
    mock_runner.recognize.side_effect = [
        ("ශ්‍රී ලංකා", 0.95),
        ("English", 0.92),
    ]
    recognizer = TesseractRecognizer(mock_runner)
    img1 = np.zeros((30, 100, 3), dtype=np.uint8)
    img2 = np.zeros((30, 80, 3), dtype=np.uint8)

    results, elapse = recognizer([img1, img2])
    assert len(results) == 2
    assert results[0] == ("ශ්‍රී ලංකා", 0.95)
    assert results[1] == ("English", 0.92)
    assert elapse >= 0.0


def test_tesseract_ocr_rec_only_mode() -> None:
    """验证 det=False, rec=True 批量切片转写模式。"""
    mock_runner = Mock()
    mock_runner.recognize.return_value = ("හෙලෝ", 0.98)

    with (
        patch("mineru.model.ocr.tesseract_ocr.ensure_tessdata") as mock_ensure,
        patch(
            "mineru.model.ocr.tesseract_ocr._TesseractCtypesRunner",
            return_value=mock_runner,
        ),
    ):
        mock_ensure.return_value = "/mock/tessdata"
        ocr = TesseractOCR(lang="sin+eng", text_detector=Mock())
        crops = [
            np.zeros((20, 50, 3), dtype=np.uint8),
            np.zeros((20, 50, 3), dtype=np.uint8),
        ]
        res = ocr.ocr(crops, det=False, rec=True)

        assert len(res) == 1
        assert len(res[0]) == 2
        assert res[0][0] == ("හෙලෝ", 0.98)
        assert res[0][1] == ("හෙලෝ", 0.98)


def test_tesseract_ocr_det_and_rec_mode() -> None:
    """验证 det=True, rec=True 完整表格/单图识别模式。"""
    mock_runner = Mock()
    mock_runner.recognize.return_value = ("සිංහල", 0.91)

    mock_detector = Mock()
    box1 = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
    mock_detector.return_value = (np.array([box1]), 0.01)

    with (
        patch("mineru.model.ocr.tesseract_ocr.ensure_tessdata") as mock_ensure,
        patch(
            "mineru.model.ocr.tesseract_ocr._TesseractCtypesRunner",
            return_value=mock_runner,
        ),
    ):
        mock_ensure.return_value = "/mock/tessdata"
        ocr = TesseractOCR(lang="sin+eng", text_detector=mock_detector)
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)

        res = ocr.ocr(test_img, det=True, rec=True)
        assert len(res) == 1
        assert res[0] is not None
        assert len(res[0]) == 1
        item_box, item_rec = res[0][0]
        assert item_rec == ("සිංහල", 0.91)


def test_tesseract_ocr_det_only_mode() -> None:
    """验证 det=True, rec=False 纯检测模式。"""
    mock_detector = Mock()
    box = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=np.float32)
    mock_detector.return_value = (np.array([box]), 0.01)

    with (
        patch("mineru.model.ocr.tesseract_ocr.ensure_tessdata") as mock_ensure,
        patch(
            "mineru.model.ocr.tesseract_ocr._TesseractCtypesRunner", return_value=Mock()
        ),
    ):
        mock_ensure.return_value = "/mock/tessdata"
        ocr = TesseractOCR(lang="sin+eng", text_detector=mock_detector)
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)

        res = ocr.ocr(test_img, det=True, rec=False)
        assert len(res) == 1
        assert res[0] is not None
        assert len(res[0]) == 1


def test_hybrid_atom_model_init_routes_to_tesseract() -> None:
    """验证 atom_model_init 在 lang='sin' 时正确返回 TesseractOCR。"""
    with (
        patch("mineru.model.ocr.tesseract_ocr.ensure_tessdata") as mock_ensure,
        patch(
            "mineru.model.ocr.tesseract_ocr._TesseractCtypesRunner", return_value=Mock()
        ),
        patch.object(TesseractOCR, "_init_default_text_detector", return_value=Mock()),
    ):
        mock_ensure.return_value = "/mock/tessdata"
        model = hybrid.atom_model_init(AtomicModelName.OCR, lang="sin")
        assert isinstance(model, TesseractOCR)
        assert model.lang == "sin+eng"


def test_hybrid_local_model_context_singleton_caches_by_lang() -> None:
    """验证 HybridLocalModelContextSingleton 针对不同语言分别维护上下文缓存。"""
    with patch.object(hybrid.HybridLocalModelContext, "__init__", return_value=None):
        singleton = hybrid.HybridLocalModelContextSingleton()
        singleton._models.clear()

        ctx_ch = singleton.get_model(lang="ch")
        ctx_sin = singleton.get_model(lang="sin")

        assert ctx_ch is not ctx_sin
        # 再次获取应当击中缓存
        assert singleton.get_model(lang="ch") is ctx_ch
        assert singleton.get_model(lang="sin") is ctx_sin


def test_create_job_request_accepts_lang() -> None:
    """验证 API server 的 CreateJobRequest 接受 lang 参数并具有合理的缺省值。"""
    req_default = CreateJobRequest(
        files=[{"source": {"type": "local", "path": "/tmp/doc.pdf"}}]
    )
    assert req_default.lang == "ch"

    req_sin = CreateJobRequest(
        files=[{"source": {"type": "local", "path": "/tmp/doc.pdf"}}],
        lang="sin",
    )
    assert req_sin.lang == "sin"


def test_api_parser_serializes_lang() -> None:
    """验证 MinerUApiParser 正确保存并序列化 lang 字段到请求 payload。"""
    parser = MinerUApiParser(api_url="http://localhost:8000", lang="sin")
    payload = parser._build_payload({"type": "local", "path": "/tmp/doc.pdf"}, "")
    assert payload.get("lang") == "sin"


def test_mineru_parser_stores_lang() -> None:
    """验证 MinerUParser 构造函数正确接收并保存 lang。"""
    parser = MinerUParser(tier="flash", lang="sin")
    assert parser.lang == "sin"
