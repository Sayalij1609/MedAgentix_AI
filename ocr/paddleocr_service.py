# -*- coding: utf-8 -*-
"""
MedAgentix AI -- PaddleOCR Service (Module A)
===============================================
Standalone OCR text extraction service using PaddleOCR (PP-OCRv4).
This module handles ONLY text extraction from medical document images.

It has NO dependency on Gemini, no medical logic, and no network calls
beyond the initial PaddleOCR model download.

Returns:
    {
        "raw_text": "full extracted text",
        "segments": [
            {"text": "...", "confidence": 0.98, "bbox": [x1,y1,x2,y2]}
        ]
    }
"""

import os
import time
import logging
import tempfile

# Pre-load PyTorch if available to initialize Windows OpenMP and DLL dependencies cleanly
try:
    import torch  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger("medagentix.ocr")

# ---------------------------------------------------------------------------
# Supported file types and size limits
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_PDF_EXTENSIONS
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Lazy-loaded singleton for the PaddleOCR engine
# ---------------------------------------------------------------------------
_ocr_engine = None


def _apply_paddle_fix():
    """
    Disable MKLDNN on Paddle inference predictors.
    This resolves a known issue in PaddlePaddle 3.3.1 on Windows CPU where
    ConvertPirAttribute2RuntimeAttribute throws NotImplementedError for
    pir::ArrayAttribute<pir::DoubleAttribute> inside the oneDNN executor.
    """
    try:
        import paddle.inference as pi
        if not getattr(pi, "_medagentix_patched", False):
            _orig_create = pi.create_predictor

            def _safe_create_predictor(config):
                try:
                    config.disable_mkldnn()
                except Exception:
                    pass
                return _orig_create(config)

            pi.create_predictor = _safe_create_predictor
            pi._medagentix_patched = True
            logger.info("Paddle inference MKLDNN patch applied successfully.")
    except Exception as e:
        logger.warning("Could not apply Paddle MKLDNN patch: %s", e)


def _get_ocr_engine():
    """Lazily initialize and cache the PaddleOCR model instance."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    # Ensure MKLDNN patch is active before PaddleOCR loads models
    _apply_paddle_fix()

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError(
            "PaddleOCR is not installed. "
            "Install it with: pip install paddlepaddle paddleocr"
        )

    logger.info("Initializing PaddleOCR engine...")
    start = time.time()

    # In PaddleOCR 3.7+, use_textline_orientation replaces deprecated use_angle_cls
    try:
        _ocr_engine = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
        )
    except (TypeError, ValueError):
        # Fallback for older PaddleOCR versions
        _ocr_engine = PaddleOCR(
            lang="en",
            use_angle_cls=True,
        )

    elapsed = time.time() - start
    logger.info("PaddleOCR engine ready (%.2fs)", elapsed)
    return _ocr_engine


def validate_file(filename: str, file_size_bytes: int = 0) -> str:
    """
    Validate uploaded file by extension and size.

    Returns:
        str: The lowercase file extension.

    Raises:
        ValueError: If the file type is unsupported or exceeds size limit.
    """
    if not filename:
        raise ValueError("No filename provided.")

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File size ({file_size_bytes / (1024*1024):.1f} MB) exceeds "
            f"the {MAX_FILE_SIZE_MB} MB limit."
        )

    return ext


def _extract_from_image(image_path: str) -> dict:
    """
    Run PaddleOCR on a single image file.

    Args:
        image_path: Absolute path to an image file.

    Returns:
        dict with "raw_text" and "segments".
    """
    engine = _get_ocr_engine()

    start = time.time()
    if hasattr(engine, "predict"):
        results = engine.predict(image_path)
    else:
        results = engine.ocr(image_path)
    elapsed = time.time() - start

    segments = []
    text_parts = []

    if results:
        items = results if isinstance(results, list) else [results]
        for res in items:
            if not res:
                continue

            # PaddleX/PaddleOCR 3.7+ dictionary format
            if isinstance(res, dict) and "rec_texts" in res:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                polys = res.get("rec_polys", [])

                for i in range(len(texts)):
                    text = texts[i]
                    confidence = float(scores[i]) if i < len(scores) else 0.9
                    bbox_points = polys[i] if (polys is not None and i < len(polys)) else None

                    if bbox_points is not None and len(bbox_points) > 0:
                        try:
                            xs = [float(pt[0]) for pt in bbox_points]
                            ys = [float(pt[1]) for pt in bbox_points]
                            bbox = [
                                int(round(min(xs))), int(round(min(ys))),
                                int(round(max(xs))), int(round(max(ys)))
                            ]
                        except Exception:
                            bbox = [0, 0, 0, 0]
                    else:
                        bbox = [0, 0, 0, 0]

                    segments.append({
                        "text": text,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    })
                    text_parts.append(text)

            # Legacy PaddleOCR 2.x list format
            elif isinstance(res, list):
                for line in res:
                    if not line:
                        continue
                    bbox_points = line[0]          # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    text = line[1][0]              # Detected text string
                    confidence = float(line[1][1]) # Confidence score

                    xs = [pt[0] for pt in bbox_points]
                    ys = [pt[1] for pt in bbox_points]
                    bbox = [
                        round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))
                    ]

                    segments.append({
                        "text": text,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    })
                    text_parts.append(text)

    raw_text = "\n".join(text_parts)

    # Log metadata only — never log patient text content
    logger.info(
        "OCR completed: segments=%d, avg_confidence=%.3f, time=%.2fs",
        len(segments),
        sum(s["confidence"] for s in segments) / max(len(segments), 1) if segments else 0.0,
        elapsed,
    )

    return {
        "raw_text": raw_text,
        "segments": segments,
    }


def _convert_pdf_to_images(pdf_path: str) -> list:
    """
    Convert a PDF file to a list of temporary image file paths.

    Uses PyMuPDF (fitz) for PDF-to-image conversion.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        list of temporary image file paths (caller must clean up).
    """
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed. "
            "Install it with: pip install PyMuPDF"
        )

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Render at 300 DPI for OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)

        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"medagentix_ocr_page_{page_num}_{os.getpid()}.png"
        )
        pix.save(temp_path)
        image_paths.append(temp_path)

    doc.close()
    logger.info("PDF converted: %d page(s) extracted", len(image_paths))
    return image_paths


def extract_text(file_path: str, file_extension: str = None) -> dict:
    """
    Main entry point: extract text from an image or PDF file.

    Args:
        file_path: Absolute path to the uploaded file.
        file_extension: Optional override for the file extension (lowercase, with dot).

    Returns:
        dict: {"raw_text": str, "segments": list[dict]}

    Raises:
        ValueError: If the file is empty or unsupported.
        RuntimeError: If OCR fails.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    if file_extension is None:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

    # ----- Handle PDF input -----
    if file_extension in ALLOWED_PDF_EXTENSIONS:
        temp_images = []
        try:
            temp_images = _convert_pdf_to_images(file_path)
            if not temp_images:
                return {"raw_text": "", "segments": []}

            # OCR each page and merge results
            all_segments = []
            all_text_parts = []

            for img_path in temp_images:
                page_result = _extract_from_image(img_path)
                all_segments.extend(page_result["segments"])
                if page_result["raw_text"]:
                    all_text_parts.append(page_result["raw_text"])

            return {
                "raw_text": "\n\n".join(all_text_parts),
                "segments": all_segments,
            }
        finally:
            # Clean up temporary page images
            for img_path in temp_images:
                try:
                    os.remove(img_path)
                except OSError:
                    pass

    # ----- Handle image input -----
    elif file_extension in ALLOWED_IMAGE_EXTENSIONS:
        return _extract_from_image(file_path)

    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
