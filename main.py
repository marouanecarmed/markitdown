from __future__ import annotations

import asyncio
import os
import secrets
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from markitdown import MarkItDown


app = FastAPI(
    title="MarkItDown API",
    version="1.0.0",
)

converter = MarkItDown(enable_plugins=False)
conversion_gate = asyncio.Semaphore(1)

MAX_UPLOAD_BYTES = int(
    os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".msg",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".xml",
    ".txt",
    ".md",
    ".zip",
    ".epub",
}


def require_api_key(
    x_api_key: Annotated[
        str | None,
        Header(alias="X-API-Key"),
    ] = None,
) -> None:
    expected = os.getenv("API_KEY")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API_KEY is not configured.",
        )

    if not secrets.compare_digest(x_api_key or "", expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "MarkItDown API",
        "status": "running",
        "docs": "/docs",
        "convert": "/convert",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post(
    "/convert",
    dependencies=[Depends(require_api_key)],
)
async def convert_document(
    file: Annotated[UploadFile, File()],
) -> dict[str, str | int]:
    filename = Path(file.filename or "upload").name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension: {extension or 'none'}",
        )

    temporary_path: Path | None = None
    uploaded_bytes = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            while chunk := await file.read(1024 * 1024):
                uploaded_bytes += len(chunk)

                if uploaded_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file is too large.",
                    )

                temporary_file.write(chunk)

        async with conversion_gate:
            result = await run_in_threadpool(
                converter.convert_local,
                str(temporary_path),
            )

        markdown = getattr(result, "markdown", None)

        if markdown is None:
            markdown = getattr(result, "text_content", "")

        return {
            "filename": filename,
            "bytes": uploaded_bytes,
            "markdown": markdown,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Conversion failed: {type(exc).__name__}: {exc}",
        ) from exc

    finally:
        await file.close()

        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)