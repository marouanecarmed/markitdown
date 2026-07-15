from __future__ import annotations

import asyncio
import os
import secrets
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from markitdown import MarkItDown


app = FastAPI(
    title="MarkItDown API",
    description="HTTP API backed by the local MarkItDown source repository.",
    version="1.0.0",
)

# Keep plugins disabled unless you explicitly install and configure them.
converter = MarkItDown(enable_plugins=False)

# The free instance has limited memory. Process one conversion at a time.
conversion_lock = asyncio.Lock()

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


class ConversionResponse(BaseModel):
    filename: str
    bytes: int
    markdown: str


def require_api_key(
    x_api_key: Annotated[
        str | None,
        Header(alias="X-API-Key"),
    ] = None,
) -> None:
    expected_key = os.getenv("API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API_KEY is not configured on the server.",
        )

    if not secrets.compare_digest(x_api_key or "", expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "MarkItDown API",
        "status": "running",
        "documentation": "/docs",
        "conversion_endpoint": "/convert",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post(
    "/convert",
    response_model=ConversionResponse,
    dependencies=[Depends(require_api_key)],
)
async def convert_document(
    file: Annotated[UploadFile, File(description="Document to convert")],
) -> ConversionResponse:
    original_filename = Path(file.filename or "upload").name
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported extension: {extension or 'none'}",
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

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                uploaded_bytes += len(chunk)

                if uploaded_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit."
                        ),
                    )

                temporary_file.write(chunk)

        async with conversion_lock:
            result = await run_in_threadpool(
                converter.convert_local,
                str(temporary_path),
            )

        markdown = getattr(result, "markdown", None)

        # Compatibility with older MarkItDown result objects.
        if markdown is None:
            markdown = getattr(result, "text_content", "")

        return ConversionResponse(
            filename=original_filename,
            bytes=uploaded_bytes,
            markdown=markdown,
        )

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