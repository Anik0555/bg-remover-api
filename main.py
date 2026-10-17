import io
import re
import logging
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from rembg import new_session, remove

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("passport_bg_removal")

# Global session cache container
ml_models = {}

# ------------------------------------------------------------------------------
# Application Lifespan (Startup / Shutdown Session Lifecycle)
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load IS-Net model globally once on server startup to eliminate cold starts.
    """
    model_name = "isnet-generaluse"
    logger.info(f"Initializing global rembg session with model: '{model_name}'...")
    try:
        # Pre-warm session into global state
        ml_models["session"] = new_session(model_name)
        logger.info("rembg model session successfully loaded and ready.")
    except Exception as e:
        logger.critical(f"Fatal error loading model session: {str(e)}", exc_info=True)
        raise RuntimeError("Failed to initialize vision model session.") from e

    yield

    # Teardown logic
    logger.info("Cleaning up session resources...")
    ml_models.clear()


app = FastAPI(
    title="Passport Studio HD Background Removal Engine",
    description="Studio-grade cutout API with hair-level alpha matting and dynamic canvas injection.",
    version="1.0.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------------------
# CORS Middleware Configuration
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed input image MIME types
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/tiff",
}

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def validate_hex_color(hex_code: str) -> tuple[int, int, int]:
    """
    Validates and converts a hex string (e.g., #FFFFFF or #00A3FF) to RGB tuple.
    """
    clean_hex = hex_code.strip().lstrip("#")
    if not re.match(r"^[0-9a-fA-F]{6}$", clean_hex):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid hex color format '{hex_code}'. Must be in 6-character format like '#FFFFFF' or '#00A3FF'.",
        )
    return tuple(int(clean_hex[i : i + 2], 16) for i in (0, 2, 4))


def refine_alpha_edges(rgba_image: Image.Image, feather_radius: float = 0.5) -> Image.Image:
    """
    Post-processes the alpha mask via OpenCV to eliminate harsh aliasing,
    staircasing artifacts, and border fringing on hair strands without
    blurring core facial features or garment contours.
    """
    np_img = np.array(rgba_image)
    r, g, b, alpha = cv2.split(np_img)

    # Apply edge antialiasing solely to transition pixels (semi-transparent boundaries)
    # Using Gaussian blur directly on alpha transition zones prevents halo effects
    soft_alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_radius, sigmaY=feather_radius)

    # Preserve hard foreground (>245) and hard background (<10)
    solid_mask = alpha > 245
    void_mask = alpha < 10
    transition_mask = ~(solid_mask | void_mask)

    # Blend original and feathered alpha along the boundaries only
    final_alpha = alpha.copy()
    final_alpha[transition_mask] = soft_alpha[transition_mask]

    refined_rgba = cv2.merge([r, g, b, final_alpha])
    return Image.fromarray(refined_rgba, mode="RGBA")


def composite_solid_background(
    foreground_rgba: Image.Image,
    bg_color_rgb: tuple[int, int, int]
) -> Image.Image:
    """
    Flawlessly composites an RGBA cutout onto a solid RGB background at 100% native resolution.
    Uses Pillow's native C-optimized alpha_composite to prevent edge color leaks.
    """
    # Create matching solid RGBA background
    background = Image.new("RGBA", foreground_rgba.size, (*bg_color_rgb, 255))
    
    # Composite foreground on top of the solid canvas
    composited = Image.alpha_composite(background, foreground_rgba)
    
    # Convert to pure RGB (discarding alpha channel) for JPEG export
    return composited.convert("RGB")


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK, tags=["Monitoring"])
async def health_check():
    """
    Probes model readiness and service availability.
    """
    if "session" not in ml_models:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "initializing", "detail": "Model session not ready."},
        )
    return {"status": "ok"}


@app.post("/api/v1/remove-bg", tags=["Studio Image Processing"])
async def remove_background(
    file: UploadFile = File(..., description="High-resolution portrait photo (JPG, PNG, WEBP)"),
    color: Optional[str] = Form(
        None,
        description="Optional hex color for the background (e.g., '#FFFFFF' for white, '#00A3FF' for passport blue). If omitted, returns transparent PNG.",
    ),
    apply_feathering: bool = Form(
        True,
        description="Enable OpenCV antialiasing/feathering on boundaries for studio-grade printing.",
    ),
):
    """
    Cuts the portrait background using IS-Net deep alpha matting.
    - If `color` is supplied, returns a maximum quality (`quality=100`) JPEG.
    - If `color` is omitted, returns an uncompressed, lossless RGBA PNG.
    """
    # 1. Validate Input Content-Type
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{file.content_type}'. Supported formats: {list(SUPPORTED_IMAGE_TYPES)}",
        )

    # 2. Read raw bytes and validate PIL readability
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded.",
            )
        input_image = Image.open(io.BytesIO(contents))
        # Ensure EXIF orientations (phone camera flips) are normalized
        input_image = ImageOps.exif_transpose(input_image)
    except Exception as e:
        logger.error(f"Failed to read image payload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image data.",
        )

    # 3. Retrieve global model session
    session = ml_models.get("session")
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision inference engine is unavailable.",
        )

    # 4. Perform AI Alpha Matting Removal
    try:
        # Settings tuned for high hair-strand detail and crisp collar definition
        result_rgba: Image.Image = remove(
            input_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_process_mask=True,
        )
    except Exception as e:
        logger.error(f"Inference error during background extraction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occurred while executing neural background segmentation.",
        )

    # 5. Apply edge smoothing/antialiasing filter if requested
    if apply_feathering:
        result_rgba = refine_alpha_edges(result_rgba, feather_radius=0.6)

    # 6. Branch: Solid Color Background (JPEG) vs. Transparent Cutout (PNG)
    out_buffer = io.BytesIO()

    if color:
        rgb_color = validate_hex_color(color)
        final_image = composite_solid_background(result_rgba, rgb_color)

        # Save as studio-grade JPEG at 100% quality with 4:4:4 chroma subsampling (no compression blur)
        final_image.save(
            out_buffer,
            format="JPEG",
            quality=100,
            subsampling=0,
            optimize=True,
        )
        media_type = "image/jpeg"
        filename = "processed_studio_photo.jpg"
    else:
        # Save as lossless RGBA PNG
        result_rgba.save(
            out_buffer,
            format="PNG",
            optimize=False,
            compress_level=1,  # Fast encoding while remaining 100% lossless
        )
        media_type = "image/png"
        filename = "transparent_cutout.png"

    out_buffer.seek(0)

    return Response(
        content=out_buffer.getvalue(),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )