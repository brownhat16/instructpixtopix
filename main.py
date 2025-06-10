from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image
import io
import logging
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InstructPix2Pix Image Editor", version="1.0.0")

# Rate limiter: Max 10 requests per minute per IP
request_counts = defaultdict(list)

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()

    request_times = request_counts[client_ip]
    request_times.append(now)
    request_times[:] = [t for t in request_times if t > now - 60]

    if len(request_times) > 10:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests"}
        )

    response = await call_next(request)
    return response

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the pipeline
pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    """Load the InstructPix2Pix model on startup"""
    global pipe
    try:
        logger.info("Loading InstructPix2Pix model...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
        logger.info(f"Model loaded successfully on {device}!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.on_event("shutdown")
async def unload_model():
    """Unload model and clear memory on shutdown"""
    global pipe
    if pipe is not None:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded and memory cleared.")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "InstructPix2Pix Image Editor API is running!"}

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if pipe is None:
        return {"status": "Model not loaded"}
    
    return {
        "model": "timbrooks/instruct-pix2pix",
        "status": "loaded",
        "device": device,
        "torch_dtype": str(pipe.unet.dtype).split('.')[-1]  # e.g., float16 or float32
    }

@app.post("/edit-image")
async def edit_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(30)
):
    """
    Edit an uploaded image using text instructions
    
    - **file**: Image file to edit (JPEG, PNG)
    - **prompt**: Text instruction for editing (e.g., "make the sky blue")
    - **guidance_scale**: How closely to follow the prompt (default: 7.5)
    - **num_inference_steps**: Number of denoising steps (default: 30)
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only .png, .jpg, and .jpeg files are allowed")

    if len(prompt) > 256:
        raise HTTPException(status_code=400, detail="Prompt must be under 256 characters")

    if num_inference_steps < 10 or num_inference_steps > 100:
        raise HTTPException(status_code=400, detail="num_inference_steps must be between 10 and 100")

    try:
        # Read and process the uploaded image
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB").resize((512, 512))
        
        logger.info(f"Processing image with prompt: {prompt}")
        
        # Generate edited image
        with torch.no_grad():
            edited_image = pipe(
                image=input_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).images[0]
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        edited_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            io.BytesIO(img_byte_arr.read()), 
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=edited_image.png"}
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
