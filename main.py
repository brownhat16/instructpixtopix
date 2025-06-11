import json
import time
import logging
import base64
import requests
from io import BytesIO
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import aiohttp
import ssl
import certifi
from PIL import Image
import tempfile
import os
from together import Together
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic Models
class BannerRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    seed: Optional[int] = 0
    return_format: str = "json"  # "json", "image", or "both"


class BannerResponse(BaseModel):
    status: str
    request_id: str
    structured_data: Optional[Dict] = None
    flux_prompt: Optional[str] = None
    short_prompt: Optional[str] = None
    image_base64: Optional[str] = None
    download_url: Optional[str] = None
    processing_time: float
    error: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: str


# Global storage for async jobs and generated images
processing_jobs = {}
generated_images = {}


class CompleteBannerPipeline:
    def __init__(self, together_api_key: str, nvidia_api_key: str):
        if not together_api_key or not nvidia_api_key:
        raise ValueError("Both API keys are required")
    
        self.together_api_key = together_api_key
        self.nvidia_api_key = nvidia_api_key
        self.together_base_url = "https://api.together.xyz/v1/chat/completions"
        self.nvidia_image_url = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"
        self.model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    
        # Initialize Together client with error handling
        try:
            self.together_client = Together(api_key=together_api_key)
            logger.info("✅ Together client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Together client: {e}")
            raise
        self.color_descriptors = {
            "Red": "vibrant red", "Green": "festive green", "Blue": "deep blue",
            "Yellow": "bright yellow", "Orange": "warm orange", "Purple": "rich purple",
            "Pink": "soft pink", "White": "clean white", "Black": "bold black",
            "Brown": "warm brown", "Grey": "neutral grey"
        }
        self.mood_descriptors = {
            "Calm": "serene and peaceful", "Energetic": "dynamic and vibrant",
            "Luxurious": "elegant and premium", "Playful": "fun and engaging",
            "Professional": "clean and corporate", "Cozy": "warm and inviting"
        }
        self.theme_descriptors = {
            "Modern": "contemporary and sleek", "Classic": "timeless and traditional",
            "Retro": "vintage-inspired", "Minimalist": "clean and uncluttered",
            "Corporate": "professional and structured", "Festive": "celebratory and joyful"
        }

    def generate_short_prompt(self, user_prompt: str) -> str:
        """
        Takes a user-provided prompt and returns a refined, detailed short prompt.
        """
        # System prompt to instruct the LLM
        system_instruction = (
            "You are a prompt optimization assistant. Your task is to rewrite the user prompt "
            "into a short, refined, and vivid prompt suitable for a text-to-image generation model. "
            "Keep it concise and structured with the following format: "
            "1. Clearly describe the offer text and CTA text, ensuring the model renders them correctly. "
            "2. List all key elements in the banner such as people, products, or objects. "
            "3. Describe the background environment vividly. "
            "Any additional details like orientation or lighting should be mentioned at the end. "
            "Output only the improved prompt. Do not explain your reasoning or process."
        )
        try:
            # Call the LLM
            response = self.together_client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=200,
                top_p=0.9
            )
            short_prompt = response.choices[0].message.content.strip()
            logger.info(f"Generated short prompt: {short_prompt[:100]}...")
            return short_prompt
        except Exception as e:
            logger.error(f"Failed to generate short prompt: {e}")
            return user_prompt  # Fallback to original prompt

    def create_system_prompt(self) -> str:
        schema = {
            "Dominant colors": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Teal/Magenta",
            "Secondary colors": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Teal/Magenta/None",
            "Accent colors": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Teal/Magenta/None",
            "Brightness": "Light/Dark/Medium",
            "Warm vs cool tones": "Warm/Cool/Neutral",
            "Contrast level": "High/Medium/Low",
            "Color harmony": "Monochromatic/Analogous/Complementary/Triadic/Split-Complementary",
            "Financial instruments": "string",
            "Offer text present": "Yes/No",
            "Offer text content": "string/None",
            "Offer text size": "Small/Medium/Large/Extra Large/None",
            "Offer Text language": "English/Hindi/Marathi/Tamil/Telugu/Bengali/Gujarati/Kannada/Malayalam/Punjabi/Others/Mixed",
            "Offer text Font style": "Bold/Serif/Sans-serif/Script/Display/Handwritten/Monospace",
            "Offer text Font weight": "Thin/Light/Regular/Medium/Bold/Black",
            "Offer text position": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/Banner/Sticker/None",
            "Offer text color": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Gold/Silver/Brand Color/High Contrast",
            "Offer text background": "None/Solid/Gradient/Banner/Badge/Burst/Ribbon",
            "People present": "Yes/No",
            "No. Of people": "1/2/3/3+",
            "Description of People": "string",
            "Action of person": "string",
            "Emotion of people": "Happy/Excited/Calm/Serious/Surprised/Confident/Relaxed/Energetic/Focused/Not Applicable",
            "Product elements": ["string"],
            "Product element positioning": "Center/Left/Right/Top/Bottom/Scattered/Grid/Linear",
            "Product element size emphasis": "Equal/Hero Product/Varied Sizes",
            "Design density": "Minimal/Medium/Dense",
            "Text-to-image ratio": "10%/30%/50%/70%/90%",
            "Left vs right alignment": "Left/Right/Center",
            "Symmetry": "Symmetrical/Asymmetrical",
            "Whitespace usage": "Low/Medium/High",
            "Headline text": "string/None",
            "Headline position": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/Overlay/None",
            "Headline size": "Small/Medium/Large/Extra Large",
            "Headline color": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Brand Color/Contrast Color",
            "Headline style": "Bold/Italic/Underlined/Shadow/Outline/Gradient/None",
            "Subheading text": "string/None",
            "Subheading position": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/Below Headline/None",
            "Subheading size": "Small/Medium/Large",
            "Subheading color": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brown/Grey/Gold/Silver/Brand Color/Contrast Color",
            "Festival special occasion logo": "Yes/No",
            "Festival name": "Diwali/Holi/Christmas/Eid/New Year/Valentine/Mother's Day/Father's Day/Independence Day/Republic Day/Dussehra/Ganesh Chaturthi/Karva Chauth/Raksha Bandhan/None",
            "Brand logo visible": "Yes/No",
            "Logo size": "Small/Medium/Large/Extra Large",
            "Logo placement": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/Corner/Watermark",
            "Logo style": "Full Logo/Icon Only/Text Only/Monogram",
            "Logo transparency": "Opaque/Semi-Transparent/Watermark",
            "Brand tagline": "string/Not Applicable",
            "Call-to-action button present": "Yes/No",
            "CTA text": "string/None",
            "CTA placement": "Top/Center/Bottom/Left/Right/Floating/Multiple Positions",
            "CTA position detail": "Top Left/Top Center/Top Right/Center Left/Center/Center Right/Bottom Left/Bottom Center/Bottom Right/None",
            "CTA size": "Small/Medium/Large/Full Width",
            "CTA shape": "Rectangular/Rounded/Circular/Custom/Pill",
            "CTA style": "Filled/Outlined/Text Only/3D/Gradient",
            "CTA color": "Red/Yellow/Blue/Green/Orange/Purple/Pink/White/Black/Brand Color/Accent Color",
            "CTA text color": "White/Black/Brand Color/Contrast Color",
            "CTA contrast": "High/Medium/Low",
            "CTA animation": "None/Hover Effect/Pulse/Glow/Bounce",
            "Banner layout orientation": "Horizontal/Vertical/Square",
            "Aspect ratio": "16:9/4:3/1:1/3:4/9:16/21:9/Custom",
            "Theme": "Modern/Classic/Retro/Minimalist/Corporate/Festive/Luxury/Playful/Artistic/Tech",
            "Tone & Mood": "Energetic/Calm/Luxurious/Playful/Professional/Cozy/Urgent/Trustworthy/Innovative/Nostalgic",
            "Visual style": "Realistic/Illustrated/Abstract/Photographic/Graphic/Mixed",
            "Background Scene": "string",
            "Background texture": "Solid/Gradient/Pattern/Photographic/Abstract/Geometric/Organic",
            "Background complexity": "Simple/Moderate/Complex",
            "Device orientation": "Portrait/Landscape/Both/Adaptive",
            "Language direction": "LTR/RTL/Mixed/Vertical/Not Applicable",
        }
        schema_json = json.dumps(schema, indent=2)
        prompt = (
            "You are an AI assistant that converts user requests for advertising banners into structured JSON. "
            "Extract relevant attributes and use sensible defaults for unmentioned attributes.\n"
            "Respond with ONLY a valid JSON object using this schema:\n"
            f"{schema_json}\n"
            "Rules:\n"
            "1. Respond with ONLY the JSON object, no additional text\n"
            "2. Use exact field names and values as specified\n"
            "3. Choose appropriate defaults for unmentioned attributes\n"
            "4. Set \"None\" or \"Not Applicable\" when elements are not relevant"
        )
        return prompt

    async def extract_metadata_async(self, user_prompt: str, max_retries: int = 3) -> Optional[Dict]:
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": f"Convert this banner request to JSON: {user_prompt}"}
            ],
            "max_tokens": 2100,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        logger.info(f"Extracting metadata for prompt: {user_prompt[:100]}...")
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                    async with session.post(
                            self.together_base_url,
                            headers=headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"API Error {response.status}: {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None
                        result = await response.json()
                        if "error" in result:
                            logger.error(f"API Error: {result['error']}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return None
                        assistant_message = result["choices"][0]["message"]["content"].strip()
                        cleaned_message = self._clean_json_response(assistant_message)
                        try:
                            metadata = json.loads(cleaned_message)
                            logger.info("✅ Metadata extraction successful")
                            return metadata
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON: {e}")
                            return None
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
        logger.error("❌ All metadata extraction attempts failed")
        return None

    def _clean_json_response(self, response: str) -> str:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            response = response[start_idx:end_idx + 1]
        return response

    def convert_to_flux_prompt(self, metadata: Dict[str, Any]) -> str:
        prompt_parts = []
        background_scene = metadata.get("Background Scene", "")
        if background_scene and background_scene not in ["None", "none", ""]:
            prompt_parts.append(f"Background details: '{background_scene}'")

        # === 1. BANNER LAYOUT & FORMAT ===
        orientation = metadata.get("Banner layout orientation", "Horizontal")
        aspect_ratio = metadata.get("Aspect ratio", "1:1")
        theme = metadata.get("Theme", "Modern")
        mood = metadata.get("Tone & Mood", "Professional")
        visual_style = metadata.get("Visual style", "Realistic")
        prompt_parts.append(
            f"{visual_style} advertising banner with {theme} theme and {mood.lower()} tone, "
            f"{orientation} orientation ({aspect_ratio})"
        )

        # === 2. COLOR SCHEME ===
        dominant_color = metadata.get("Dominant colors", "").split("/")
        secondary_colors = metadata.get("Secondary colors", "").split("/")
        accent_colors = metadata.get("Accent colors", "").split("/")
        brightness = metadata.get("Brightness", "Medium").lower()
        warm_cool = metadata.get("Warm vs cool tones", "Neutral")
        contrast_level = metadata.get("Contrast level", "Medium").lower()
        color_harmony = metadata.get("Color harmony", "Monochromatic").lower()

        if dominant_color != "None":
            color_desc = f"vibrant {dominant_color} palette with {color_harmony} harmony, "
            if secondary_colors:
                color_desc += f"secondary accents in {', '.join(secondary_colors[:3])}, "
            if accent_colors:
                color_desc += f"accent highlights in {', '.join(accent_colors[:2])}, "
            color_desc += f"{brightness} brightness, {contrast_level} contrast, {warm_cool} tone balance"
            prompt_parts.append(color_desc)

        # === 3. TEXT ELEMENTS ===
        offer_text_present = metadata.get("Offer text present", "No")
        if offer_text_present == "Yes":
            offer_content = metadata.get("Offer text content", "Special Offer")
            offer_position = metadata.get("Offer text position", "Top Center")
            offer_size = metadata.get("Offer text size", "Medium")
            offer_color = metadata.get("Offer text color", "Red")
            offer_font_style = metadata.get("Offer text Font style", "Sans-serif")
            prompt_parts.append(
                f"Prominent {offer_size} '{offer_content}' text at {offer_position}, "
                f"{offer_color} color, {offer_font_style} font"
            )

        # === 4. HEADLINE AND SUBHEADING ===
        headline_text = metadata.get("Headline text", "string")
        if headline_text not in ["string", "None"]:
            headline_position = metadata.get("Headline position", "Top Left")
            headline_size = metadata.get("Headline size", "Large")
            headline_color = metadata.get("Headline color", "Brand Color")
            headline_style = metadata.get("Headline style", "Bold")
            prompt_parts.append(
                f"{headline_size} '{headline_text}' headline at {headline_position}, "
                f"{headline_color} color, styled as {headline_style}"
            )

        subheading_text = metadata.get("Subheading text", "string")
        if subheading_text not in ["string", "None"]:
            subheading_position = metadata.get("Subheading position", "Below Headline")
            subheading_size = metadata.get("Subheading size", "Medium")
            subheading_color = metadata.get("Subheading color", "Brand Color")
            prompt_parts.append(
                f"{subheading_size} '{subheading_text}' subheading at {subheading_position}, "
                f"{subheading_color} color"
            )

        # === 5. PEOPLE PRESENTATION ===
        people_present = metadata.get("People present", "No")
        if people_present == "Yes":
            no_of_people = metadata.get("No. Of people", "1")
            description_of_people = metadata.get("Description of People", "")
            emotion_of_people = metadata.get("Emotion of people", "Happy")
            action_of_person = metadata.get("Action of person", "No")
            people_desc = f"Featuring {no_of_people} person(s)"
            if action_of_person:
                people_desc += f" in {action_of_person}"
            if description_of_people:
                people_desc += f" described as {description_of_people}"
            people_desc += f", displaying {emotion_of_people.lower()} emotion"
            prompt_parts.append(people_desc)

        # === 6. PRODUCT ELEMENTS ===
        product_elements = metadata.get("Product elements", [])
        if product_elements:
            product_element_positioning = metadata.get("Product element positioning", "Center")
            product_element_size_emphasis = metadata.get("Product element size emphasis", "Equal")
            if isinstance(product_elements, list) and len(product_elements) > 0:
                if len(product_elements) == 1:
                    product_names = product_elements[0]
                else:
                    product_names = ", ".join(product_elements[:-1]) + " and " + product_elements[-1]
            else:
                product_names = str(product_elements) if product_elements else ""
            product_desc = f"Showcasing {len(product_elements)} product(s): {product_names}, "
            product_desc += f"{product_element_positioning} positioning, "
            product_desc += f"{product_element_size_emphasis} sizing"
            prompt_parts.append(product_desc)

        # === 7. CALL TO ACTION ===
        cta_present = metadata.get("Call-to-action button present", "No")
        if cta_present == "Yes":
            cta_text = metadata.get("CTA text", "Shop Now")
            cta_position = metadata.get("CTA position detail", "Bottom Right")
            cta_shape = metadata.get("CTA shape", "Rectangular")
            cta_style = metadata.get("CTA style", "Filled")
            cta_color = metadata.get("CTA color", "Brand Color")
            cta_text_color = metadata.get("CTA text color", "White")
            cta_desc = (
                f"CTA text as '{cta_text}', at {cta_position} position, "
                f"Prominent {cta_shape} CTA button styled as {cta_style}, "
                f"{cta_color} background, {cta_text_color} text"
            )
            prompt_parts.append(cta_desc)

        # === 8. BACKGROUND ===
        background_texture = metadata.get("Background texture", "Solid")
        background_complexity = metadata.get("Background complexity", "Simple")
        background_desc = f"{background_complexity.lower()} {background_texture.lower()} background"
        prompt_parts.append(background_desc)

        # === 9. DESIGN ELEMENTS ===
        design_density = metadata.get("Design density", "Medium")
        whitespace_usage = metadata.get("Whitespace usage", "Medium")
        text_image_ratio = metadata.get("Text-to-image ratio", "50%")
        symmetry = metadata.get("Symmetry", "Symmetrical")
        design_desc = f"{design_density} design density, {whitespace_usage} whitespace usage, "
        design_desc += f"{text_image_ratio} text-to-image ratio, {symmetry} symmetry"
        prompt_parts.append(design_desc)

        # === 10. BRAND ELEMENTS ===
        brand_logo_visible = metadata.get("Brand logo visible", "No")
        if brand_logo_visible == "Yes":
            logo_placement = metadata.get("Logo placement", "Top Left")
            logo_size = metadata.get("Logo size", "Medium")
            prompt_parts.append(f"Featuring company logo at {logo_placement}, {logo_size} size")

        brand_tagline = metadata.get("Brand tagline", "string")
        if brand_tagline not in ["string", "Not Applicable"]:
            prompt_parts.append(f"Brand tagline: '{brand_tagline}'")

        # === 11. FESTIVAL ELEMENTS ===
        festival_logo = metadata.get("Festival special occasion logo", "No")
        if festival_logo == "Yes":
            festival_name = metadata.get("Festival name", "Diwali")
            prompt_parts.append(f"Featuring {festival_name} celebration graphics")

        # === FINAL PROMPT ASSEMBLY ===
        quality_enhancers = [
            "high-resolution vector-style rendering",
            "sharp details",
            "clean composition",
            "professional-grade output"
        ]
        complete_prompt = ", ".join(prompt_parts + quality_enhancers)
        logger.info(f"Generated comprehensive FLUX prompt: {complete_prompt[:150]}...")
        return complete_prompt

    async def generate_image_nvidia(self, prompt: str, width: int = 1024, height: int = 1024,
                                    num_inference_steps: int = 50, guidance_scale: float = 5.0,
                                    seed: Optional[int] = 0) -> Optional[str]:
        """Generate image using NVIDIA's FLUX.1-dev API"""
        payload = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "cfg_scale": guidance_scale,
            "mode": "base",
            "samples": 1,
            "seed": seed if seed is not None else 0,
            "steps": num_inference_steps
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.nvidia_api_key}"
        }
        logger.info(f"Generating image with NVIDIA FLUX.1-dev: {width}x{height}")
        try:
            response = requests.post(self.nvidia_image_url, json=payload, headers=headers, timeout=120)
            if response.status_code != 200:
                logger.error(f"NVIDIA API Error {response.status_code}: {response.text}")
                return None
            result = response.json()
            if "artifacts" in result and len(result["artifacts"]) > 0:
                image_b64 = result["artifacts"][0]["base64"]  # Access first artifact with [0]
                logger.info("✅ NVIDIA image generation successful")
                return image_b64
            else:
                logger.error("No image data returned from NVIDIA API")
                return None
        except Exception as e:
            logger.error(f"NVIDIA image generation failed: {e}")
            return None

    async def process_banner_request_async(self, user_prompt: str, width: int = 1024,
                                           height: int = 1024, num_inference_steps: int = 50,
                                           guidance_scale: float = 5.0, seed: Optional[int] = 0) -> Dict:
        start_time = time.time()
        logger.info(f"🔄 Processing banner request: {user_prompt[:100]}...")
        # Step 1: Extract metadata
        metadata = await self.extract_metadata_async(user_prompt)
        if not metadata:
            return {
                "status": "error",
                "error": "Failed to extract metadata",
                "processing_time": time.time() - start_time
            }
        # Step 2: Generate comprehensive FLUX prompt
        flux_prompt = self.convert_to_flux_prompt(metadata)
        # Step 3: Generate short refined prompt
        short_prompt = self.generate_short_prompt(user_prompt)
        # Step 4: Generate image using NVIDIA API with short prompt
        image_b64 = await asyncio.get_event_loop().run_in_executor(
            None,
            self.generate_image_nvidia_sync,
            short_prompt, width, height, num_inference_steps, guidance_scale, seed
        )

        processing_time = time.time() - start_time
        if image_b64:
            logger.info(f"✅ Complete pipeline successful in {processing_time:.2f}s")
            return {
                "status": "success",
                "structured_data": metadata,
                "flux_prompt": flux_prompt,
                "short_prompt": short_prompt,
                "image_base64": image_b64,
                "processing_time": processing_time
            }
        else:
            logger.error("❌ Image generation failed")
            return {
                "status": "partial_success",
                "structured_data": metadata,
                "flux_prompt": flux_prompt,
                "short_prompt": short_prompt,
                "error": "Image generation failed",
                "processing_time": processing_time
            }

    def generate_image_nvidia_sync(self, prompt: str, width: int = 1024, height: int = 1024,
                                   num_inference_steps: int = 50, guidance_scale: float = 5.0,
                                   seed: Optional[int] = 0) -> Optional[str]:
        """Synchronous version of NVIDIA image generation for use with executor"""
        payload = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "cfg_scale": guidance_scale,
            "mode": "base",
            "samples": 1,
            "seed": seed if seed is not None else 0,
            "steps": num_inference_steps
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.nvidia_api_key}"
        }
        try:
            response = requests.post(self.nvidia_image_url, json=payload, headers=headers, timeout=120)
            if response.status_code != 200:
                logger.error(f"NVIDIA API Error {response.status_code}: {response.text}")
                return None
            result = response.json()
            if "artifacts" in result and len(result["artifacts"]) > 0:
                image_b64 = result["artifacts"][0]["base64"]  # ✅ Access first artifact with [0]
                return image_b64
            else:
                logger.error("No image data returned from NVIDIA API")
                return None
        except Exception as e:
            logger.error(f"NVIDIA image generation failed: {e}")
            return None

    def save_image_temporarily(self, image_b64: str, request_id: str) -> str:
        """Save image to temporary file and return file path"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            filename = f"banner_{request_id}.png"
            file_path = os.path.join(temp_dir, filename)
            # Save image
            image.save(file_path, format='PNG')
            # Store in global dict for cleanup
            generated_images[request_id] = {
                "file_path": file_path,
                "created_at": datetime.now(),
                "filename": filename
            }
            logger.info(f"Image saved temporarily: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save image temporarily: {e}")
            return None

# Load environment variables
load_dotenv()

# Get API keys from environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Validate API keys
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable is required")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable is required")

# Initialize Pipeline with error handling
try:
    pipeline = CompleteBannerPipeline(TOGETHER_API_KEY, NVIDIA_API_KEY)
    logger.info("✅ Pipeline initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize pipeline: {e}")
    raise


# FastAPI App
app = FastAPI(
    title="Banner Generation API",
    description="Generate advertising banners from natural language prompts using NVIDIA FLUX.1-dev",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=StatusResponse)
async def root():
    return StatusResponse(
        status="online",
        message="Banner Generation API with NVIDIA FLUX.1-dev is running",
        timestamp=datetime.now().isoformat()
    )


@app.post("/generate-banner")
async def generate_banner(request: BannerRequest):
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    logger.info(f"📨 Received banner request {request_id}: {request.prompt}")
    # Store job immediately
    processing_jobs[request_id] = {
        "status": "processing",
        "progress": "Starting...",
        "created_at": datetime.now().isoformat(),
        "result": None
    }
    try:
        result = await pipeline.process_banner_request_async(
            user_prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        processing_time = time.time() - start_time
        # Update job status
        processing_jobs[request_id].update({
            "status": "completed" if result["status"] == "success" else "partial_success",
            "progress": "Complete",
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        download_url = None
        # Handle different return formats
        if request.return_format == "image" and result.get("image_base64"):
            # Return image directly for download
            try:
                image_data = base64.b64decode(result["image_base64"])
                image = Image.open(BytesIO(image_data))
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return StreamingResponse(
                    img_byte_arr,
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f"attachment; filename=banner_{request_id}.png",
                        "X-Request-ID": request_id
                    }
                )
            except Exception as e:
                logger.error(f"Error serving direct image download: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to process image for download: {str(e)}"}
                )
        # For JSON or both formats, save image temporarily and provide download URL
        if result.get("image_base64"):
            file_path = pipeline.save_image_temporarily(result["image_base64"], request_id)
            if file_path:
                download_url = f"/download/{request_id}"
        # Prepare response based on format
        response_data = {
            "status": result["status"],
            "request_id": request_id,
            "processing_time": processing_time,
            "error": result.get("error")
        }
        if request.return_format in ["json", "both"]:
            response_data.update({
                "structured_data": result.get("structured_data"),
                "flux_prompt": result.get("flux_prompt"),
                "short_prompt": result.get("short_prompt"),
                "download_url": download_url
            })
            if request.return_format == "both":
                response_data["image_base64"] = result.get("image_base64")
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"❌ Request {request_id} failed: {str(e)}")
        processing_jobs[request_id].update({
            "status": "failed",
            "progress": f"Error: {str(e)}",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "request_id": request_id,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
        )


@app.get("/download/{request_id}")
async def download_image(request_id: str):
    """Direct download endpoint for generated images"""
    if request_id not in generated_images:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    image_info = generated_images[request_id]
    file_path = image_info["file_path"]
    filename = image_info["filename"]
    if not os.path.exists(file_path):
        # Clean up expired entry
        del generated_images[request_id]
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/generate-banner-async")
async def generate_banner_async(request: BannerRequest, background_tasks: BackgroundTasks):
    request_id = f"async_req_{int(time.time() * 1000)}"
    logger.info(f"📨 Received async banner request {request_id}: {request.prompt}")
    processing_jobs[request_id] = {
        "status": "processing",
        "progress": "Starting...",
        "created_at": datetime.now().isoformat(),
        "result": None
    }
    background_tasks.add_task(
        process_async_banner,
        request_id,
        request.prompt,
        request.width,
        request.height,
        request.num_inference_steps,
        request.guidance_scale,
        request.seed
    )
    return {
        "request_id": request_id,
        "status": "accepted",
        "message": "Banner generation started",
        "check_status_url": f"/status/{request_id}",
        "download_url": f"/download/{request_id}"
    }


@app.get("/status/{request_id}")
async def get_job_status(request_id: str):
    if request_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = processing_jobs[request_id]
    response = {
        "request_id": request_id,
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "result": job["result"]
    }
    # Add download URL if image is available
    if job["status"] == "completed" and request_id in generated_images:
        response["download_url"] = f"/download/{request_id}"
    return response


@app.get("/image/{request_id}")
async def get_generated_image(request_id: str):
    """Legacy endpoint - redirects to download endpoint"""
    return await download_image(request_id)


@app.delete("/cleanup/{request_id}")
async def cleanup_image(request_id: str):
    """Clean up temporary image files"""
    if request_id in generated_images:
        image_info = generated_images[request_id]
        file_path = image_info["file_path"]
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            del generated_images[request_id]
            return {"message": f"Image {request_id} cleaned up successfully"}
        except Exception as e:
            logger.error(f"Failed to cleanup image {request_id}: {e}")
            return {"error": f"Failed to cleanup: {str(e)}"}
    else:
        raise HTTPException(status_code=404, detail="Image not found")


@app.get("/cleanup-old-images")
async def cleanup_old_images():
    """Clean up images older than 1 hour"""
    current_time = datetime.now()
    cleaned_count = 0
    for request_id in list(generated_images.keys()):
        image_info = generated_images[request_id]
        age = current_time - image_info["created_at"]
        if age.total_seconds() > 3600:  # 1 hour
            try:
                file_path = image_info["file_path"]
                if os.path.exists(file_path):
                    os.remove(file_path)
                del generated_images[request_id]
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Failed to cleanup old image {request_id}: {e}")
    return {"message": f"Cleaned up {cleaned_count} old images"}


async def process_async_banner(request_id: str, prompt: str, width: int, height: int,
                             num_inference_steps: int, guidance_scale: float, seed: Optional[int]):
    try:
        result = await pipeline.process_banner_request_async(
            user_prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        # Save image temporarily for download
        if result.get("image_base64"):
            pipeline.save_image_temporarily(result["image_base64"], request_id)
        processing_jobs[request_id].update({
            "status": "completed" if result["status"] == "success" else "partial_success",
            "progress": "Complete",
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        logger.info(f"✅ Async request {request_id} completed successfully")
    except Exception as e:
        processing_jobs[request_id].update({
            "status": "failed",
            "progress": f"Error: {str(e)}",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        logger.error(f"❌ Async request {request_id} failed: {str(e)}")


# Startup event to schedule cleanup
@app.on_event("startup")
async def startup_event():
    import asyncio
    # Schedule cleanup every hour
    asyncio.create_task(periodic_cleanup())


async def periodic_cleanup():
    """Periodic cleanup of old temporary files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Wait 1 hour
            await cleanup_old_images()
            logger.info("🧹 Periodic cleanup completed")
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")

# Run Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
