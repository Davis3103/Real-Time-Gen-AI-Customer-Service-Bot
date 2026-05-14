from __future__ import annotations
from pathlib import Path
from typing import Optional
from PIL import Image

from .config import GEMINI_API_KEY

def analyze_image(image_path: str | Path, prompt: str = "Describe this image") -> dict:
    path = Path(image_path)
    if not path.exists():
        return {"success": False, "response": "Image not found."}
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            image = Image.open(path)
            result = model.generate_content([prompt, image])
            return {"success": True, "response": result.text or "No response.", "provider": "gemini"}
        except Exception as e:
            return {"success": False, "response": f"Gemini multimodal analysis failed: {e}"}
    image = Image.open(path)
    return {
        "success": True,
        "provider": "local-fallback",
        "response": (
            f"Image loaded successfully. Size: {image.size[0]}x{image.size[1]}. "
            f"Mode: {image.mode}. For richer visual understanding, configure GEMINI_API_KEY."
        ),
    }

def build_image_generation_prompt(user_request: str, context: str = "") -> str:
    return (
        "Create a clean support-style illustration for the following request. "
        f"Request: {user_request}. Context: {context}. "
        "Style: modern, friendly, clear, high-resolution, minimal background."
    )
