# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pytesseract
import numpy as np
import base64
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

leetcode_indicators = ["description", "editorial", "submissions", "leetcode"]
save_folder = "leetcode_screenshots"
os.makedirs(save_folder, exist_ok=True)
BLUR_THRESHOLD = 70

def is_clear_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm > BLUR_THRESHOLD

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.post("/process-frame")
async def process_frame(image: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # OCR processing
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(frame, config=custom_config)
        
        # Check for LeetCode indicators
        if any(keyword in text.lower() for keyword in leetcode_indicators):
            if is_clear_image(frame):
                timestamp = int(time.time())
                screenshot_path = os.path.join(save_folder, f"leetcode_screenshot_{timestamp}.png")
                cv2.imwrite(screenshot_path, frame)
                
                base64_img = encode_image(screenshot_path)
                
                # Send to GPT
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Explain this LeetCode problem and provide a solution."},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                            ]
                        }
                    ],
                    temperature=0.2
                )
                
                return {
                    "success": True,
                    "screenshot_path": screenshot_path,
                    "response": response.choices[0].message.content,
                    "detected_text": text
                }
            else:
                return {
                    "success": False,
                    "error": "Image too blurry",
                    "detected_text": text
                }
        else:
            return {
                "success": False,
                "error": "No LeetCode content detected",
                "detected_text": text
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)