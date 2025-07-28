import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import json
import asyncio
import io
import os
import base64
import cv2
import numpy as np
from datetime import datetime
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_blur_score(image):
    """Calculate blur score using Laplacian variance method"""
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate Laplacian variance (higher = sharper)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
        
    except Exception as e:
        print(f"❌ Error calculating blur score: {e}")
        return 0.0

def select_best_frame(frames_data):
    """Select the sharpest frame based on blur scores"""
    best_frame = None
    best_score = 0
    best_key = None
    
    print("🔍 Analyzing frames for sharpness...")
    
    for frame_data in frames_data:
        blur_score = calculate_blur_score(frame_data['image'])
        frame_data['blur_score'] = blur_score
        
        print(f"📊 {frame_data['key']}: blur score = {blur_score:.2f}")
        
        if blur_score > best_score:
            best_score = blur_score
            best_frame = frame_data
            best_key = frame_data['key']
    
    print(f"🏆 Best frame: {best_key} with score {best_score:.2f}")
    return best_frame, best_score

@app.post("/process-multiple-frames-stream")
async def process_multiple_frames_stream(request: Request):
    """Analyze frames captured in rapid succession, pick best one, and identify LeetCode problem"""
    
    try:
        print("=" * 50)
        print("🚀 RAPID FRAME ANALYSIS STARTED...")
        
        # Parse form data
        form_data = await request.form()
        print(f"📥 Form data items: {len(form_data)}")
        print(f"🔍 Form keys: {list(form_data.keys())}")
        
        # Create frames directory
        os.makedirs("rapid_frames", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect all frames with blur analysis
        frames_data = []
        frame_count = 0
        
        for key, value in form_data.items():
            if key.startswith('frame_') and hasattr(value, 'read'):
                try:
                    # Read file data
                    file_data = await value.read()
                    print(f"📝 Processing {key}: {len(file_data)} bytes")
                    
                    # Open image
                    img = Image.open(io.BytesIO(file_data))
                    
                    # Store frame data
                    frame_data = {
                        "key": key,
                        "image": img,
                        "file_data": file_data,
                        "filename": getattr(value, 'filename', 'unknown'),
                        "size_bytes": len(file_data),
                        "image_size": f"{img.width}x{img.height}",
                        "timestamp": timestamp
                    }
                    frames_data.append(frame_data)
                    frame_count += 1
                    
                except Exception as frame_error:
                    print(f"❌ Error processing {key}: {frame_error}")
        
        if not frames_data:
            return JSONResponse({
                "success": False,
                "error": "No valid frames received"
            })
        
        print(f"📊 Received {frame_count} frames for analysis")
        
        # Select the best (sharpest) frame
        best_frame, best_blur_score = select_best_frame(frames_data)
        
        if not best_frame:
            return JSONResponse({
                "success": False,
                "error": "No valid frame could be selected"
            })
        
        # Save the best frame
        best_frame_path = f"rapid_frames/best_frame_{timestamp}_{best_frame['key']}_score_{best_blur_score:.2f}.png"
        best_frame['image'].save(best_frame_path)
        print(f"💾 Saved best frame: {best_frame_path}")
        
        # Save all frames for comparison
        all_frames_info = []
        for frame_data in frames_data:
            frame_path = f"rapid_frames/all_{timestamp}_{frame_data['key']}_score_{frame_data['blur_score']:.2f}.png"
            frame_data['image'].save(frame_path)
            all_frames_info.append({
                "key": frame_data['key'],
                "blur_score": frame_data['blur_score'],
                "path": frame_path,
                "selected": frame_data['key'] == best_frame['key']
            })
        
        # Create streaming response
        async def generate_analysis_stream():
            # Initial response
            initial_response = {
                "success": True,
                "message": f"Selected best frame from {frame_count} rapid captures",
                "best_frame": {
                    "key": best_frame['key'],
                    "blur_score": best_blur_score,
                    "path": best_frame_path,
                    "size": best_frame['image_size']
                },
                "all_frames": all_frames_info,
                "timestamp": timestamp,
                "type": "initial"
            }
            yield f"data: {json.dumps(initial_response)}\n\n"
            
            await asyncio.sleep(0.3)
            
            try:
                # Convert best frame to base64 for OpenAI
                base64_img = base64.b64encode(best_frame['file_data']).decode('utf-8')
                
                print("🤖 Sending best frame to OpenAI for LeetCode problem identification...")
                
                # Prepare content for OpenAI API - focused on problem identification
                content = [
                    {
                        "type": "text", 
                        "text": "This is a screenshot of a LeetCode problem webpage. "
                        "Please identify ONLY the problem number and name. "
                        "Return the result in this exact format: '[NUMBER]. [PROBLEM NAME]' "
                        "For example: '1. Two Sum' or '42. Trapping Rain Water'. "
                        "If you cannot clearly identify the problem number and name, return 'Unable to identify problem'."
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    }
                ]
                
                # Stream response from OpenAI
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",  # Using gpt-4o-mini for faster response
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    temperature=0.1,  # Low temperature for consistent identification
                    max_tokens=100,   # Short response expected
                    stream=True
                )
                
                accumulated_content = ""
                step = 0
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content_chunk = chunk.choices[0].delta.content
                        accumulated_content += content_chunk
                        step += 1
                        
                        stream_data = {
                            "type": "stream",
                            "content": content_chunk,
                            "step": step,
                            "accumulated": accumulated_content
                        }
                        yield f"data: {json.dumps(stream_data)}\n\n"
                        
                        await asyncio.sleep(0.01)
                
                print("✅ OpenAI identification completed!")
                
                # Extract the problem identification
                problem_identification = accumulated_content.strip()
                print(f"🎯 Identified problem: {problem_identification}")
                
                # Add separator before solution generation
                separator_data = {
                    "type": "stream",
                    "content": "\n\n" + "="*50 + "\n🔍 Searching for solution...\n" + "="*50 + "\n\n",
                    "step": step + 1
                }
                yield f"data: {json.dumps(separator_data)}\n\n"
                await asyncio.sleep(0.5)
                
                # Second GPT call with internet search for solution
                if problem_identification and "Unable to identify" not in problem_identification:
                    print("🌐 Calling GPT with internet search for solution...")
                    
                    solution_stream = client.chat.completions.create(
                        model="gpt-4.1",  # Using GPT-4.1 which has internet access
                        messages=[
                            {
                                "role": "user",
                                "content": f"Search the internet for the LeetCode problem '{problem_identification}' and provide a complete working Python code solution\n"
                                           f"Make sure the code is clean, well-commented, and ready to run on LeetCode."
                            }
                        ],
                        temperature=0.3,
                        stream=True
                    )
                    
                    solution_content = ""
                    solution_step = 0
                    
                    for chunk in solution_stream:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            content_chunk = chunk.choices[0].delta.content
                            solution_content += content_chunk
                            solution_step += 1
                            
                            stream_data = {
                                "type": "solution_stream",
                                "content": content_chunk,
                                "step": solution_step,
                                "accumulated_solution": solution_content
                            }
                            yield f"data: {json.dumps(stream_data)}\n\n"
                            
                            await asyncio.sleep(0.02)
                    
                    print("✅ Solution generation completed!")
                else:
                    # If problem identification failed, provide fallback
                    fallback_solution = {
                        "type": "solution_stream",
                        "content": "❌ Could not identify the problem clearly, so unable to search for solution. Please try capturing the screen again with the problem title clearly visible.",
                        "step": 1
                    }
                    yield f"data: {json.dumps(fallback_solution)}\n\n"
                
            except Exception as api_error:
                print(f"❌ OpenAI API Error: {api_error}")
                error_data = {
                    "type": "stream",
                    "content": f"❌ Error calling OpenAI API: {str(api_error)}\n",
                    "error": True
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            
            # Final completion message
            final_data = {
                "type": "complete",
                "content": "🎯 Problem identification and solution generation complete!\n",
                "analysis_summary": {
                    "total_frames_captured": frame_count,
                    "best_frame_selected": best_frame['key'],
                    "best_blur_score": best_blur_score,
                    "save_directory": "rapid_frames/",
                    "timestamp": timestamp,
                    "problem_identified": accumulated_content.strip() if 'accumulated_content' in locals() else "Analysis failed",
                    "solution_generated": "solution_content" in locals() and len(solution_content) > 0
                }
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
            # Send completion signal
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_analysis_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": f"Rapid frame analysis failed: {str(e)}"
        })

@app.get("/")
async def root():
    return {
        "message": "🚀 Rapid Frame LeetCode Analyzer",
        "description": "Captures frames in rapid succession, selects the sharpest one, and identifies LeetCode problems",
        "endpoints": {
            "/process-multiple-frames-stream": "POST - Analyze rapid frame captures for LeetCode problem identification"
        }
    }

# Start server
async def start_rapid_analyzer_server():
    import uvicorn
    try:
        print("🚀 Starting RAPID FRAME LEETCODE ANALYZER server on http://localhost:8000")
        print("📸 Captures multiple frames in rapid succession")
        print("🔍 Selects sharpest frame using blur detection")
        print("🤖 Uses OpenAI to identify LeetCode problem number and name")
        print("🌐 Searches internet for complete solution using GPT-4o")
        print("📁 Saves frames to: rapid_frames/ directory")
        print("🔑 Make sure your OPENAI_API_KEY is set in .env file")
        
        print("\n💡 Usage:")
        print("POST /process-multiple-frames-stream with form data containing rapid frame captures")
        print("Expected response: Problem identification + complete solution with explanation\n")
        
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    asyncio.run(start_rapid_analyzer_server())
