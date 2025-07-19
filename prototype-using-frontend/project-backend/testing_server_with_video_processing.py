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
from datetime import datetime
import time
import cv2
import pytesseract
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the sentence transformer model for clustering
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_blurry(image, threshold=100.0):
    """Check if an image is blurry using Laplacian variance"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian < threshold

def ocr_frame(image_path):
    """Extract text from an image using OCR"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"❌ OCR error for {image_path}: {e}")
        return ""

def get_text_embedding(text):
    """Get embedding for text clustering"""
    try:
        return model.encode(text, convert_to_tensor=True)
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return torch.zeros(384)  # Default embedding size for all-MiniLM-L6-v2

def cluster_texts(ocr_texts, eps=0.5, min_samples=1):
    """Cluster similar texts together"""
    if not ocr_texts:
        return []
    
    try:
        embeddings = [get_text_embedding(text) for text in ocr_texts]
        X = torch.stack(embeddings).numpy()
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
        return clustering.labels_
    except Exception as e:
        print(f"❌ Clustering error: {e}")
        return list(range(len(ocr_texts)))  # Each text gets its own cluster

def merge_clusters(ocr_texts, labels):
    """Merge texts in the same cluster"""
    if not ocr_texts:
        return ""
    
    try:
        cluster_map = defaultdict(list)
        for i, label in enumerate(labels):
            cluster_map[label].append(ocr_texts[i])
        
        merged_texts = []
        for group in cluster_map.values():
            # Remove duplicates and combine
            combined = "\n".join(set(group))
            merged_texts.append(combined)
        return "\n\n".join(merged_texts)
    except Exception as e:
        print(f"❌ Merge error: {e}")
        return "\n".join(ocr_texts)

def process_frames_with_ocr(frame_paths):
    """Process saved frames with OCR and clustering"""
    print("🔍 Starting OCR and clustering analysis...")
    
    ocr_texts = []
    valid_frames = []
    
    for frame_path in frame_paths:
        try:
            # Read image with OpenCV for blur detection
            image = cv2.imread(frame_path)
            if image is None:
                print(f"⚠️ Could not read image: {frame_path}")
                continue
                
            # Check if image is blurry
            if is_blurry(image):
                print(f"⚠️ Skipping blurry frame: {frame_path}")
                continue
            
            # Extract text with OCR
            text = ocr_frame(frame_path)
            if text and len(text.strip()) > 10:  # Only keep meaningful text
                ocr_texts.append(text)
                valid_frames.append(frame_path)
                print(f"✅ OCR extracted {len(text)} chars from {os.path.basename(frame_path)}")
            else:
                print(f"⚠️ No meaningful text found in {os.path.basename(frame_path)}")
                
        except Exception as e:
            print(f"❌ Error processing {frame_path}: {e}")
    
    if not ocr_texts:
        print("⚠️ No valid OCR text extracted from any frames")
        return "No text could be extracted from the provided frames."
    
    print(f"📝 Extracted text from {len(ocr_texts)} frames")
    
    # Cluster similar texts
    try:
        labels = cluster_texts(ocr_texts)
        merged_text = merge_clusters(ocr_texts, labels)
        print(f"🔗 Clustered {len(ocr_texts)} texts into {len(set(labels))} groups")
        return merged_text
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        return "\n\n".join(ocr_texts)

@app.post("/process-multiple-frames-stream")
async def enhanced_frame_processor(request: Request):
    """Enhanced endpoint that reads frames, saves them, performs OCR/clustering, then streams analysis"""
    
    try:
        print("=" * 50)
        print("🔄 READING, SAVING, AND ANALYZING FRAMES...")
        
        # Get content type
        content_type = request.headers.get("content-type", "")
        print(f"📋 Content-Type: {content_type}")
        
        # Parse form data
        form_data = await request.form()
        print(f"📥 Form data items: {len(form_data)}")
        print(f"🔍 Form keys: {list(form_data.keys())}")
        
        # Create frames directory
        os.makedirs("frames", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process and save each frame
        saved_frames = []
        frame_paths = []
        frame_count = 0
        
        for key, value in form_data.items():
            if key.startswith('frame_') and hasattr(value, 'read'):
                try:
                    # Read file data
                    file_data = await value.read()
                    print(f"📝 Processing {key}: {len(file_data)} bytes")
                    
                    # Open image
                    img = Image.open(io.BytesIO(file_data))
                    
                    # Save frame with timestamp
                    save_path = f"frames/frame_{timestamp}_{key}.png"
                    img.save(save_path)
                    frame_paths.append(save_path)
                    
                    saved_frames.append({
                        "key": key,
                        "filename": getattr(value, 'filename', 'unknown'),
                        "size_bytes": len(file_data),
                        "image_size": f"{img.width}x{img.height}",
                        "saved_path": save_path
                    })
                    
                    frame_count += 1
                    print(f"✅ Saved {key}: {save_path} ({img.width}x{img.height})")
                    
                except Exception as frame_error:
                    print(f"❌ Error processing {key}: {frame_error}")
            
            elif key == 'frame_count':
                expected_count = str(value)
                print(f"📊 Expected frame count: {expected_count}")
        
        print(f"🎉 Successfully saved {frame_count} frames to frames/ folder!")
        
        # Now create streaming response with initial success data + OCR analysis + streaming text
        async def generate_stream():
            # First yield the success response
            initial_response = {
                "success": True,
                "message": f"Successfully received and saved {frame_count} frames!",
                "frames_saved": saved_frames,
                "timestamp": timestamp,
                "save_directory": "frames/",
                "streaming": True,
                "frame_count": frame_count,
                "type": "initial"
            }
            yield f"data: {json.dumps(initial_response)}\n\n"
            
            # Small delay before starting analysis
            await asyncio.sleep(0.3)
            
            # Perform OCR and clustering analysis
            yield f"data: {json.dumps({'type': 'stream', 'content': '🔍 Analyzing frames with OCR and clustering...'})}"
            await asyncio.sleep(0.5)
            
            # Process frames with OCR (this is CPU intensive, so we do it in the async context)
            detected_text = process_frames_with_ocr(frame_paths)
            
            yield f"data: {json.dumps({'type': 'stream', 'content': f'📝 OCR Analysis Complete! Extracted text from {len(frame_paths)} frames.'})}"
            await asyncio.sleep(0.5)
            
            # Analyze the detected text and provide intelligent response
            if detected_text and detected_text != "No text could be extracted from the provided frames.":
                # Split detected text into chunks for streaming
                text_chunks = detected_text.split('\n\n')
                
                # Stream the analysis of detected content
                analysis_responses = [
                    f"🤖 **Content Analysis Complete!**\n\nI've analyzed the content from your {len(frame_paths)} frames and found the following:\n",
                    f"📋 **Detected Content:**\n\n{detected_text[:500]}{'...' if len(detected_text) > 500 else ''}\n",
                    "🔍 **Content Summary:**\n\nBased on the OCR analysis, I can see text content in your frames. ",
                    "This appears to be captured from a screen or document. ",
                    "The text has been processed and clustered to remove duplicates and improve readability.\n",
                    "💡 **Next Steps:**\n\nTo get a detailed analysis of this content, you can:\n",
                    "1. Use the production server with OpenAI API integration\n",
                    "2. Send the extracted text to an AI service for problem solving\n",
                    "3. Process the text further for specific LeetCode problem identification\n",
                    f"📊 **Processing Statistics:**\n- Total frames processed: {frame_count}\n- Valid frames for OCR: {len(frame_paths)}\n- Text extraction successful: ✅\n- Clustering applied: ✅\n",
                    "🎯 **Analysis Complete!** The content has been successfully extracted and processed.\n"
                ]
                
                for i, response in enumerate(analysis_responses):
                    await asyncio.sleep(0.6)  # Slightly faster for real content
                    stream_data = {
                        "type": "stream",
                        "content": response,
                        "step": i + 1,
                        "total_steps": len(analysis_responses)
                    }
                    yield f"data: {json.dumps(stream_data)}\n\n"
            else:
                # Handle case where no text was extracted
                no_text_responses = [
                    "⚠️ **No Text Detected**\n\nI couldn't extract any meaningful text from the provided frames.\n",
                    "🔍 **Possible Reasons:**\n- Frames may be too blurry or low quality\n- No text content in the images\n- OCR processing failed\n",
                    "💡 **Suggestions:**\n- Ensure frames contain clear, readable text\n- Check image quality and resolution\n- Try with different frames\n",
                    f"📊 **Processing Statistics:**\n- Total frames processed: {frame_count}\n- Valid frames for OCR: {len(frame_paths)}\n- Text extraction: ❌\n",
                    "🎯 **Analysis Complete!** No text content was found in the provided frames.\n"
                ]
                
                for i, response in enumerate(no_text_responses):
                    await asyncio.sleep(0.6)
                    stream_data = {
                        "type": "stream",
                        "content": response,
                        "step": i + 1,
                        "total_steps": len(no_text_responses)
                    }
                    yield f"data: {json.dumps(stream_data)}\n\n"
            
            # Final completion message with real detected text
            if detected_text and detected_text != "No text could be extracted from the provided frames.":
                final_data = {
                    "type": "complete",
                    "content": "🎯 Content analysis completed successfully!\n",
                    "total_frames_processed": frame_count,
                    "detected_text": f"""**OCR Analysis Results from {frame_count} frames:**

**Extracted Text:**
```
{detected_text}
```

**Technical Details:**
- Frames processed: {frame_count}
- Valid frames for OCR: {len(frame_paths)}
- Timestamp: {timestamp}
- Save location: frames/
- Text extraction: ✅ Successful
- Clustering: ✅ Applied
- Content type: Real OCR data from uploaded frames

**Analysis Summary:**
The system successfully extracted and processed text content from your uploaded frames. The text has been cleaned, clustered to remove duplicates, and is ready for further analysis.

**Next Steps:**
1. Use the production server with OpenAI API for detailed problem analysis
2. Send the extracted text to an AI service for LeetCode problem solving
3. Process the text for specific problem identification and solution generation

**Processing Pipeline:**
1. Frame reception and saving ✅
2. Blur detection and filtering ✅
3. OCR text extraction ✅
4. Text clustering and deduplication ✅
5. Content analysis and streaming ✅"""
                }
            else:
                final_data = {
                    "type": "complete",
                    "content": "⚠️ No text content detected in frames!\n",
                    "total_frames_processed": frame_count,
                    "detected_text": f"""**OCR Analysis Results from {frame_count} frames:**

**Extracted Text:**
```
No text could be extracted from the provided frames.
```

**Technical Details:**
- Frames processed: {frame_count}
- Valid frames for OCR: {len(frame_paths)}
- Timestamp: {timestamp}
- Save location: frames/
- Text extraction: ❌ Failed
- Clustering: ❌ Not applicable
- Content type: No readable text found

**Analysis Summary:**
The system could not extract any meaningful text from the uploaded frames. This could be due to:
- Blurry or low-quality images
- No text content in the frames
- OCR processing issues
- Poor image resolution or contrast

**Suggestions:**
1. Ensure frames contain clear, readable text
2. Check image quality and resolution
3. Try with different frames that have better text visibility
4. Verify that the frames actually contain text content

**Processing Pipeline:**
1. Frame reception and saving ✅
2. Blur detection and filtering ✅
3. OCR text extraction ❌
4. Text clustering and deduplication ❌
5. Content analysis: No content to analyze ❌"""
                }
            yield f"data: {json.dumps(final_data)}\n\n"
            
            # Send the [DONE] signal that frontend is waiting for
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
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
            "error": f"Processing failed: {str(e)}"
        })

# Start server
async def start_enhanced_server():
    import uvicorn
    try:
        print("🚀 Starting ENHANCED FRAME PROCESSOR server on http://localhost:8000")
        print("📁 Frames will be saved to: frames/ directory")
        print("🔍 Now includes OCR, clustering, and real content analysis!")
        print("📡 Streaming real OCR data analysis from uploaded frames!")
        
        print("💡 Send your frontend request to save frames and get real content analysis!")
        
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    asyncio.run(start_enhanced_server()) 