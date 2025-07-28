import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
import io
import os
from datetime import datetime
import time
from PIL import Image
import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import base64
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

class ImageProcessor:
    @staticmethod
    def order_points(pts):
        """Order points in the order: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        """Apply perspective transform to get bird's eye view"""
        rect = ImageProcessor.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def document_scanner_homography(img, save_debug=True, debug_dir=None, frame_key="unknown", blur_threshold=100.0):
        """Apply document scanner homography transformation with quality checks, rejection, and blur detection"""
        try:
            img_cv = np.array(img)
            if len(img_cv.shape) == 3:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            original = img_cv.copy()
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            if save_debug and debug_dir:
                debug_frame_dir = os.path.join(debug_dir, frame_key)
                os.makedirs(debug_frame_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_frame_dir, "01_grayscale.png"), gray)
                cv2.imwrite(os.path.join(debug_frame_dir, "02_blurred.png"), blurred)
                cv2.imwrite(os.path.join(debug_frame_dir, "03_edges.png"), edged)
            contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            print(f"🔍 Found {len(contours)} contours for {frame_key}")
            debug_contours = original.copy()
            cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
            screenCnt = None
            contour_info = []
            rejection_reason = None
            for idx, c in enumerate(contours):
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                contour_info.append({
                    "index": idx,
                    "area": area,
                    "perimeter": peri,
                    "points": len(approx)
                })
                print(f"  Contour {idx}: Area={area:.0f}, Perimeter={peri:.0f}, Points={len(approx)}")
                if len(approx) >= 3:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(debug_contours, f"{idx}({len(approx)}p)", (cx-20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if len(approx) == 4:
                    min_area = (img_cv.shape[0] * img_cv.shape[1]) * 0.1
                    if area < min_area:
                        rejection_reason = f"contour_too_small_area_{area:.0f}_min_{min_area:.0f}"
                        print(f"  ❌ Contour {idx} rejected: too small (area: {area:.0f} < {min_area:.0f})")
                        continue
                    rect = ImageProcessor.order_points(approx.reshape(4, 2))
                    (tl, tr, br, bl) = rect
                    width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    avg_width = (width1 + width2) / 2
                    avg_height = (height1 + height2) / 2
                    aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height)
                    if aspect_ratio > 5.0:
                        rejection_reason = f"extreme_aspect_ratio_{aspect_ratio:.2f}"
                        print(f"  ❌ Contour {idx} rejected: extreme aspect ratio ({aspect_ratio:.2f})")
                        continue
                    def angle_between_vectors(v1, v2):
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        return np.degrees(np.arccos(cos_angle))
                    corners = approx.reshape(4, 2)
                    angles = []
                    for i in range(4):
                        p1 = corners[i]
                        p2 = corners[(i + 1) % 4]
                        p3 = corners[(i + 2) % 4]
                        v1 = p1 - p2
                        v2 = p3 - p2
                        angle = angle_between_vectors(v1, v2)
                        angles.append(angle)
                    angle_threshold = 45
                    bad_angles = [abs(angle - 90) for angle in angles if abs(angle - 90) > angle_threshold]
                    if bad_angles:
                        rejection_reason = f"bad_corner_angles_{bad_angles}"
                        print(f"  ❌ Contour {idx} rejected: bad corner angles {angles}")
                        continue
                    print(f"  ✅ Found valid 4-point contour at index {idx}!")
                    print(f"    Area: {area:.0f}, Aspect ratio: {aspect_ratio:.2f}, Angles: {[f'{a:.1f}°' for a in angles]}")
                    screenCnt = approx
                    cv2.drawContours(debug_contours, [approx], -1, (0, 0, 255), 3)
                    for i, point in enumerate(approx):
                        cv2.circle(debug_contours, tuple(point[0]), 8, (255, 255, 0), -1)
                        cv2.putText(debug_contours, str(i), tuple(point[0] + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    break
            if save_debug and debug_dir:
                cv2.imwrite(os.path.join(debug_frame_dir, "04_contours_detected.png"), debug_contours)
                with open(os.path.join(debug_frame_dir, "contour_info.txt"), "w") as f:
                    f.write(f"Contour Analysis for {frame_key}\n")
                    f.write("=" * 40 + "\n")
                    for info in contour_info:
                        f.write(f"Contour {info['index']}: Area={info['area']:.0f}, "
                                f"Perimeter={info['perimeter']:.0f}, Points={info['points']}\n")
                    f.write(f"\nSelected contour: {'Found' if screenCnt is not None else 'None'}\n")
                    if rejection_reason:
                        f.write(f"Rejection reason: {rejection_reason}\n")
            if screenCnt is not None:
                print(f"✅ Applying homography transformation for {frame_key}")
                warped = ImageProcessor.four_point_transform(original, screenCnt.reshape(4, 2))
                min_warped_size = 200
                if warped.shape[0] < min_warped_size or warped.shape[1] < min_warped_size:
                    rejection_reason = f"warped_too_small_{warped.shape[1]}x{warped.shape[0]}"
                    print(f"  ❌ Warped result rejected: too small ({warped.shape[1]}x{warped.shape[0]})")
                    raise ValueError(f"Warped image too small: {warped.shape}")
                warp_aspect = max(warped.shape[0], warped.shape[1]) / min(warped.shape[0], warped.shape[1])
                if warp_aspect > 10.0:
                    rejection_reason = f"warped_extreme_aspect_{warp_aspect:.2f}"
                    print(f"  ❌ Warped result rejected: extreme aspect ratio ({warp_aspect:.2f})")
                    raise ValueError(f"Warped image too distorted: aspect ratio {warp_aspect:.2f}")
                gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                non_zero_pixels = np.count_nonzero(gray_warped > 50)
                total_pixels = gray_warped.shape[0] * gray_warped.shape[1]
                content_ratio = non_zero_pixels / total_pixels
                if content_ratio < 0.3:
                    rejection_reason = f"warped_mostly_empty_{content_ratio:.2f}"
                    print(f"  ❌ Warped result rejected: mostly empty ({content_ratio:.2%} content)")
                    raise ValueError(f"Warped image mostly empty: {content_ratio:.2%} content")
                # Blur detection using variance of Laplacian
                laplacian_var = cv2.Laplacian(gray_warped, cv2.CV_64F).var()
                if laplacian_var < blur_threshold:
                    rejection_reason = f"warped_blurry_var_{laplacian_var:.1f}_thresh_{blur_threshold}"
                    print(f"  ❌ Warped result rejected: blurry (Laplacian variance {laplacian_var:.1f} < {blur_threshold})")
                    if save_debug and debug_dir:
                        with open(os.path.join(debug_frame_dir, "failure_reason.txt"), "a") as f:
                            f.write(f"Blur detection: Laplacian variance {laplacian_var:.1f} < {blur_threshold}\n")
                    raise ValueError(f"Warped image too blurry: Laplacian variance {laplacian_var:.1f}")
                if save_debug and debug_dir:
                    cv2.imwrite(os.path.join(debug_frame_dir, "05_warped_result.png"), warped)
                    with open(os.path.join(debug_frame_dir, "success_info.txt"), "w") as f:
                        f.write(f"Homography successful for {frame_key}\n")
                        f.write(f"Warped size: {warped.shape[1]}x{warped.shape[0]}\n")
                        f.write(f"Warped aspect ratio: {warp_aspect:.2f}\n")
                        f.write(f"Content ratio: {content_ratio:.2%}\n")
                        f.write(f"Blur (Laplacian variance): {laplacian_var:.1f}\n")
                if len(warped.shape) == 3:
                    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                return Image.fromarray(warped), "success"
            else:
                if not rejection_reason:
                    rejection_reason = "no_4_point_contour_found"
                print(f"⚠️ No valid 4-point contour found for {frame_key}")
                if save_debug and debug_dir:
                    with open(os.path.join(debug_frame_dir, "failure_reason.txt"), "w") as f:
                        f.write(f"Homography failed for {frame_key}\n")
                        f.write(f"Reason: {rejection_reason}\n\n")
                        f.write("Suggestions:\n")
                        f.write("1. Check if image has clear document edges\n")
                        f.write("2. Adjust Canny edge detection parameters\n")
                        f.write("3. Try different contour approximation epsilon\n")
                        f.write("4. Ensure good contrast between document and background\n")
                        f.write("5. Make sure document occupies significant portion of image\n")
                return None, rejection_reason
        except Exception as e:
            error_msg = f"error_{str(e).replace(' ', '_')}"
            print(f"❌ Homography error for {frame_key}: {e}")
            if save_debug and debug_dir:
                debug_frame_dir = os.path.join(debug_dir, frame_key)
                os.makedirs(debug_frame_dir, exist_ok=True)
                with open(os.path.join(debug_frame_dir, "error.txt"), "w") as f:
                    f.write(f"Error processing {frame_key}: {str(e)}\n")
            return None, error_msg

@app.post("/process-multiple-frames-stream")
async def process_frames_development(request: Request):
    """Development endpoint with homography and panorama functionality with frame rejection (NO cropping)"""
    try:
        print("=" * 50)
        print("🔄 DEVELOPMENT FRAME PROCESSING WITH HOMOGRAPHY & PANORAMA & REJECTION (NO CROPPING)...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = "development"
        session_dir = os.path.join(base_dir, timestamp)
        before_dir = os.path.join(session_dir, "before_homography")
        after_dir = os.path.join(session_dir, "after_homography")
        rejected_dir = os.path.join(session_dir, "rejected_homography")
        debug_dir = os.path.join(session_dir, "debug_visualization")
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        form_data = await request.form()
        print(f"📥 Form data items: {len(form_data)}")
        all_images = []
        rejected_images = []
        processing_results = []
        frame_count = 0
        rejected_count = 0
        for key, value in form_data.items():
            if key.startswith('frame_') and hasattr(value, 'read'):
                try:
                    file_data = await value.read()
                    print(f"📝 Processing {key}: {len(file_data)} bytes")
                    original_img = Image.open(io.BytesIO(file_data))
                    print(f"✅ Processing {key}")
                    processed_img, status = ImageProcessor.document_scanner_homography(
                        original_img,
                        save_debug=True,
                        debug_dir=debug_dir,
                        frame_key=key,
                        blur_threshold=100.0  # You can adjust this threshold
                    )
                    if processed_img is not None and status == "success":
                        image_data = {
                            "key": key,
                            "original_image": original_img,
                            "processed_image": processed_img,
                            "file_data": file_data,
                            "filename": getattr(value, 'filename', 'unknown'),
                            "size_bytes": len(file_data),
                            "image_size": f"{original_img.width}x{original_img.height}",
                            "processed_size": f"{processed_img.width}x{processed_img.height}",
                            "status": "accepted"
                        }
                        all_images.append(image_data)
                        before_path = os.path.join(before_dir, f"{key}_original.png")
                        after_path = os.path.join(after_dir, f"{key}_homography.png")
                        original_img.save(before_path)
                        processed_img.save(after_path)
                        processing_results.append({
                            "frame_key": key,
                            "original_path": before_path,
                            "processed_path": after_path,
                            "debug_path": os.path.join(debug_dir, key),
                            "original_size": f"{original_img.width}x{original_img.height}",
                            "processed_size": f"{processed_img.width}x{processed_img.height}",
                            "original_file_size": len(file_data),
                            "processed_file_size": os.path.getsize(after_path),
                            "status": "accepted"
                        })
                        frame_count += 1
                        print(f"✅ ACCEPTED {key} - homography successful")
                    else:
                        rejected_path = os.path.join(rejected_dir, f"{key}_rejected_{status}.png")
                        original_img.save(rejected_path)
                        rejected_images.append({
                            "key": key,
                            "filename": getattr(value, 'filename', 'unknown'),
                            "size_bytes": len(file_data),
                            "image_size": f"{original_img.width}x{original_img.height}",
                            "rejection_reason": status,
                            "rejected_path": rejected_path,
                            "debug_path": os.path.join(debug_dir, key)
                        })
                        rejected_count += 1
                        print(f"❌ REJECTED {key} - {status}")
                except Exception as frame_error:
                    print(f"❌ Error processing {key}: {frame_error}")
        print(f"🎉 Successfully processed {frame_count} frames!")
        print(f"❌ Rejected {rejected_count} frames!")
        print(f"📋 {len(all_images)} accepted images loaded into list for testing")
        base64_images = []
        for img_data in all_images:
            buffered = io.BytesIO()
            img_data["processed_image"].save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_base64)
        
        async def generate_development_stream():
            initial_response = {
                "success": True,
                "message": f"Development processing complete! {frame_count} accepted, {rejected_count} rejected (NO cropping)",
                "timestamp": timestamp,
                "session_directory": session_dir,
                "directories": {
                    "session": session_dir,
                    "before_homography": before_dir,
                    "after_homography": after_dir,
                    "rejected_homography": rejected_dir,
                    "debug_visualization": debug_dir
                },
                "frame_count": frame_count,
                "rejected_count": rejected_count,
                "all_images_count": len(all_images),
                "header_footer_cropped": False,
                "type": "initial"
            }
            yield f"data: {json.dumps(initial_response)}\n\n"
            
            try:
                # Prepare content for OpenAI API
                # Prepare content for OpenAI API
                content = [
                    {"type": "text", 
                    "text": "Analyze these LeetCode screenshots captured while scrolling. "
                    "Different frames may show different parts of the same problem (description, examples, constraints, code editor). "
                    "Combine information from all frames to provide: 1) Problem name/number 2) Complete working code solution 3) Brief explanation. "
                    "Be concise - code first, minimal explanation."
                    "Ensure not to use any imports or libraries in the code solution"}
                ]
                # Add all base64 images to the content
                for base64_img in base64_images:
                    content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                    })
                
                print(f"🤖 Sending {len(base64_images)} images to OpenAI API...")
                
                # Stream response from OpenAI
                stream = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    temperature=0.2,
                    stream=True
                )
                
                accumulated_content = ""
                step = 0
                
                for chunk in stream:
                    # Handle content chunks
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
                        
                        # Small delay to make streaming visible
                        await asyncio.sleep(0.01)
                
                print("✅ OpenAI streaming completed!")
                
            except Exception as api_error:
                print(f"❌ OpenAI API Error: {api_error}")
                error_data = {
                    "type": "stream",
                    "content": f"❌ Error calling OpenAI API: {str(api_error)}\n\nUsing fallback response...\n",
                    "error": True
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                
                # Fallback message
                fallback_data = {
                    "type": "stream",
                    "content": "🤖 Unable to analyze images with AI. Please check your OpenAI API key and try again.\n"
                }
                yield f"data: {json.dumps(fallback_data)}\n\n"
            
            # Final completion message
            final_data = {
                "type": "complete",
                "content": "🎯 Analysis complete!\n",
                "total_frames_processed": frame_count,
                "detected_text": f"""**Placeholder for detected text from {frame_count} frames:**

This will be replaced with actual OCR text extraction in future versions.
For now, the AI analysis above contains the problem understanding and solution.

**Technical Details:**
- Frames processed: {frame_count}
- Images sent to AI: {len(base64_images)}
- Timestamp: {timestamp}
- Save location: frames/

**Next Steps:**
- Implement OCR text extraction
- Add text preprocessing
- Enhance problem detection accuracy"""
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
            # Send the [DONE] signal that frontend is waiting for
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            generate_development_stream(),
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
            "error": f"Development processing failed: {str(e)}"
        })

def main():
    print("🚀 Starting ENHANCED DEVELOPMENT server on http://localhost:8000")
    print("📁 Images organized: before_homography → after_homography")
    print("🔧 Document scanner homography with QUALITY CONTROL!")
    print("❌ Frame rejection for poor quality transforms!")
    print("🔍 DEBUG VISUALIZATION with quality metrics!")
    print("📂 Directory structure: development/TIMESTAMP/[before|after|rejected|debug]/")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()