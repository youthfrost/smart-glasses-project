# 🚀 Enhanced LeetCode Assistant Server with Video Processing

This enhanced version of the LeetCode assistant server incorporates video processing capabilities from `clustering.py`, including OCR text extraction, clustering, and intelligent text analysis.

## 🔧 Features

- **Frame Processing**: Saves uploaded frames to local storage
- **OCR Text Extraction**: Uses Tesseract to extract text from images
- **Blur Detection**: Automatically filters out blurry frames
- **Text Clustering**: Groups similar text content using sentence embeddings
- **Streaming Response**: Real-time analysis updates via Server-Sent Events
- **Mock AI Solutions**: Pre-written LeetCode problem solutions for testing

## 📋 Prerequisites

### System Dependencies

1. **Tesseract OCR** (required for text extraction):

   ```bash
   # macOS
   brew install tesseract

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

2. **Python Dependencies**:
   ```bash
   pip install -r requirements_enhanced.txt
   ```

## 🚀 Quick Start

1. **Install dependencies**:

   ```bash
   cd prototype-using-frontend/project-backend
   pip install -r requirements_enhanced.txt
   ```

2. **Run the enhanced server**:

   ```bash
   python testing_server_with_video_processing.py
   ```

3. **Server will start on**: `http://localhost:8000`

## 📡 API Endpoint

### `POST /process-multiple-frames-stream`

**Input:**

- Form data with image frames (`frame_0`, `frame_1`, etc.)
- Frame count metadata

**Output:**

- Server-Sent Events stream with:
  - Frame saving confirmation
  - OCR analysis progress
  - Text clustering results
  - Mock LeetCode solution
  - Final analysis summary

**Example Response Stream:**

```json
{"type": "initial", "success": true, "message": "Successfully received and saved 5 frames!"}
{"type": "stream", "content": "🔍 Analyzing frames with OCR and clustering...\n"}
{"type": "stream", "content": "📝 OCR Analysis Complete! Extracted text from 5 frames.\n"}
{"type": "stream", "content": "🤖 I can see you're working on the Two Sum problem!..."}
{"type": "complete", "detected_text": "**OCR Analysis Results...**"}
{"type": "done"}
```

## 🔍 Processing Pipeline

1. **Frame Reception**: Receives multiple image frames via form data
2. **Frame Saving**: Saves frames to `frames/` directory with timestamps
3. **Blur Detection**: Filters out blurry frames using Laplacian variance
4. **OCR Extraction**: Extracts text from clear frames using Tesseract
5. **Text Clustering**: Groups similar text using sentence embeddings
6. **Analysis**: Provides mock LeetCode solution with real OCR data
7. **Streaming**: Returns results via Server-Sent Events

## 🛠️ Technical Details

### OCR Processing

- Uses `pytesseract` for text extraction
- Filters frames with blur detection (Laplacian variance < 100)
- Minimum text length requirement (10 characters)

### Text Clustering

- Uses `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
- DBSCAN clustering with cosine similarity
- Automatic duplicate removal and text merging

### Error Handling

- Graceful handling of OCR failures
- Fallback mechanisms for clustering errors
- Detailed logging for debugging

## 📁 File Structure

```
project-backend/
├── testing_server_with_video_processing.py  # Enhanced server
├── requirements_enhanced.txt                # Dependencies
├── clustering.py                           # Original video processing
├── frames/                                 # Saved frames directory
└── README_enhanced_server.md              # This file
```

## 🔧 Configuration

### Blur Detection Threshold

```python
def is_blurry(image, threshold=100.0):  # Adjust threshold as needed
```

### Clustering Parameters

```python
def cluster_texts(ocr_texts, eps=0.5, min_samples=1):  # Adjust clustering sensitivity
```

### OCR Settings

```python
# Minimum text length to consider valid
if text and len(text.strip()) > 10:  # Adjust minimum length
```

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract not found**:

   ```bash
   # Ensure tesseract is installed and in PATH
   tesseract --version
   ```

2. **CUDA/GPU issues with PyTorch**:

   ```bash
   # Install CPU-only version if needed
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory issues with large images**:
   - Reduce image resolution before upload
   - Adjust blur detection threshold

### Debug Mode

The server provides detailed logging. Check console output for:

- Frame processing status
- OCR extraction results
- Clustering information
- Error messages

## 🔄 Integration with Frontend

The enhanced server maintains the same API interface as the original testing server, so your existing frontend code should work without modification. The main difference is that you'll now receive real OCR data in the response.

## 📈 Performance Considerations

- **OCR Processing**: CPU-intensive, may take several seconds for many frames
- **Clustering**: Uses sentence transformers, first run downloads model (~90MB)
- **Memory Usage**: Scales with number and size of frames
- **Response Time**: Includes processing delay for OCR and clustering

## 🎯 Next Steps

1. **Real AI Integration**: Replace mock responses with actual OpenAI API calls
2. **Advanced OCR**: Implement better text preprocessing and error correction
3. **Problem Detection**: Add LeetCode problem identification from OCR text
4. **Performance Optimization**: Implement async OCR processing
5. **Caching**: Add model and result caching for better performance
