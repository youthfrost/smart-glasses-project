import cv2
import os
import pytesseract
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict
import openai

def extract_frames(video_path, output_folder, fps=2):
    os.makedirs(output_folder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if frame_count % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian < threshold


def ocr_frame(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text


# Alternative: Use CLIP or ResNet image embeddings if working with raw images
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embedding(text):
    return model.encode(text, convert_to_tensor=True)


def cluster_texts(ocr_texts, eps=0.5, min_samples=1):
    embeddings = [get_text_embedding(text) for text in ocr_texts]
    X = torch.stack(embeddings).numpy()
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(X)
    return clustering.labels_

def merge_clusters(ocr_texts, labels):
    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_map[label].append(ocr_texts[i])
    
    merged_texts = []
    for group in cluster_map.values():
        # Naive merge, or could use LLM or fuzzy logic here
        combined = "\n".join(set(group))  # Removes duplicates
        merged_texts.append(combined)
    return "\n\n".join(merged_texts)


def clean_with_llm(merged_text):
    prompt = f"""
You are a helpful assistant. Clean up and format the following LeetCode question so it's readable, removing duplicates and correcting any OCR errors:

{merged_text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']



#example usage
def process_video(video_path):
    #video_path = "leetcode_scroll.mp4"
    frame_folder = "frames"

    extract_frames(video_path, frame_folder)

    ocr_texts = []
    for fname in os.listdir(frame_folder):
        img_path = os.path.join(frame_folder, fname)
        image = cv2.imread(img_path)
        if not is_blurry(image):
            ocr_texts.append(ocr_frame(img_path))

    labels = cluster_texts(ocr_texts)
    merged_text = merge_clusters(ocr_texts, labels)

    clean_question = clean_with_llm(merged_text)
    return clean_question
