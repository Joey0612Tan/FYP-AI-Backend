import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)

print("Loading ONNX model...")
session = ort.InferenceSession("resnet50_final.onnx")
input_name = session.get_inputs()[0].name

def extract_features(image_source):
    try:
        if isinstance(image_source, str) and image_source.startswith('http'):
            response = requests.get(image_source, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_source).convert('RGB')
        
        img = img.resize((224, 224))
        img_data = np.array(img, dtype=np.float32)  
        img_data = img_data / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)
        
        features = session.run(None, {input_name: img_data})[0].flatten()
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
    except Exception as e:
        print(f"Error: {e}")
        return None

print("Loading product images...")
json_path = 'all_product_images.json'
with open(json_path, 'r') as f:
    data = json.load(f)

if isinstance(data, dict):
    items = list(data.values())[0]
else:
    items = data

products = {}
for item in items:
    if isinstance(item, dict):
        pid = item.get('product_id')
        url = item.get('image_url')
        if pid and url and pid not in products:
            products[pid] = url

print(f"Loaded {len(products)} products")

print("Pre-computing features...")
product_ids = []
product_features = []
for pid, url in products.items():
    print(f"Processing {pid}...")
    feat = extract_features(url)
    if feat is not None:
        product_ids.append(pid)
        product_features.append(feat)

print(f"Ready. {len(product_ids)} products")

@app.route('/visual_search', methods=['POST'])
def visual_search():
    try:
        file = request.files['image']
        file.save('temp.jpg')
        query = extract_features('temp.jpg')
        os.remove('temp.jpg')
        
        if query is None:
            return jsonify({'status': 'error'}), 500
        
        scores = []
        for i, feat in enumerate(product_features):
            sim = cosine_similarity([query], [feat])[0][0]
            scores.append((product_ids[i], sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        matches = [int(s[0]) for s in scores[:5] if s[1] > 0.3]
        top_score = scores[0][1] if scores else 0
        
        return jsonify({
            'status': 'success',
            'matches': matches,
            'top_score': float(top_score)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'products': len(product_ids)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
