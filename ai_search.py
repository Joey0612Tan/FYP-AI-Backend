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
CORS(app, resources={r"/*": {"origins": "*"}})

print("Loading ONNX model...")
session = ort.InferenceSession('resnet50_final.onnx')

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array = img_array / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_features(image_source):
    try:
        if image_source.startswith('http'):
            response = requests.get(image_source, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_source).convert('RGB')
        
        input_tensor = preprocess_image(img)
        outputs = session.run(None, {'input': input_tensor})
        features = outputs[0].squeeze()
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

print("Loading product images...")

json_path = 'all_product_images.json'
if not os.path.exists(json_path):
    json_path = os.path.join(os.path.dirname(__file__), 'all_product_images.json')

with open(json_path, 'r', encoding='utf-8') as f:
    content = f.read()
    data = json.loads(content)

product_dict = {}

if isinstance(data, dict):
    first_key = list(data.keys())[0]
    items = data[first_key]
else:
    items = data

for item in items:
    if isinstance(item, dict):
        product_id = item.get('product_id')
        image_url = item.get('image_url')
        
        if product_id and image_url:
            if product_id not in product_dict:
                product_dict[product_id] = image_url
                print(f"Added product {product_id}")

print(f"Loaded {len(product_dict)} unique products")

print("Pre-computing features...")
product_ids = list(product_dict.keys())
product_features = []

for i, (pid, url) in enumerate(product_dict.items()):
    print(f"Processing product {i+1}/{len(product_dict)}: ID={pid}")
    features = extract_features(url)
    if features is not None:
        product_features.append(features)
    else:
        print(f"  Failed to extract features for product {pid}")
        product_features.append(np.zeros(2048))

print(f"Successfully processed {len(product_features)} products")

@app.route('/visual_search', methods=['POST'])
def visual_search():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        temp_path = '/tmp/temp_upload.jpg'
        file.save(temp_path)
        
        query_features = extract_features(temp_path)
        
        try:
            os.remove(temp_path)
        except:
            pass
        
        if query_features is None:
            return jsonify({'status': 'error', 'error': 'Failed to extract features'}), 500
        
        similarities = []
        for i, prod_feat in enumerate(product_features):
            if np.linalg.norm(prod_feat) > 0:
                sim = cosine_similarity([query_features], [prod_feat])[0][0]
                similarities.append((product_ids[i], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        top_matches = [m for m in similarities if m[1] > 0.2][:8]
        matches = [int(m[0]) for m in top_matches]
        top_score = top_matches[0][1] if top_matches else 0
        
        print(f"Search completed. Found {len(matches)} matches. Top score: {top_score:.4f}")
        
        if matches:
            return jsonify({
                'status': 'success',
                'matches': matches,
                'top_score': float(top_score)
            })
        else:
            return jsonify({
                'status': 'no_match',
                'matches': [],
                'message': 'No products found'
            })
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'products_loaded': len(product_dict),
        'product_ids': product_ids[:10]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
