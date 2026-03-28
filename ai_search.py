import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)

print("Loading ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_source):
    try:
        if image_source.startswith('http'):
            response = requests.get(image_source, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_source).convert('RGB')
        
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            features = model(img_tensor)
        
        features = features.squeeze().numpy()
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
        
    except Exception as e:
        print(f"Error: {e}")
        return None

print("Loading product images...")

json_path = 'all_product_images.json'
if not os.path.exists(json_path):
    print(f"ERROR: {json_path} not found!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    exit(1)

with open(json_path, 'r', encoding='utf-8') as f:
    content = f.read()
    data = json.loads(content)

print(f"JSON loaded, type: {type(data)}")

product_dict = {}  

if isinstance(data, dict):
    first_key = list(data.keys())[0]
    items = data[first_key]
    print(f"Found data under key: {first_key}")
else:
    items = data

print(f"Total items in JSON: {len(items)}")

for item in items:
    if isinstance(item, dict):
        product_id = item.get('product_id')
        image_url = item.get('image_url')
        
        if product_id and image_url:
            if product_id not in product_dict:
                product_dict[product_id] = image_url
                print(f"Added product {product_id}: {image_url[:50]}...")

print(f"\nLoaded {len(product_dict)} unique products")

if len(product_dict) == 0:
    print("ERROR: No products loaded! Check JSON format.")
    exit(1)

product_ids = list(product_dict.keys())
product_urls = list(product_dict.values())

print("\nPre-computing features for all products...")
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
        
        temp_path = 'temp_upload.jpg'
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
        top_matches = [m for m in similarities if m[1] > 0.3][:8]
        matches = [int(m[0]) for m in top_matches]
        top_score = top_matches[0][1] if top_matches else 0
        
        print(f"Search completed. Found {len(matches)} matches. Top score: {top_score:.4f}")
        
        if matches:
            return jsonify({
                'status': 'success',
                'matches': matches,
                'top_score': float(top_score),
                'all_scores': [[int(m[0]), float(m[1])] for m in top_matches]
            })
        else:
            return jsonify({
                'status': 'no_match',
                'matches': [],
                'message': 'No products with similarity > 0.3 found'
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
    print("\n" + "="*50)
    print("Starting AI Search Server...")
    print(f"Products loaded: {len(product_dict)}")
    print(f"Product IDs: {product_ids[:10]}...")
    print("API available at: http://localhost:5000/visual_search")
    print("Health check: http://localhost:5000/health")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
