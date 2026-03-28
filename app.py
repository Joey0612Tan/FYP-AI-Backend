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
import google.generativeai as genai
import re

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
ai_model = genai.GenerativeModel('gemma-3-4b-it')

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

print("Loading product vectors...")
try:
    vector_db = np.load('product_vectors.npy', allow_pickle=True)
    print(f"✅ Vector DB Loaded: {len(vector_db)} products")
    all_ids = [item['pid'] for item in vector_db]
    print(f"   Product IDs: {all_ids}")
except Exception as e:
    print(f"❌ Vector DB Error: {e}")
    vector_db = []

def extract_features(image_source):
    """Extract feature vector from image (URL or file)"""
    try:
        if isinstance(image_source, str) and image_source.startswith('http'):
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
        print(f"Error extracting features: {e}")
        return None

def get_ai_styled_response(prompt, style_class):
    """Generate AI response with formatting"""
    try:
        response = ai_model.generate_content(prompt)
        res_text = response.text
        res_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', res_text)
        res_text = res_text.replace('*', '')
        replacements = {
            'Overall Sentiment': '📊 Overall Sentiment',
            'Pros': '✅ Pros',
            'Cons': '❌ Cons',
            'Final Verdict': '⚖️ Final Verdict',
            'Key Highlights': '🌟 Key Highlights',
            'Usage Scenarios': '🎯 Usage Scenarios',
            'Expert Tip': '💡 Expert Tip',
            'The Winner': '🏆 The Winner',
            'Specs Showdown': '⚔️ Specs Showdown'
        }
        for old, new in replacements.items():
            res_text = res_text.replace(old, new)
        
        res_text = res_text.replace('\n', '<br>')
        return f'<div class="ai-response-container {style_class}">{res_text}</div>'
    except Exception as e:
        return f'<div class="ai-response-container">😵 AI error: {str(e)}</div>'

@app.route('/visual_search', methods=['POST'])
def visual_search():
    """Visual search: upload image, return matching product IDs"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        file = request.files['image']
        temp_path = '/tmp/temp_upload.jpg'
        file.save(temp_path)
        
        query_features = extract_features(temp_path)
        
        try:
            os.remove(temp_path)
        except:
            pass
        
        if query_features is None:
            return jsonify({"status": "error", "error": "Failed to extract features"}), 500
        
        similarities = []
        for item in vector_db:
            db_vec = item['vector']
            if isinstance(db_vec, list):
                db_vec = np.array(db_vec)
            
            sim = cosine_similarity([query_features], [db_vec])[0][0]
            similarities.append((item['pid'], sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        top_matches = [m for m in similarities if m[1] > 0.3][:8]
        matches = [int(m[0]) for m in top_matches]
        top_score = top_matches[0][1] if top_matches else 0
        
        print(f"Found {len(matches)} matches. Top score: {top_score:.4f}")
        
        if matches:
            return jsonify({
                "status": "success",
                "matches": matches,
                "top_score": float(top_score)
            })
        else:
            return jsonify({"status": "no_match", "matches": []})
            
    except Exception as e:
        print(f"Visual Search Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/summarize_reviews', methods=['POST'])
def summarize_reviews():
    data = request.json
    prompt = f"Summarize these reviews: {data.get('reviews')}. Use <b> and <li>. Summary, Pros/Cons, Verdict."
    return jsonify({'summary': get_ai_styled_response(prompt, 'summarizer-style')})

@app.route('/analyze_product_deep', methods=['POST'])
def analyze_product_deep():
    data = request.json
    prompt = f"Deeply analyze product: {data.get('name')}, Specs: {data.get('specs')}. Structure with Highlights, Scenarios, Tip."
    return jsonify({'analysis': get_ai_styled_response(prompt, 'analysis-style')})

@app.route('/compare_products_ai', methods=['POST'])
def compare_products_ai():
    data = request.json
    context = ""
    for i, p in enumerate(data.get('products', [])):
        context += f"P{i+1}: {p['name']}, Specs: {p['specs']}\n"
    prompt = f"Compare these items:\n{context}\nProvide Specs Showdown, Sentiment, and a Winner."
    return jsonify({'analysis': get_ai_styled_response(prompt, 'comparison-style')})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'products_loaded': len(vector_db),
        'model_loaded': True
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
