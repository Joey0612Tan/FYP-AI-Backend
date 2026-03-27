import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import io, re, os
from PIL import Image
import google.generativeai as genai
import onnxruntime as ort

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
ai_model = genai.GenerativeModel('gemma-3-4b-it') 

app = Flask(__name__)
CORS(app) 

print("⌛ Loading Optimized ResNet50 (ONNX) and Vector DB...")

try:
    session = ort.InferenceSession("resnet50_final.onnx")
    input_name = session.get_inputs()[0].name
    print("✅ ONNX Model Loaded")
except Exception as e:
    print(f"❌ ONNX Model Error: {e}")
    session = None

try:
    vector_db = np.load('product_vectors.npy', allow_pickle=True)
    print(f"✅ Vector DB Loaded: {len(vector_db)} products")
    print(f"   Sample product IDs: {[item['pid'] for item in vector_db[:5]]}")
except Exception as e:
    print(f"❌ Vector DB Error: {e}")
    vector_db = []

def get_ai_styled_response(prompt, style_class):
    try:
        response = ai_model.generate_content(prompt)
        res_text = response.text
        res_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', res_text)
        res_text = res_text.replace('*', '') 
        replacements = {
            'Overall Sentiment': '📊 Overall Sentiment',
            'Pros': '✅ Pros', 'Cons': '❌ Cons',
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
        return f'<div class="ai-response-container">😵 AI is dizzy: {str(e)}</div>'

def extract_features_from_image(image_file):
    """Extract feature vectors from uploaded images"""
    try:
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        img = img.resize((224, 224))
        img_data = np.array(img).astype('float32')
        img_data = img_data / 255.0
        img_data = np.expand_dims(img_data, axis=0)
        
        features = session.run(None, {input_name: img_data})[0].flatten()
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

@app.route('/visual_search', methods=['POST'])
def visual_search():
    print("--- VISUAL SEARCH REQUEST ---")
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        file = request.files['image']
        print(f"Received image: {file.filename}, size: {len(file.read())} bytes")
        file.seek(0)  
        
        query_vec = extract_features_from_image(file)
        
        if query_vec is None:
            return jsonify({"status": "error", "error": "Failed to extract features"}), 500
        
        print(f"Query vector norm: {np.linalg.norm(query_vec)}")
        
        all_results = []
        for item in vector_db:
            db_vec = item['vector']
            if isinstance(db_vec, list):
                db_vec = np.array(db_vec)
            
            norm_a = np.linalg.norm(query_vec)
            norm_b = np.linalg.norm(db_vec)
            
            if norm_a > 0 and norm_b > 0:
                similarity = np.dot(query_vec, db_vec) / (norm_a * norm_b)
                
                if similarity > 0.2:  # 从 0.4 降到 0.2
                    all_results.append({
                        "pid": int(item['pid']), 
                        "score": float(similarity)
                    })
                    print(f"  Product {item['pid']}: similarity = {similarity:.4f}")
        
        all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:10]
        
        print(f"Found {len(all_results)} matches")
        
        if not all_results:
            print("No matches found")
            return jsonify({"status": "no_match", "matches": []})
        
        matches = [r['pid'] for r in all_results]
        top_score = all_results[0]['score']
        
        print(f"Top match: ID={matches[0]}, Score={top_score:.4f}")
        print(f"All matches: {matches[:5]}")
        
        return jsonify({
            "status": "success", 
            "matches": matches, 
            "top_score": top_score
        })

    except Exception as e:
        print(f"Visual Search Error: {e}")
        import traceback
        traceback.print_exc()
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'products_loaded': len(vector_db),
        'model_loaded': session is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}")
    print(f"Vector DB contains {len(vector_db)} products")
    app.run(host='0.0.0.0', port=port)
