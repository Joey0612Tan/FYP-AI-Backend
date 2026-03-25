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
    vector_db = np.load('product_vectors.npy', allow_pickle=True)
    print("✅ System Ready: ONNX Model & Vector DB Loaded.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

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

@app.route('/visual_search', methods=['POST'])
def visual_search():
    print("--- RECEIVED REQUEST ---") # 如果 Render 日志没印这个，就是 403 挡在门外了
    if 'image' not in request.files:
        return jsonify({"error": "No image"}), 400
    
    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((224, 224))
        img_data = np.array(img).astype('float32')
        img_data /= 255.0
        img_data = np.expand_dims(img_data, axis=0) # (1, 224, 224, 3)

        query_vec = session.run(None, {input_name: img_data})[0].flatten()

        all_results = []
        for item in vector_db:
            norm_a = np.linalg.norm(query_vec)
            norm_b = np.linalg.norm(item['vector'])
            similarity = np.dot(query_vec, item['vector']) / (norm_a * norm_b)
            
            if similarity > 0.4: 
                all_results.append({"pid": str(item['pid']), "score": float(similarity)})
        
        all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:10]
        
        if not all_results:
            return jsonify({"status": "no_match", "matches": []})
            
        return jsonify({
            "status": "success", 
            "matches": [r['pid'] for r in all_results], 
            "top_score": all_results[0]['score']
        })

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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
