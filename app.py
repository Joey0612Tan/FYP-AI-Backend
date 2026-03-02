import numpy as np
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify
from flask_cors import CORS
import io, re
from PIL import Image
import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
ai_model = genai.GenerativeModel('gemma-3-4b-it') 

app = Flask(__name__)
CORS(app)

print("⌛ Loading ResNet50 for real-time matching...")
# cv_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
vector_db = np.load('product_vectors.npy', allow_pickle=True)

def get_ai_styled_response(prompt, style_class):
    try:
        response = ai_model.generate_content(prompt)
        res_text = response.text
        
        res_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', res_text)
        res_text = res_text.replace('*', '') 
        res_text = res_text.replace('Overall Sentiment', '📊 Overall Sentiment')
        res_text = res_text.replace('Pros', '✅ Pros').replace('Cons', '❌ Cons')
        res_text = res_text.replace('Final Verdict', '⚖️ Final Verdict')
        res_text = res_text.replace('Key Highlights', '🌟 Key Highlights')
        res_text = res_text.replace('Usage Scenarios', '🎯 Usage Scenarios')
        res_text = res_text.replace('Expert Tip', '💡 Expert Tip')
        res_text = res_text.replace('The Winner', '🏆 The Winner')
        res_text = res_text.replace('Specs Showdown', '⚔️ Specs Showdown')
        res_text = res_text.replace('\n', '<br>')
        
        styled_res = f'''
        <div class="ai-response-container {style_class}">
            {res_text}
        </div>
        '''
        return styled_res
    except Exception as e:
        return f'<div class="ai-response-container">😵 AI is dizzy: {str(e)}</div>'

# @app.route('/visual_search', methods=['POST'])
# def visual_search():
#     if 'image' not in request.files: return jsonify({"error": "No image"}), 400
#     try:
#         file = request.files['image']
#         img = Image.open(io.BytesIO(file.read())).convert('RGB').resize((224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         query_vec = cv_model.predict(x, verbose=0).flatten()

#         all_results = []
#         for item in vector_db:
#             similarity = 1 - cosine(query_vec, item['vector'])
#             if similarity > 0.5:
#                 all_results.append({"pid": str(item['pid']), "score": float(similarity)})
        
#         all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:10]
#         return jsonify({"status": "success", "matches": [r['pid'] for r in all_results], "top_score": all_results[0]['score'] if all_results else 0})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/visual_search', methods=['POST'])
def visual_search():
    return jsonify({"status": "success", "matches": [], "message": "Visual search is offline for maintenance"})

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
