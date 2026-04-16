import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import re

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
ai_model = genai.GenerativeModel('gemma-3-4b-it')

app = Flask(__name__)
CORS(app)

def get_ai_styled_response(prompt, style_class):
    """Generate AI response with formatting"""
    try:
        full_prompt = prompt + ". VERY IMPORTANT: Keep response under 50 words, use simple language, and keep it friendly."
        response = ai_model.generate_content(full_prompt)
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
        return f'<div class="ai-response-container">😵 AI error: {str(e)}</div>'

@app.route('/chat_and_search', methods=['POST'])
def chat_and_search():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        prompt = f"""
        You are an AI Shopping Assistant. Analyze: "{user_message}"
        Tasks:
        1. Reply: max 10 words.
        2. Extract 1 keyword (Bottle, Bowl, Cup, or a spec like 1000ml).
        Format as STRICT JSON:
        {{
            "reply": "...",
            "search_keyword": "..."
        }}
        """
        response = ai_model.generate_content(prompt) 
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            return jsonify(json.loads(json_match.group()))
        else:
            return jsonify({"reply": response.text, "search_keyword": None})
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"reply": "I'm having a bit of trouble.", "search_keyword": None}), 500

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
        'model_loaded': True  
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
