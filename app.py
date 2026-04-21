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
        if "VERY IMPORTANT" not in prompt:
            prompt = prompt + " Keep response under 50 words, use simple language, and keep it friendly."
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
        return f'<div class="ai-response-container">😵 AI error: {str(e)}</div>'

@app.route('/chat_and_search', methods=['POST'])
def chat_and_search():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        prompt = f"""
        Analyze: "{user_message}"
        
        Return ONLY valid JSON, no other text.
        
        Rules:
        1. Reply: friendly, max 8 words.
        2. search_keyword: Extract 1-2 MOST IMPORTANT words ONLY.
           - Include product type (bottle/cup/bowl/tumbler/mug)
           - Include key attributes if mentioned (BPA-free, ceramic, 1000ml, stainless, insulated, durable)
           - DO NOT include full sentences or more than 2 words
           - Translate Malay to English:
             * "botol" → bottle
             * "cawan" → cup
             * "mangkuk" → bowl
             * "tahan panas" → heat resistant
             * "tidak mudah pecah" → durable
        
        Examples:
        - "BPA-free microwave safe bottle" → "BPA-free bottle"
        - "ceramic mug for microwave" → "ceramic mug"
        - "durable cup for kids" → "durable cup"
        - "stainless steel tumbler" → "stainless tumbler"
        - "1000ml water bottle" → "1000ml bottle"
        - "cari botol yang murah" → "bottle"
        - "cawan yang tahan panas" → "heat resistant cup"
        
        {{
            "reply": "your short reply",
            "search_keyword": "1-2 word keyword"
        }}
        """
        
        response = ai_model.generate_content(prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group())
            keyword = result.get('search_keyword', '')
            if len(keyword.split()) > 3:
                keyword = ' '.join(keyword.split()[:2])
            print(f"User: {user_message} -> Search: {keyword}")
            return jsonify({
                'reply': result.get('reply', 'Let me help you find that!'),
                'search_keyword': keyword
            })
        else:
            return jsonify({"reply": "Let me help you find that!", "search_keyword": None})
            
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
        seller = p.get('seller_name', 'Unknown seller')
        price = p.get('price', 'N/A')
        rating = p.get('rating', 'N/A')
        context += f"P{i+1}: {p['name']}, Seller: {seller}, Price: RM{price}, Rating: {rating}, Specs: {p['specs']}\n"
    
    prompt = f"""Compare these items:\n{context}
    
    Please provide a detailed comparison with the following structure:
    
    1. **Specs Showdown**: Compare the key specifications of each product. Highlight differences in material, capacity, features, etc.
    
    2. **Seller & Price Comparison**: Compare the sellers (official store vs authorized reseller vs regular store), price differences, and any seller-related information from reviews.
    
    3. **Sentiment Analysis**: Summarize customer sentiment based on the reviews provided. Note any positive or negative feedback.
    
    4. **Winner**: Based on specs, price, seller reliability, and customer sentiment, declare a winner and explain why.
    
    Keep the response informative but friendly. Use bullet points or short paragraphs for readability."""
    
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
