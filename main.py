from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pdfminer.high_level import extract_text
from flask import Flask, request, jsonify
import os
from flask_cors import CORS

__version__ = '0.2'
__author__ = 'YousefHUT'

# Cihaza göre CUDA veya CPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sohbet modelini yükle (İngilizce konuşabilen herhangi bir dil modelini uygun olacak şekilde ayarlayabilirsiniz.)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.half()  # FP16 moduna geçiş (Optimizasyon)
model.to(device)

#Dosya yazısı için olan değişteni belirleme
filetext = ""

# Çeviri pipeline'ları
translator_tr_to_en = pipeline(
    "translation",
    model="models/tr-to-en",      # Türkçe'den İngilizce
    tokenizer="models/tr-to-en",
    device=-1  # CPU üzerinde çalıştır
)

translator_en_to_tr = pipeline(
    "translation",
    model="models/en-to-tr",      # İngilizce'den Türkçe
    tokenizer="models/en-to-tr",
    device=-1  # CPU üzerinde çalıştır
)

# Sohbet geçmişi hafızası
conversation_history = []
MAX_HISTORY_LENGTH = 3  # Mesaj hatırlama sayısı

# Flask uygulaması
app = Flask(__name__)
CORS(app)

# Flask uygulaması için dosya yükleme ayarları
ALLOWED_EXTENSIONS = ['pdf','docx','txt']
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum dosya boyutu 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ekran kartı belleğini temizleme fonksiyonu
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/chat', methods=['GET'])
def chat_get():
    with open('chat.html', 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/chat', methods=['POST'])
def chat_message():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Mesaj bulunamadı."})
    user_message = data['message']
    reply_text = message(user_message)
    return jsonify({"reply": reply_text})

@app.route('/upload', methods=['POST'])
def upload_file():
    global filetext
    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı."})

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "Dosya adı boş olamaz."})

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Yalnızca .pdf, .docx ve .txt dosyaları desteklenmektedir."})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    try:
        uploaded_file.save(file_path)

        if os.path.getsize(file_path) == 0:
            return jsonify({"error": "Dosya boş. Lütfen geçerli bir dosya yükleyin."})

        filetext = extract_text(file_path)  # Dosyadan yazı çıkarma
        question = request.form.get('question', '')
        if not question:
            return jsonify({"error": "Soru alanı boş olamaz."})
        return jsonify({"answer": uploaded_file.filename, "text": filetext})

    except Exception as e:
        return jsonify({"error": f"Yazı işlenirken bir hata oluştu: {str(e)}"})

def message(prompt):
    global conversation_history
    global filetext

    # Kullanıcının mesajını Türkçe'den İngilizce'ye çevir
    translation_result = translator_tr_to_en(prompt)
    user_input_en = translation_result[0]['translation_text']

    # Kullanıcının girdisini sohbet geçmişine ekle (etiket ekleyerek)
    conversation_history.append(f"User: {user_input_en}")
    if filetext != "":
        conversation_history.append(f"PDF: {filetext}")
        conversation_history = conversation_history[-(MAX_HISTORY_LENGTH + 1):]
        filetext = ""
    else:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]

    # Sohbet geçmişini formatla (bu kısım sadece referans amaçlı, final yanıta eklenmeyecek)
    formatted_history = "\n".join(conversation_history)

    # AI için açıklama metni (Kendi dil modelinize göre ayarlayabilirsiniz)
    prompt_text = (
        "You are GaziAI, a highly capable and efficient AI support assistant. "
        "Your role is to provide a concise, focused answer to the user's inquiry using only the conversation context "
        "and any supplemental content provided (such as PDF text). "
        "Do NOT repeat any conversation labels (e.g., 'User:' or 'GaziAI') or meta instructions in your final answer. "
        "If asked about your language, reply with 'I am writing Turkish.' "
        "Answer the inquiry directly without echoing any instructions or context.\n\n"
        "Conversation context:\n" + formatted_history + "\n\nAnswer:"
    )

    # Özel çıktılar
    if user_input_en.lower().strip() in ["which language are you speaking?", "what language are you speaking?"]:
        english_response = "I am writing Turkish."
    if user_input_en.lower().strip() in ["whats your name", "Who are you"]:
        english_response = "I am GaziAI. I am an helpful AI assistant"
    else:
        # Tokenize işlemi
        inputs_tok = tokenizer(prompt_text, return_tensors="pt")
        inputs_tok = {k: v.to(device) for k, v in inputs_tok.items()}
        inputs_tok["input_ids"] = inputs_tok["input_ids"].clamp(0, tokenizer.vocab_size - 1)

        # Modelden İngilizce yanıt üretimi
        with torch.no_grad():
            outputs = model.generate(
                **inputs_tok,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.5,
                repetition_penalty=1.5,
                top_k=30,
                top_p=0.8,
                eos_token_id=[tokenizer.eos_token_id]
            )

        # Üretilen yanıtı decode etme
        english_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Yanıtı temizleme (etiketleri kaldırma)
        if "Answer:" in english_response:
            english_response = english_response.split("Answer:")[-1].strip()

    # Üretilen İngilizce cevabı Türkçe'ye çevirme
    translation_response = translator_en_to_tr(english_response)
    turkish_response = translation_response[0]['translation_text']
    return turkish_response


if __name__ == "__main__":
    app.run(debug=False)
    clear_gpu_memory()