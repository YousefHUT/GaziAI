from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pdfminer.high_level import extract_text
from flask import Flask, request, jsonify
import os
from flask_cors import CORS

__version__ = '0.1'
__author__ = 'YousefHUT'

# Cihaza göre CUDA veya CPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sohbet modelini yükle (İngilizce konuşabilen herhangi bir dil modelini uygun olacak şekilde ayarlayabilirsiniz.)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.half()  # FP16 moduna geçiş (Optimizasyon)
model.to(device)

#PDF yazısı için olan değişteni belirleme
pdftext = ""

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
        return jsonify({"error": "Mesaj bulunamadı."}), 400
    user_message = data['message']
    reply_text = message(user_message)
    return jsonify({"reply": reply_text})

@app.route('/upload', methods=['GET'])
def upload_form():
    return '''
    <!doctype html>
    <title>Dosya Yükle</title>
    <h1>Dosya Yükle</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="text" name="question" placeholder="Sorunuzu buraya yazın">
      <input type="submit" value="Upload">
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    global pdftext
    uploaded_file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(file_path)
    pdftext = extract_file_text(file_path)  # PDF'ten yazı çıkarma
    question = request.form['question']
    answer = message(question)
    return jsonify({"answer": answer})

def message(prompt):
    global conversation_history
    global pdftext

    # Kullanıcının mesajını Türkçe'den İngilizce'ye çevir
    translation_result = translator_tr_to_en(prompt)
    user_input_en = translation_result[0]['translation_text']

    # Kullanıcının girdisini sohbet geçmişine ekle (etiket ekleyerek)
    conversation_history.append(f"User: {user_input_en}")
    if pdftext != "":
        conversation_history.append(f"PDF: {pdftext}")
        conversation_history = conversation_history[-(MAX_HISTORY_LENGTH + 1):]
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
    if "Answer:" in english_response:
        english_response = english_response.split("Answer:")[-1].strip()

    # Özel çıktılar
    if prompt.lower().strip() in ["which language are you speaking?", "what language are you speaking?"]:
        english_response = "I am writing Turkish."
    if prompt.lower().strip() in ["whats your name", "Who are you"]:
        english_response = "I am GaziAI. I am an helpful AI assistant"

    # Üretilen İngilizce cevabı Türkçe'ye çevirme
    translation_response = translator_en_to_tr(english_response)
    turkish_response = translation_response[0]['translation_text']
    return turkish_response

def extract_file_text(file_path):
    if not os.path.exists(file_path):
        return "Dosya bulunamadı."
    if os.path.getsize(file_path) == 0:
        return "Dosya boş. Lütfen geçerli bir PDF dosyası yükleyin."
    if not os.access(file_path, os.R_OK):
        return "Dosyaya erişim izni yok. Lütfen dosyanın izinlerini kontrol edin."
    if not os.path.isfile(file_path):
        return "Geçersiz dosya yolu. Lütfen geçerli bir dosya yolu sağlayın."
    if not file_path.endswith('.pdf'):
        return "Yalnızca PDF dosyaları desteklenmektedir."
    try:
        return extract_text(file_path)
    except Exception as e:
        return f"Yazıyı çıkarırken hata oluştu {str(e)}"

if __name__ == "__main__":
    app.run(debug=False)
    clear_gpu_memory()