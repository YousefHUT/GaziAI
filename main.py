from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pdfminer.high_level import extract_text
from flask import Flask, request, jsonify
import os
from flask_cors import CORS

# Desteklenen dosya uzantıları
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Cihaza göre CUDA veya CPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sohbet modelini yükle (FP16 modunda çalıştırarak bellek kullanımını azaltıyoruz)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.half()  # FP16 moduna geçiş
model.to(device)

# Çeviri pipeline'ları (CPU üzerinde çalıştırarak GPU belleğini boşaltıyoruz)
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

# Sohbet geçmişi belleği
conversation_history = []
MAX_HISTORY_LENGTH = 3  # Max mesaj sayısı

# Flask uygulaması
app = Flask(__name__)
CORS(app)

# Flask uygulaması için dosya yükleme ayarları
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum dosya boyutu 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Çıkış için ekran kartını temizle
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

@app.route('/')
def index():
    with open('index.html', 'r') as file:
        return file.read()

@app.route('/chat', methods=['GET'])
def chat():
    with open('chat.html', 'r') as file:
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
    <title>Upload PDF</title>
    <h1>Upload PDF</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=text name=question placeholder="Sorunuzu buraya yazın">
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(file_path)
    text = extract_file_text(file_path)  # Extract text from the uploaded PDF
    question = request.form['question']
    answer = message(question)
    return jsonify({"answer": answer})

def message(prompt):
    global conversation_history
    # Kullanıcının mesajını Türkçe'den İngilizce'ye çevir
    translation_result = translator_tr_to_en(prompt)
    user_input_en = translation_result[0]['translation_text']

    # Kullanıcının girdisini sohbet geçmişine ekle
    conversation_history.append(f"User: {user_input_en}")
    conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]
    # Sohbet geçmişini formatla ve cevap başlığını ekle
    formatted_history = "\n".join(conversation_history) + "\nGaziAI:"

    # Yeni prompt: Direkt olarak cevap üretmesini iste
    prompt_text = ("Your name is GaziAI, an helpful AI support assistant. Provide a concise and helpful response "
                   "to the user's query based on the following conversation, without reiterating your introduction or instructions:\n"
                   + formatted_history)
    
    # Tokenize işlemi
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = inputs["input_ids"].clamp(0, tokenizer.vocab_size - 1)

    # Modelden İngilizce cevap üretimi (torch.no_grad() ile bellek optimizasyonu)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Daha kısa yanıtlar için token sayısını düşürdük
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.5,
            top_k=50,
            top_p=0.9,
            eos_token_id=[tokenizer.eos_token_id]
        )

    # Üretilen cevap hazırlandıktan sonra decode edilip işleniyor
    english_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "GaziAI:" in english_response:
        english_response = english_response.split("GaziAI:")[-1].strip()

    # Üretilen İngilizce cevabı Türkçe'ye çevir
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
    app.run(debug=True)
    clear_gpu_memory()
