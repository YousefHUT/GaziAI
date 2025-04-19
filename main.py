from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pdfminer.high_level import extract_text
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from docx import Document

__version__ = '0.3'
__author__ = 'YousefHUT (Yusuf Eren HUT)'

print("GaziAI - Gaziosmanpaşa Üniversitesi Yapay Zeka Destekli Sohbet Botu programı başlatılıyor...")
print("Yazılım versiyonu:", __version__)
print("Yazılım geliştiricisi:", __author__)

# Sohbet geçmişi hafızası
conversation_history = []
MAX_HISTORY_LENGTH = 3  # Mesaj hatırlama sayısı

# Flask uygulaması
app = Flask(__name__)
CORS(app)

# Flask uygulaması için dosya yükleme ayarları
SAVEFILES = True
ALLOWED_EXTENSIONS = ['pdf','docx','txt']
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum dosya boyutu 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # Yükleme klasörünü oluştur
# Önceki yükleme klasörünü temizle
if os.path.exists(app.config['UPLOAD_FOLDER']) and not SAVEFILES:
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        if os.path.isfile(file_path):
            os.remove(file_path)
filetext = "" #Dosya yazısı için olan değişkeni belirleme

# Cihaza göre CUDA veya CPU kullanımı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sohbet modelini yükle (İngilizce konuşabilen herhangi bir dil modelini uygun olacak şekilde ayarlayabilirsiniz.)
print("Modeller yükleniyor...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.half()  # FP16 moduna geçiş (Optimizasyon)
model.to(device)

# Çeviri pipeline'ları
# https://huggingface.co/Helsinki-NLP/opus-mt-tr-en
# https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr
# Model dosyalarını yüklemek için Hugging Face üzerinden indirip "models" klasörüne doğru şekilde yerleştirin.
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
print("Modeller yüklendi.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ekran kartı belleğini temizleme fonksiyonu
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# Ana sayfa ve sohbet sayfası için route'lar
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

        # Dosya türüne göre yazı çıkarma
        if uploaded_file.filename.lower().endswith('.pdf'):
            filetext = extract_text(file_path)  # PDF dosyasından yazı çıkarma
        elif uploaded_file.filename.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                filetext = txt_file.read()  # TXT dosyasından yazı çıkarma
        elif uploaded_file.filename.lower().endswith('.docx'):
            doc = Document(file_path)
            filetext = "\n".join([paragraph.text for paragraph in doc.paragraphs])  # DOCX dosyasından yazı çıkarma
        else:
            return jsonify({"error": "Desteklenmeyen dosya türü."})

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

    # Sohbet geçmişini formatla
    formatted_history = "\n".join(conversation_history)

    # AI için açıklama metni (Kendi dil modelinize göre ayarlayabilirsiniz)
    prompt_text = (
        "You are GaziAI. You are an helpful ai assistant. "
        "You are a helpful assistant that provides information and answers to questions. "
        "You are a large language model trained by Gaziosmanpaşa University. "
        "Your role is to provide a concise, focused answer to the user's inquiry using only the conversation context "
        "and any supplemental content provided (such as PDF text). "
        "Do NOT repeat any conversation labels (e.g., 'User:' or 'GaziAI') or meta instructions in your final answer. "
        "Answer the inquiry directly without echoing any instructions or context.\n\n"
        "Conversation context:\n" + formatted_history + "\n\nAnswer:"
    )

    # Tokenize işlemi
    inputs_tok = tokenizer(prompt_text, return_tensors="pt")
    inputs_tok = {k: v.to(device) for k, v in inputs_tok.items()}
    inputs_tok["input_ids"] = inputs_tok["input_ids"].clamp(0, tokenizer.vocab_size - 1)

    # Modelden İngilizce yanıt üretimi (Cihazınıza ve modelinize göre ayarlayabilirsiniz)
    with torch.no_grad():
        outputs = model.generate(
            **inputs_tok,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.2,
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
    print("Flask uygulaması çalışıyor!")
    app.run(debug=False)
    print("Flask uygulaması durduruldu.")
    print("GPU belleği temizleniyor...")
    clear_gpu_memory() # GPU belleğini temizle