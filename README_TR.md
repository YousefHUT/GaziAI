# GaziAI
GaziAI basit soruları cevaplayan ilkel bir yapay zeka asistanıdır. Kendisi [TinyLamma](https://huggingface.co/TinyLlama/TinyLlama-1.1B-step-50K-105b), [opus-mt-tr-en](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en) ve [opus-mt-tc-big-en-tr](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr) modellerini kullanıyor. İsteddiğiniz modele göre ayarlayıp istediğiniz gibi ayarlayabilirsiniz.
![resim](https://github.com/user-attachments/assets/ec5f49c9-cafa-4d3f-babc-5774439e2d04)

## installation:
Gerekli kütüphaneleri kurmak için requirements.txt dosyasını kullanın.
```
pip install requirements.txt
```
Çeviri için kullanılan AI modellerini kurun.
- [opus-mt-tr-en](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en)
- [opus-mt-tc-big-en-tr](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr)
- Model dosyalarını indirin ve /models klasörüne taşıyın.
- Model klasörlerinin adlarını "tr-to-en" ve "en-to-tr" olarak görevlerine göre değiştirin.
- main.py dosyasını çalıştırın.
```
python main.py
```
Tarayıcınızı açın ve sunucunuzun ip ve portuyla giriş yapın. (http://127.0.0.1:5000)

