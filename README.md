# GaziAI
GaziAI is an simple ai assistant for explaining basic things. It uses [TinyLamma](https://huggingface.co/TinyLlama/TinyLlama-1.1B-step-50K-105b), [opus-mt-tr-en](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en) and [opus-mt-tc-big-en-tr](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr) models. You can use other models for you want.
![resim](https://github.com/user-attachments/assets/ec5f49c9-cafa-4d3f-babc-5774439e2d04)

## installation:
You need to install required libraries using requirements.txt.
```
pip install requirements.txt
```
You need translator ai models for translation between tr and english prompts.
- [opus-mt-tr-en](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en)
- [opus-mt-tc-big-en-tr](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr)
- Download model files and move to /models folder.
- rename model folder names as "tr-to-en" and "en-to-tr".
- Run main.py
```
python main.py
```
Open your browser and enter to your local servers ip and port. (http://127.0.0.1:5000)
