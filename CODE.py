import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from googletrans import Translator, LANGUAGES 
from textblob import TextBlob  
import os
import speech_recognition as sr
from transformers import MarianMTModel, MarianTokenizer , TrainingArguments , Trainer
from PIL import Image
import pytesseract
from langdetect import detect
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from gtts import gTTS
import torch
from torch.utils.data import Dataset

class LanguageTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Translator")
        self.root.iconbitmap("languages.ico")

        self.translator = Translator()  
        self.languages = {lang: code for code, lang in LANGUAGES.items()} # Mapping language names to language codes

        self.label_text = tk.StringVar()
        self.label_text.set("Enter text to translate:")

        self.label = ttk.Label(root, textvariable=self.label_text)
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        self.input_text = ttk.Entry(root, width=50)
        self.input_text.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)

        self.language_label = ttk.Label(root, text="Translate to:")
        self.language_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)

        self.language_combobox = ttk.Combobox(root, values=list(self.languages.values()), width=20)
        self.language_combobox.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)

        self.upload_button = ttk.Button(root, text="Upload File", command=self.upload_file)
        self.upload_button.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        self.translate_button = ttk.Button(root, text="Translate", command=self.translate_text)
        self.translate_button.grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)

        self.output_label = ttk.Label(root, text="Translated Text:")
        self.output_label.grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)

        self.output_text = tk.Text(root, width=50, height=5)
        self.output_text.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)

        self.sentiment_label = ttk.Label(root, text="Sentiment Analysis:")
        self.sentiment_label.grid(row=6, column=0, padx=10, pady=5, sticky=tk.W)

        self.sentiment_text = tk.Text(root, width=50, height=1)
        self.sentiment_text.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)

        self.search_label = ttk.Label(root, text="Search Results:")
        self.search_label.grid(row=8, column=0, padx=10, pady=5, sticky=tk.W)

        self.search_text = tk.Text(root, width=50, height=1)
        self.search_text.grid(row=9, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W)

        # Thêm nút hiển thị kết quả tìm kiếm
        self.show_search_button = ttk.Button(root, text="Show Search Results", command=self.show_search_results)
        self.show_search_button.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W+tk.E)

        # Button for converting text to speech
        self.text_to_speech_button = ttk.Button(root, text="Text to Speech", command=self.convert_to_speech)
        self.text_to_speech_button.grid(row=11, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W+tk.E)

        # Button for voice input
        self.voice_input_button = ttk.Button(root, text="Voice Input", command=self.voice_input)
        self.voice_input_button.grid(row=12, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W+tk.E)

        # Initialize SpeechRecognition recognizer
        self.recognizer = sr.Recognizer()

        # Khởi tạo tokenizer và model của mô hình MarianMT
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi")

        self.copy_button = ttk.Button(root, text="Copy Translated Text", command=self.copy_translated_text)
        self.copy_button.grid(row=13, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W+tk.E)

        self.menu_bar = tk.Menu(root)
        root.config(menu=self.menu_bar)

        # Create File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Refresh", command=self.refresh_content)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.quit)
    
        self.recognizer = sr.Recognizer()

        # Initialize tokenizer and model
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        
        self.max_length = 500  # Đặt độ dài tối đa cho mỗi phần
    def train_transformers_model(self, english_texts, vietnamese_texts):
        # Define training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_dir='./logs',
            overwrite_output_dir=True,
        )

        # Tokenize English and Vietnamese texts
        tokenized_data = self.tokenizer.prepare_seq2seq_batch(src_texts=english_texts, tgt_texts=vietnamese_texts)

        # Train Transformers model
        trainer = Trainer(
            model=self.transformers_model[0],
            args=training_args,
            train_dataset=tokenized_data['train'],
            eval_dataset=tokenized_data['validation']
        )

        trainer.train()
    def train_lstm_model(self, english_texts, vietnamese_texts):
        # Tokenize English and Vietnamese texts
        self.english_tokenizer.fit_on_texts(english_texts)
        self.vietnamese_tokenizer.fit_on_texts(vietnamese_texts)

        # Convert texts to sequences
        english_sequences = self.english_tokenizer.texts_to_sequences(english_texts)
        vietnamese_sequences = self.vietnamese_tokenizer.texts_to_sequences(vietnamese_texts)

        # Pad sequences
        english_padded = pad_sequences(english_sequences, maxlen=self.max_len, padding='post')
        vietnamese_padded = pad_sequences(vietnamese_sequences, maxlen=self.max_len, padding='post')

        # Train LSTM model
        self.lstm_model.fit(english_padded, vietnamese_padded, epochs=10, batch_size=64)

    # Other methods
    def translate_text(self):
        input_text = self.input_text.get()
        selected_language = self.language_combobox.get()
        detected_language = detect(input_text)
        target_language = self.language_combobox.get()
    
        if detected_language != target_language:
            try:
                # Translate using the pretrained Transformers model
                input_encoding = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                translated_output = self.model.generate(**input_encoding)
                translated_text = self.tokenizer.decode(translated_output[0], skip_special_tokens=True)

                # Update output text widget with translated text
                self.output_text.delete('1.0', tk.END)
                self.output_text.insert(tk.END, translated_text)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during translation:\n{str(e)}")
    
        else:
            messagebox.showinfo("Information", "Source and target languages are the same.") 

        if selected_language not in self.languages.values():
            messagebox.showerror("Error", "Please select a valid target language.")
            return
        
        target_language = next((code for code, lang in self.languages.items() if lang == selected_language), None)
        if not target_language:
            messagebox.showerror("Error", "Please select a valid target language.")
            return

        # Chia nhỏ văn bản thành các phần
        input_parts = [input_text[i:i+self.max_length] for i in range(0, len(input_text), self.max_length)]

        translated_parts = []
        for part in input_parts:
            try:
                translation = self.translator.translate(part, dest=target_language)
                translated_parts.append(translation.text)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during translation:\n{str(e)}")

        # Kết hợp các phần đã dịch thành một đoạn văn bản hoàn chỉnh
        translated_text = "\n".join(translated_parts)
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, translated_text)

        # Thực hiện phân tích cảm xúc cho toàn bộ văn bản
        blob = TextBlob(translated_text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            sentiment_text = "Positive"
        elif sentiment < 0:
            sentiment_text = "Negative"
        else:
            sentiment_text = "Neutral"

        self.sentiment_text.delete('1.0', tk.END)
        self.sentiment_text.insert(tk.END, sentiment_text)
    def translate_with_lstm(self, input_text):
        # Tokenize input text
        self.lstm_tokenizer.fit_on_texts([input_text])
        input_sequence = self.lstm_tokenizer.texts_to_sequences([input_text])
        input_sequence_padded = pad_sequences(input_sequence, maxlen=100, padding='post')

        # Translate using LSTM model
        translated_text_sequence = self.lstm_model.predict(input_sequence_padded)
        translated_text = self.lstm_tokenizer.sequences_to_texts(translated_text_sequence)[0]
        
        return translated_text
    def translate_with_transformers(self, input_text):
        # Translate using Transformers model
        input_encoding = self.transformers_tokenizer(input_text, return_tensors="pt")
        translated_output = self.transformers_model.generate(**input_encoding)
        translated_text = self.transformers_tokenizer.decode(translated_output[0], skip_special_tokens=True)
        
        return translated_text
    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
        if file_path:
            try:
                if file_path.lower().endswith(('.docx', '.pdf')):
                    text = self.extract_text_from_doc_pdf(file_path)
                elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    text = self.extract_text_from_image(file_path)
                else:
                    with open(file_path, 'r') as file:
                        text = file.read()
                self.input_text.delete(0, tk.END)
                self.input_text.insert(tk.END, text)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while reading the file:\n{str(e)}")

    def extract_text_from_doc_pdf(self, file_path):
        if file_path.lower().endswith('.docx'):
            import docx
            doc = docx.Document(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        elif file_path.lower().endswith('.pdf'):
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfFileReader(file)
                text = ''
                for page_num in range(reader.numPages):
                    text += reader.getPage(page_num).extractText()
        return text

    def extract_text_from_image(self, file_path):
        image = Image.open(file_path)
        pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        text = pytesseract.image_to_string(image)
        return text

    def show_search_results(self):
        input_text = self.input_text.get()
        if input_text:
            search_results = self.perform_search(input_text)
            self.search_text.delete('1.0', tk.END)
            self.search_text.insert(tk.END, search_results)
        else:
            messagebox.showinfo("Information", "Please enter text to search.")

    def perform_search(self, query):
        # Code for performing search based on the query
        # Replace this with your own implementation
        search_results = f"Search results for: {query}"
        return search_results

    def convert_to_speech(self):
        translated_text = self.output_text.get('1.0', tk.END)
        if translated_text:
            try:
                tts = gTTS(text=translated_text, lang='en')
                tts.save("translation.mp3")
                os.system("start translation.mp3")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during text-to-speech conversion:\n{str(e)}")
        else:
            messagebox.showinfo("Information", "Please translate a text before converting to speech.")

    def voice_input(self):
        with sr.Microphone() as source:
            self.input_text.delete(0, tk.END)
            self.input_text.insert(tk.END, "Listening...")
            self.root.update()

            audio = self.recognizer.listen(source)

            try:
                text = self.recognize_speech(audio)
                self.input_text.delete(0, tk.END)
                self.input_text.insert(tk.END, text)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during voice input:\n{str(e)}")

    def recognize_speech(self, audio):
        text = self.recognizer.recognize_google(audio)
        return text
    
    def copy_translated_text(self):
        translated_text = self.output_text.get('1.0', tk.END)
        if translated_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(translated_text)
            messagebox.showinfo("Information", "Translated text copied to clipboard.")
        else:
            messagebox.showinfo("Information", "No translated text available to copy.")

    def refresh_content(self):
        # Function to refresh the content
        self.input_text.delete('0', tk.END)
        self.output_text.delete('1.0', tk.END)
        self.sentiment_text.delete('1.0', tk.END)
        self.search_text.delete('1.0', tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageTranslatorApp(root)
    root.mainloop()
