# Kütüphaneler
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import warnings
import re
import gc
import os

# Makine Öğrenmesi Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Diğer Kütüphaneler
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
import tensorflow_hub as hub
import fasttext

# --- Ayarlar ve İndirmeler ---
warnings.filterwarnings("ignore")
for package in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{package}')
    except LookupError:
        print(f"NLTK '{package}' paketi indiriliyor...")
        nltk.download(package, quiet=True)
        
# --- YARDIMCI FONKSİYONLAR (GLOBAL KAPSAMDA) ---
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub('[^a-zA-ZğĞçÇşŞüÜöÖıİ]', ' ', text).lower()
    return ' '.join([word for word in text.split() if word not in stopwords.words('turkish')])

def translate_to_english(text):
    if not text.strip(): return text
    try:
        return GoogleTranslator(source='turkish', target='english').translate(text)
    except Exception:
        return text

# --- 1. VERİ HAZIRLAMA (AKILLI ÖNBELLEKLEME İLE) ---
PROCESSED_DATA_PATH = "data/preprocessed_data.pkl"  # Auto-cached after first run

if os.path.exists(PROCESSED_DATA_PATH):
    print(f" Önceden işlenmiş veri bulundu! '{PROCESSED_DATA_PATH}' dosyasından yükleniyor...")
    df = pd.read_pickle(PROCESSED_DATA_PATH)
    print("Veri yükleme ve ön işleme adımları atlandı.")
else:
    print("Önceden işlenmiş veri bulunamadı. Veri sıfırdan işlenecek...")
    
    print("Veri yükleniyor...")
    df_raw = pd.read_excel("data/raw_dataset.xlsx"  # Place your dataset here)
    print(f"Başlangıçtaki toplam evrak kaydı (tekrarlar dahil): {len(df_raw)}")
    
    df_raw['BirlesikMetin'] = df_raw['EvrakKonusu'].fillna('') + ". " + df_raw['EvrakIcerigi'].fillna('') #Konu ve içeriği birleştir
    df_raw = df_raw.dropna(subset=['BirlesikMetin', 'EvrakinGeregineKisiKullaniciBirimi'])  # Birimi olmayan satırları sil
    df_raw = df_raw[df_raw['BirlesikMetin'].str.strip() != ''] # Birleşik metinde (konu + içerik) boş olanları alma

    print("Duyuru niteliğindeki evraklar filtreleniyor...")
    metin_birim_sayisi = df_raw.groupby('BirlesikMetin')['EvrakinGeregineKisiKullaniciBirimi'].nunique() # Aynı birleşik metne sahip tüm satırları biraraya getir. Benzersiz birimleri say
    Q1, Q3 = metin_birim_sayisi.quantile(0.25), metin_birim_sayisi.quantile(0.75)
    dinamik_esik_degeri = max(3, int(Q3 + 1.5 * (Q3 - Q1)))
    print(f"Dinamik 'Duyuru Eşik Değeri': {dinamik_esik_degeri}")
    
    duyuru_metinleri = metin_birim_sayisi[metin_birim_sayisi > dinamik_esik_degeri].index   # Eşik değerini geçen metinleri bul
    df_filtrelenmis = df_raw[~df_raw['BirlesikMetin'].isin(duyuru_metinleri)]               # Eşik değerini geçen metinleri veri setinden çıkar
    
    df = df_filtrelenmis.drop_duplicates(subset="EvrakId").copy()                           #Tekrar eden kayıtları sil
    print(f"Temizlenmiş ve tekilleştirilmiş nihai veri seti boyutu: {len(df)}")
            
    print("\nMetinler ön işleniyor ve İngilizce'ye çevriliyor...")
    df['MessageOnislenmis'] = df['BirlesikMetin'].apply(preprocess_text)  # Önişleme tabi tut
    df['English'] = df['MessageOnislenmis'].apply(translate_to_english)   # Önişlemden geçmiş verileri ingilizceye çevir
    print("Kök ve yalın hal (Lemma/Stem) oluşturuluyor...")
    df['lemmatized_text'] = df['English'].apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(word) for word in x.split()]))  # İngilizcesi alınmış metinlerin lemma kökünü al
    df['stemmed_text'] = df['English'].apply(lambda x: ' '.join([PorterStemmer().stem(word) for word in x.split()]))              # İngilizcesi alınmış metinlerin stem kökünü al

    df.to_pickle(PROCESSED_DATA_PATH)  #df dataframe'i kaydet ki tektar ön işleme, çeviri ve kök işlemleri yapılmasın
    print(f" İşlenmiş veri, sonraki çalıştırmalar için '{PROCESSED_DATA_PATH}' dosyasına kaydedildi.")

# --- 2. ETİKETLEME VE MODELLERİN TANIMLANMASI ---
df = df.dropna(subset=['EvrakinGeregineKisiKullaniciBirimi'])
unit_labels = {unit: idx for idx, unit in enumerate(df['EvrakinGeregineKisiKullaniciBirimi'].unique())}   #Verisetindeki tüm benzersiz birim isimlerine 0'dan başlayarak indeks atar. indeks'i value olan ve birim ismini key olan bir sözlük yapısı oluşturur.
idx_to_unit = {i: unit for unit, i in unit_labels.items()}    # oluşan sözlüğü tersine çeviriyor yani sözlükteki indeksi key'e, birim isimlerini value'e çevirir.
df['unit_label'] = df['EvrakinGeregineKisiKullaniciBirimi'].map(unit_labels) #Verisetine unit_label sütunu ekleyerek ve satırdaki birim isminin karşılığı olan indeksi yazar.

model_loaders = {   # Modelleri yükle, bazıları lokalde, bazıları indiriliyor
    "BERT": lambda: SentenceTransformer('bert-base-nli-mean-tokens'),
    "ALBERT": lambda: SentenceTransformer('sentence-transformers/paraphrase-albert-base-v2'),
    "RoBERTa": lambda: SentenceTransformer('roberta-base-nli-mean-tokens'),
    "GloVe": lambda: KeyedVectors.load_word2vec_format("models/GloVe/glove.840B.300d-001.txt", binary=False, no_header=True),
    "Word2Vec": lambda: KeyedVectors.load_word2vec_format("models/Word2Vec/word2vec-google-news-300.bin", binary=True),
    "USE": lambda: hub.load("models/USE"),
    "FastText": lambda: fasttext.load_model("models/FastText/cc.en.300.bin")
}
model_types_map = {name: name.lower() for name in model_loaders.keys()} #Model isimlerini küçük harfe çevirerek (BERT->bert, USE->use) daha sonra get_embedding_dim ve get_single_embedding fonksiyonlarında kullanmak
            """
             #get_embedding_dim ve get_single_embedding fonksiyonları hangi modelle çalıştıklarını anlamak için şöyle kontrol yapar ve burada model isimleri küçüktür:
             
            if model_type in ['bert', 'albert', 'roberta']:
                # ... BERT benzeri bir model için bu kodu çalıştır
            elif model_type == 'use':
                # ... USE modeli için bu kodu çalıştır
            """

def get_classifiers():  # Dil Modelleri belgenin anlamını okuyup bunu evrensel bir sayısal koda (vektöre) çevirir. Sınıflandırıcılar ise bu vektörleri alıp asıl kararı verir.
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(random_state=42)
    }

# --- 3. GÖMME FONKSİYONLARI (TÜM MODELLERLE UYUMLU) ---
def get_embedding_dim(model, model_type):                                                                    # Her modelin ürettiği vektörün boyutunu (uzunluğunu) bulur.
    if model_type in ['bert', 'albert', 'roberta']: return model.get_sentence_embedding_dimension()
    elif model_type == 'use': return model(["test sentence"]).numpy().shape[1]
    elif model_type == 'fasttext': return model.get_dimension()
    elif model_type in ['glove', 'word2vec']: return model.vector_size
    raise ValueError(f"Desteklenmeyen model türü: {model_type}")

def get_single_embedding(sentence, model, model_type):                                                       # Tek bir metin cümlesini vektöre dönüştürür.
    if not isinstance(sentence, str): sentence = ""
    if model_type == 'use': return model([sentence]).numpy()[0]
    elif model_type in ['bert', 'albert', 'roberta']: return model.encode([sentence], show_progress_bar=False)[0]
    elif model_type == 'fasttext': return model.get_sentence_vector(sentence)
    elif model_type in ['glove', 'word2vec']:
        words = sentence.split()
        vectors = [model[word] for word in words if model.has_index_for(word)]
        if vectors: return np.mean(vectors, axis=0)
        else: return np.zeros(get_embedding_dim(model, model_type))
    raise ValueError(f"Desteklenmeyen model türü: {model_type}")

def get_text_embeddings(texts, model, model_type):                                                            # Bir liste dolusu metni en verimli şekilde vektörlere çevirir
    if model_type == 'use': return model(texts).numpy()
    elif model_type in ['bert', 'albert', 'roberta']: return model.encode(texts, show_progress_bar=False)
    else: return np.array([get_single_embedding(text, model, model_type) for text in texts])

# --- 4. ANA EĞİTİM DÖNGÜSÜ (3 KATMANLI) ---
preprocess_types = {'English': 'English', 'Lemma': 'lemmatized_text', 'Stem': 'stemmed_text'}
results = []
trained_classifiers = {}

for model_name, loader in model_loaders.items():   #Herbir model için işlemi yap sonr modeli sil ve gc.collect ile belleği sil
    print(f"\n{'#'*60}\n# DİL MODELİ YÜKLENİYOR: {model_name}\n{'#'*60}")
    try:
        model = loader()
        model_type = model_types_map[model_name]
    except Exception as e:
        print(f"HATA: {model_name} modeli yüklenemedi: {e}\nBu model atlanıyor.")
        continue
    
    for p_name, p_col in preprocess_types.items():
        print(f"\n--- Metin Türü: {p_name} ({model_name}) ---")
        current_df = df[df[p_col].notna() & (df[p_col] != '')].copy()
        X_text = current_df[p_col]
        y = current_df['unit_label']
        X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"  -> Vektörler oluşturuluyor...")
        X_train_vectors = get_text_embeddings(X_train_text.tolist(), model, model_type)
        X_test_vectors = get_text_embeddings(X_test_text.tolist(), model, model_type)
        
        # HATA DÜZELTMESİ: Her döngüde yeni sınıflandırıcı nesneleri oluştur
        classifiers_to_test = get_classifiers()
        
        for clf_name, classifier in classifiers_to_test.items():
            print(f"    -> Sınıflandırıcı deneniyor: {clf_name}")
            classifier.fit(X_train_vectors, y_train)
            y_pred = classifier.predict(X_test_vectors)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, target_names=[str(k) for k in unit_labels.keys()], zero_division=0)
            
            results_key = f"{model_name} - {p_name} - {clf_name}"
            results.append([results_key, report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score'], accuracy])
            trained_classifiers[results_key] = classifier
            
            print(f"       {results_key} -> F1-Score: {report['weighted avg']['f1-score']:.2%}, Accuracy: {accuracy:.2%}")

    print(f"\n--- {model_name} modeli bellekten temizleniyor ---")
    del model
    gc.collect()

# --- 5. SONUÇLARI GÖSTERME VE KAYDETME ---
if results:
    df_results = pd.DataFrame(results, columns=["Kombinasyon", "Precision", "Recall", "F1-Score", "Accuracy"])
    df_results = df_results.sort_values(by=["F1-Score", "Accuracy"], ascending=[False, False])
    
    df_results.to_excel("results/Final_Supervised_Performance.xlsx", index=False)
    print("\n" + "="*80)
    print("--- FİNAL PERFORMANS RAPORU (Denetimli Öğrenme Metodu) ---")
    print("="*80)
    print(df_results.head(10))

# --- 6. TAHMİN FONKSİYONU VE TEST ---
def predict_unit_with_best_model(document_text, best_combination_key):
    model_name, p_name, clf_name = best_combination_key.split(' - ')
    print(f"\nTahmin için en iyi kombinasyon kullanılıyor:\n  Dil Modeli: {model_name}\n  Metin Türü: {p_name}\n  Sınıflandırıcı: {clf_name}")
    
    model = model_loaders[model_name]()
    model_type = model_types_map[model_name]
    classifier = trained_classifiers[best_combination_key]

    processed_text = preprocess_text(document_text)
    english_text = translate_to_english(processed_text)
    
    final_text = ""
    if p_name == 'English': final_text = english_text
    elif p_name == 'Lemma': final_text = ' '.join([WordNetLemmatizer().lemmatize(word) for word in english_text.split()])
    elif p_name == 'Stem': final_text = ' '.join([PorterStemmer().stem(word) for word in english_text.split()])
    
    vector = get_single_embedding(final_text, model, model_type)
    predicted_label = classifier.predict([vector])[0]
    
    del model
    gc.collect()
    
    return idx_to_unit.get(predicted_label, "Birim Bulunamadı")

if results:
    print("\n--- NİHAİ TEST TAHMİNİ ---")
    document_text = "Üniversite için gerekli olan bilişim altyapısı ve yazılım ihtiyaçları konulu bu belge bilgi işlem birimine iletilmelidir."
    best_combination_key = df_results.iloc[0]["Kombinasyon"]
    
    predicted_unit = predict_unit_with_best_model(document_text, best_combination_key)
    print(f"'{document_text[:80]}...'")
    print(f" Tahmin Edilen Birim: {predicted_unit}")