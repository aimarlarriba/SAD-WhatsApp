import os

os.environ['PYTHONHASHSEED'] = '0'
import random
import numpy as np

# Congelamos toda la aleatoriedad global
random.seed(42)
np.random.seed(42)
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from langdetect import detect, LangDetectException, DetectorFactory

# Fijamos la semilla del detector de idiomas para que no sea aleatorio
DetectorFactory.seed = 0

# Inicializamos el lematizador
lemmatizer = WordNetLemmatizer()

def es_ingles(texto):
    try:
        return detect(str(texto)) == 'en'
    except LangDetectException:
        return False

def limpieza_temas(texto, stop_words):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-z\s]', '', texto)

    tokens = []
    for t in texto.split():
        if t not in stop_words and len(t) > 2:
            # Lematización: verbos, sustantivos, adjetivos y adverbios
            lema = lemmatizer.lemmatize(t, pos='v')
            lema = lemmatizer.lemmatize(lema, pos='n')
            lema = lemmatizer.lemmatize(lema, pos='a')
            lema = lemmatizer.lemmatize(lema, pos='r')
            tokens.append(lema)
    return tokens

# Todo el código de ejecución debe ir aquí dentro por el multiprocesamiento en Windows
if __name__ == '__main__':
    print("[*] Leyendo train_opiniones.csv para generar la gráfica...")
    try:
        df = pd.read_csv('train_opiniones.csv')
    except FileNotFoundError:
        print("[!] Error: No se encuentra 'train_opiniones.csv'.")
        exit(1)

    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


    df = df[df['content'].apply(es_ingles)].copy()

    # Mismas stopwords que en el clustering final
    stop_words = set(stopwords.words('english')).union({
        'whatsapp', 'telegram', 'app', 'application', 'send',
        'message', 'messages', 'can', 'just', 'like', 'im',
        'get', 'even', 'one', 'would', 'really', 'phone',
        'please', 'dont', 'cant', 'something', 'thing', 'know',
        'make', 'use', 'good', 'nice', 'great', 'best', 'excellent',
        'amazing', 'awesome', 'bad', 'worst', 'terrible', 'love',
        'hate', 'better', 'well', 'always', 'lot', 'could', 'take',
        'part', 'find', 'definitely', 'much', 'many', 'also',
        'overall', 'stand', 'arent', 'without', 'feel', 'time',
        'long', 'new', 'issue', 'problem', 'people', 'work', 'say',
        'try', 'want', 'give'
    })

    print("[*] Limpiando y lematizando texto...")
    df['tokens'] = df['content'].apply(lambda x: limpieza_temas(x, stop_words))
    # Quitamos las filas que se hayan quedado vacías de texto
    df = df[df['tokens'].map(len) > 0]

    rango_temas = [2, 3, 4, 5, 6, 7, 8]

    plt.figure(figsize=(10, 6))
    colores = {'positivo': 'deeppink', 'negativo': 'indigo'}

    # Iteramos separando los sentimientos para tener métricas reales
    for sent in ['positivo', 'negativo']:
        print(f"\n[*] Calculando coherencia para reseñas: {sent.upper()}")
        df_subset = df[df['sentiment'] == sent]

        if df_subset.empty:
            print(f"[!] No hay datos para el sentimiento {sent}.")
            continue

        # Diccionario y corpus exclusivos de este sentimiento
        id2word = corpora.Dictionary(df_subset['tokens'])
        id2word.filter_extremes(no_below=5, no_above=0.5)
        corpus = [id2word.doc2bow(text) for text in df_subset['tokens']]

        coherencias = []

        for k in rango_temas:
            print(f"    -> Probando con {k} temas...")
            lda_model = LdaModel(
                corpus=corpus,
                num_topics=k,
                id2word=id2word,
                random_state=42,
                passes=10,
                alpha='auto',
                eta='auto'
            )

            # Calculamos Coherencia C_V
            coherence_model_lda = CoherenceModel(model=lda_model, texts=df_subset['tokens'], dictionary=id2word,
                                                 coherence='c_v')
            coherencias.append(coherence_model_lda.get_coherence())

        # Pintamos la línea de este sentimiento en la gráfica
        plt.plot(rango_temas, coherencias, marker='o', color=colores[sent], linestyle='-', linewidth=2,
                 label=f'Sentimiento: {sent.upper()}')

    # Visual de la gráfica
    plt.title('Evaluación de Tópicos en LDA (Coherencia C_v)')
    plt.xlabel('Número de Tópicos (K)')
    plt.ylabel('Coherencia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    print("\n[OK] ¡Listo!.")