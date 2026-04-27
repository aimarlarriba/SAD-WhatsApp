import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel

print("[*] Leyendo train_opiniones.csv para generar la gráfica...")
try:
    df = pd.read_csv('train_opiniones.csv')
except FileNotFoundError:
    print("[!] Error: No se encuentra 'train_opiniones.csv'.")
    exit(1)

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english')).union({
    'whatsapp', 'telegram', 'app', 'application', 'send',
    'message', 'messages', 'can', 'just', 'like', 'im',
    'get', 'even', 'one', 'would', 'really', 'phone'
})


def limpieza_temas(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-z\s]', '', texto)
    # Lista de palabras
    tokens = [t for t in texto.split() if t not in stop_words and len(t) > 2]
    return tokens


# Aplicamos la limpieza
df['tokens'] = df['content'].apply(limpieza_temas)

# 3. PREPARAR DATOS PARA GENSIM
id2word = corpora.Dictionary(df['tokens'])
# Filtramos palabras muy raras o demasiado comunes
id2word.filter_extremes(no_below=2, no_above=0.9)
corpus = [id2word.doc2bow(text) for text in df['tokens']]

# 4. ITERACIÓN PARA BUSCAR EL NÚMERO ÓPTIMO CON COHERENCIA
rango_temas = [2, 3, 4, 5, 6, 7, 8]
coherencias = []

print("[*] Calculando modelos LDA en Gensim...")
for k in rango_temas:
    print(f"    -> Probando con {k} temas...")
    lda_model = LdaModel(corpus=corpus, num_topics=k, id2word=id2word, random_state=42, passes=10)

    # Calculamos Coherencia C_V
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df['tokens'], dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherencias.append(coherence_lda)

# 5. DIBUJAR LA GRÁFICA
plt.figure(figsize=(8, 5))
plt.plot(rango_temas, coherencias, marker='o', color='deeppink', linestyle='-', linewidth=2)
plt.title('Evaluación de Tópicos en LDA (Coherencia C_v)')
plt.xlabel('Número de Tópicos (K)')
plt.ylabel('Coherencia')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()