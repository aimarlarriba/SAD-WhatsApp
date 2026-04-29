import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("[*] Leyendo train_opiniones.csv...")
try:
    df = pd.read_csv('train_opiniones.csv')
except FileNotFoundError:
    print("[!] Error: No se encuentra 'train_opiniones.csv'.")
    exit(1)

# Inicializamos el lematizador
lemmatizer = WordNetLemmatizer()


def limpieza_temas(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-z\s]', '', texto)

    stop_words = set(stopwords.words('english')).union({
        'whatsapp', 'telegram', 'app', 'application', 'send',
        'message', 'messages', 'can', 'just', 'like', 'im',
        'get', 'even', 'one', 'would', 'really', 'phone'
    })

    tokens = []
    for t in texto.split():
        if t not in stop_words and len(t) > 2:
            # Lematizamos primero como verbo y luego como sustantivo
            lema = lemmatizer.lemmatize(t, pos='v')
            lema = lemmatizer.lemmatize(lema, pos='n')
            tokens.append(lema)

    return tokens


print("[*] Limpiando y lematizando texto para análisis de tópicos...")
df['tokens'] = df['content'].apply(limpieza_temas)

print("[*] Generando Diccionario y Corpus para Gensim...")
id2word = corpora.Dictionary(df['tokens'])
# Filtramos extremos para quitar ruido
id2word.filter_extremes(no_below=2, no_above=0.9)
corpus = [id2word.doc2bow(text) for text in df['tokens']]

# Número de tópicos que mejor ha salido en la gráfica del codo
n_topics = 7
print(f"[*] Entrenando modelo LDA con {n_topics} tópicos...")

lda_model = LdaModel(corpus=corpus, num_topics=n_topics, id2word=id2word, random_state=42, passes=10)

# ==========================================
# EXTRACCIÓN Y GUARDADO DE TÓPICOS EN TXT
# ==========================================
print("\n=== DISTRIBUCIÓN DE TÓPICOS ENCONTRADOS ===")
resumen_topicos = "=== RESUMEN DE TÓPICOS (LDA) ===\n\n"

for idx, topic in lda_model.print_topics(-1):
    info_topico = f"Tópico {idx}: {topic}"
    print(info_topico)
    resumen_topicos += info_topico + "\n"

# Guardamos el resumen en un txt
with open('resumen_topicos.txt', 'w', encoding='utf-8') as f:
    f.write(resumen_topicos)
print("\n[OK] Información de los tópicos guardada en 'resumen_topicos.txt'")


# ==========================================
# ASIGNACIÓN DEL TÓPICO DOMINANTE A CADA RESEÑA
# ==========================================
def format_topics_sentences(ldamodel, corpus):
    sent_topics = []
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        top_topic, prop_topic = row[0]
        sent_topics.append(int(top_topic))
    return sent_topics


df['id_topico'] = format_topics_sentences(lda_model, corpus)
df = df.drop(columns=['tokens'])

df.to_csv('train_con_lda.csv', index=False, encoding='utf-8-sig')
print("[ESTADO] Análisis finalizado. 'train_con_lda.csv' generado.")