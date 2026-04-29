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
        'get', 'even', 'one', 'would', 'really', 'phone',
        'please', 'dont', 'cant', 'something', 'thing', 'know',
        'make', 'use'
    })

    tokens = []
    for t in texto.split():
        if t not in stop_words and len(t) > 2:
            lema = lemmatizer.lemmatize(t, pos='v')
            lema = lemmatizer.lemmatize(lema, pos='n')
            tokens.append(lema)

    return tokens


print("[*] Limpiando y lematizando texto para análisis de tópicos...")
df['tokens'] = df['content'].apply(limpieza_temas)

# ==========================================
# TRANSFORMACIÓN DE DATOS (Fecha y Continente)
# Lo hacemos antes para tenerlo listo
# ==========================================
print("[*] Procesando fechas y ubicaciones...")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['mes_año'] = df['date'].dt.to_period('M').astype(str)


def obtener_continente(location_str):
    if pd.isnull(location_str): return 'Unknown'
    pais = str(location_str).split(',')[-1].strip().lower()

    if pais in ['united states', 'canada', 'mexico']:
        return 'North America'
    elif pais in ['brazil', 'argentina', 'chile', 'colombia', 'peru', 'ecuador', 'venezuela', 'bolivia', 'paraguay',
                  'uruguay']:
        return 'South America'
    elif pais in ['czech republic', 'italy', 'germany', 'united kingdom', 'netherlands', 'poland', 'spain',
                  'switzerland', 'sweden', 'france', 'norway', 'austria', 'belgium', 'ireland', 'denmark', 'finland',
                  'greece', 'hungary', 'portugal']:
        return 'Europe'
    elif pais in ['israel', 'pakistan', 'vietnam', 'thailand', 'china', 'japan', 'india', 'nepal', 'south korea',
                  'taiwan', 'indonesia', 'malaysia', 'saudi arabia', 'united arab emirates', 'qatar', 'philippines',
                  'singapore', 'bangladesh', 'sri lanka']:
        return 'Asia'
    elif pais in ['australia', 'new zealand', 'fiji', 'papua new guinea']:
        return 'Oceania'
    elif pais in ['zambia', 'ghana', 'nigeria', 'south africa', 'egypt', 'morocco', 'libya', 'uganda', 'tanzania',
                  'kenya', 'senegal', 'namibia', 'algeria', 'ethiopia', 'zimbabwe']:
        return 'Africa'
    else:
        return 'Other'


df['continente'] = df['location'].apply(obtener_continente)

# ==========================================
# ENTRENAMIENTO SEPARADO POR SENTIMIENTO
# ==========================================
sentimientos_a_analizar = ['positivo', 'negativo']
df_final = pd.DataFrame()
resumen_topicos = "=== RESUMEN DE TÓPICOS (LDA) SEPARADO POR SENTIMIENTO ===\n\n"
analisis_extra = ""

for sent in sentimientos_a_analizar:
    print(f"\n" + "=" * 50)
    print(f"[*] PROCESANDO RESEÑAS: {sent.upper()}")
    print("=" * 50)

    # Filtramos el dataset
    df_subset = df[df['sentiment'] == sent].copy()

    if df_subset.empty:
        print(f"[!] No hay datos para {sent}.")
        continue

    # Diccionario y corpus aislando el sentimiento
    id2word = corpora.Dictionary(df_subset['tokens'])
    id2word.filter_extremes(no_below=5, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in df_subset['tokens']]

    # Dentro del bucle for sent in sentimientos_a_analizar:
    if sent == 'positivo':
        n_topics = 2
    else:
        n_topics = 7

    print(f"[*] Entrenando modelo LDA ({sent.upper()}) con {n_topics} tópicos...")

    lda_model = LdaModel(
        corpus=corpus,
        num_topics=n_topics,
        id2word=id2word,
        random_state=42,
        passes=15,
        alpha='auto',
        eta='auto'
    )

    resumen_topicos += f"\n--- TÓPICOS PARA: {sent.upper()} ---\n"
    for idx, topic in lda_model.print_topics(-1):
        info_topico = f"Tópico {sent.upper()}_{idx}: {topic}"
        print(info_topico)
        resumen_topicos += info_topico + "\n"

    # Asignación del tópico con la etiqueta de sentimiento incluida
    sent_topics = []
    for row in lda_model[corpus]:
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        top_topic = row[0][0]
        # Pegamos el texto para que en Tableau se lea "POSITIVO_0" o "NEGATIVO_2"
        sent_topics.append(f"{sent.upper()}_{top_topic}")

    df_subset['id_topico'] = sent_topics

    # Analíticas para este sentimiento
    analisis_extra += f"\n\n>>> ANÁLISIS DEMOGRÁFICO PARA: {sent.upper()} <<<\n"

    analisis_extra += f"\n1. Tópico más frecuente por CONTINENTE ({sent.upper()}):\n"
    analisis_extra += df_subset.groupby('continente')['id_topico'].apply(
        lambda x: x.mode()[0] if not x.empty else 'N/A').to_string() + "\n"

    analisis_extra += f"\n2. Porcentaje de Tópicos por APLICACIÓN ({sent.upper()}):\n"
    tabla_fuente_pct = pd.crosstab(df_subset['source'], df_subset['id_topico'], normalize='index') * 100
    # ¡CORRECCIÓN IMPLACABLE AQUÍ! Faltaba el .to_string() envolviéndolo todo
    analisis_extra += (tabla_fuente_pct.round(2).astype(str) + "%").to_string() + "\n"

    # Acumulamos los datos procesados
    df_final = pd.concat([df_final, df_subset])
# ==========================================
# GUARDADO DE DATOS
# ==========================================
print("\n[*] Generando archivos finales...")
resumen_topicos += analisis_extra

with open('resumen_topicos.txt', 'w', encoding='utf-8') as f:
    f.write(resumen_topicos)

df_final = df_final.drop(columns=['tokens'])
df_final.to_csv('train_con_lda.csv', index=False, encoding='utf-8-sig')

print(
    "\n[ESTADO] Análisis finalizado. 'train_con_lda.csv' y 'resumen_topicos.txt' generados.")