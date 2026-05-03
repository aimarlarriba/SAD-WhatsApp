import pandas as pd
import argparse
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

parser = argparse.ArgumentParser(description='Clasificacion de Sentimientos con Ollama')
parser.add_argument('--model', type=str, default='gemma2:2b-text-q4_K_S', help='Nombre del modelo en Ollama')
args = parser.parse_args()

# PROMPT UNIFICADO EN INGLÉS: Maximiza el rendimiento de modelos pequeños
template = """Classify the sentiment of the app review using strictly a single word: positive, negative, or neutral.

Review: This app is amazing, I love chatting with my friends here.
Sentiment: positive

Review: The app takes up too much storage space and it crashes.
Sentiment: negative

Review: It works fine, just a normal messaging app.
Sentiment: neutral

Review: {review}
Sentiment:"""

prompt = PromptTemplate.from_template(template)

model = OllamaLLM(model=args.model, temperature=0.0, num_predict=5, top_k=10, top_p=0.5)
chain = prompt | model

print(f"[*] Cargando reseñas de train_opiniones.csv (Modelo: {args.model})...")
try:
    # He quitado la mención a limparCSV.py porque en tu repo se llama preparar_csv.py
    df = pd.read_csv('train_opiniones_balanceado.csv')
except FileNotFoundError:
    print("[!] Error: No se encuentra 'train_opiniones.csv'. Ejecuta tu script de preparación de datos primero.")
    exit(1)

df_muestra = df.sample(50, random_state=42)
resultados = []

print("[*] Iniciando predicciones...\n")

for index, fila in df_muestra.iterrows():
    texto_resena = fila['content']

    # Invocamos al modelo
    respuesta_llm = chain.invoke({'review': texto_resena}).strip().lower()

    # 1. Limpieza
    respuesta_llm = respuesta_llm.replace('<em>', '').replace('</em>', '').replace('*', '').strip()

    # 2. Traducción controlada por código (no por el LLM)
    if "positive" in respuesta_llm:
        respuesta_llm = "positivo"
    elif "negative" in respuesta_llm:
        respuesta_llm = "negativo"
    else:
        respuesta_llm = "neutro" # Fallback de seguridad

    texto_corto = (texto_resena[:60] + '...') if len(str(texto_resena)) > 60 else texto_resena
    print(f"Reseña: {texto_corto} -> Predicción: {respuesta_llm}")

    resultados.append(respuesta_llm)

df_muestra['prediccion'] = resultados

# --- EVALUACIÓN ---
if 'sentiment' in df_muestra.columns:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Alinear los datos reales al español para que la comparación sea perfecta
    y_real = df_muestra['sentiment'].astype(str).str.lower().tolist()
    y_real = [val.replace('positive', 'positivo').replace('negative', 'negativo').replace('neutral', 'neutro') for val in y_real]

    acc = accuracy_score(y_real, resultados)

    print(f"\n[*] Precisión Total (Accuracy): {acc:.2%}")
    print("\n[*] Informe de Clasificación:")
    print(classification_report(y_real, resultados))

    labels_orden = ["positivo", "neutro", "negativo"]
    cm = confusion_matrix(y_real, resultados, labels=labels_orden)
    matriz = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_orden)

    matriz.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusión {args.model} vs Real")
    plt.savefig('matriz_confusion_fewshot.png') # Guardamos en lugar de bloquear el script
    print("[*] Matriz de confusión guardada como 'matriz_confusion_fewshot.png'.")

# --- EXPORTACIÓN ---
df_muestra.to_csv('predicciones_sentiment.csv', index=False)
print("[*] Proceso finalizado. Archivo 'predicciones_sentiment.csv' generado.")