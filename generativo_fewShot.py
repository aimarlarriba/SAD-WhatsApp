import pandas as pd
import argparse
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

parser = argparse.ArgumentParser(description='Clasificacion de Sentimientos con Ollama')
parser.add_argument('--model', type=str, default='gemma2:2b-text-q4_K_S', help='Nombre del modelo en Ollama')
args = parser.parse_args()

# El truco Few-Shot: le damos 3 ejemplos claros antes de pedirle la respuesta
template = """Clasifica el sentimiento de la reseña de la aplicación usando estrictamente una sola palabra: positivo, negativo o neutro.

Reseña: This app is amazing, I love chatting with my friends here.
Sentimiento: positivo

Reseña: The app takes up too much storage space and it crashes.
Sentimiento: negativo

Reseña: It works fine, just a normal messaging app.
Sentimiento: neutro

Reseña: {review}
Sentimiento:"""

prompt = PromptTemplate.from_template(template)

# Configuración estricta
model = OllamaLLM(model=args.model, temperature=0.0, num_predict=5, top_k=10, top_p=0.5)
chain = prompt | model

print("[*] Cargando reseñas de train_opiniones.csv...")
try:
    df = pd.read_csv('train_opiniones.csv')
except FileNotFoundError:
    print("[!] Error: No se encuentra 'train_opiniones.csv'. Ejecuta limparCSV.py primero.")
    exit(1)

# Cogemos 50 aleatorias para que haya de todo
df_muestra = df.sample(50, random_state=42)
resultados = []

print("[*] Iniciando predicciones...\n")

for index, fila in df_muestra.iterrows():
    texto_resena = fila['content']

    # Invocamos al modelo solo con la reseña
    respuesta_llm = chain.invoke({'review': texto_resena}).strip().lower()

    # 1. Limpiamos asteriscos o cursivas que a veces se inventa
    respuesta_llm = respuesta_llm.replace('<em>', '').replace('</em>', '').replace('*', '').strip()

    # 2. Corregimos el error del spanglish
    if respuesta_llm == "neutral":
        respuesta_llm = "neutro"

    # 3. Filtro de seguridad vital: si no es ninguna de las 3, forzamos neutro
    if respuesta_llm not in ["positivo", "negativo", "neutro"]:
        respuesta_llm = "neutro"

    texto_corto = (texto_resena[:60] + '...') if len(str(texto_resena)) > 60 else texto_resena
    print(f"Reseña: {texto_corto} -> Predicción: {respuesta_llm}")

    resultados.append(respuesta_llm)

print("\n[*] Prueba completada con éxito.")