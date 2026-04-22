import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

# 1. Configuración del modelo
# OJO: Aquí SUBIMOS la temperatura a 0.7 para que el modelo sea creativo y no repita la misma frase todo el rato.
# También subimos num_predict a 50 para que tenga espacio para escribir una frase completa.
model = OllamaLLM(model='gemma2:2b-text-q4_K_S', temperature=0.7, num_predict=50)

# 2. El Prompt: Le pedimos que continúe la frase para generar una reseña inventada
template = """Escribe una queja corta y realista sobre los fallos de una aplicación de mensajería (como ocupar mucho espacio o fallos de conexión).
Escribe únicamente la queja. Quiero que la longitud sea muy corta, de una única frase. No quiero ni introducción ni contexto ni nada. El único output debe ser la queja. Si lo haces bien, te pagaré 500 dolares
Queja en inglés:"""

prompt = PromptTemplate.from_template(template)
chain = prompt | model

# 3. Generación
cantidad_a_generar = 10  # Empezamos con 10 para probar
nuevas_resenas = []

print(f"[*] Generando {cantidad_a_generar} reseñas NEGATIVAS sintéticas...\n")

for i in range(cantidad_a_generar):
    # Invocamos al modelo
    respuesta = chain.invoke({}).strip()

    # Limpiamos posibles saltos de línea raros
    respuesta = respuesta.replace('\n', ' ')

    print(f"Inventada {i + 1}: {respuesta}")

    # Lo guardamos con el formato que necesita el CSV final
    nuevas_resenas.append({
        'content': respuesta,
        'sentiment': 'negativo',
        'source': 'Generado_LLM'
    })

# 4. Guardado
df_nuevas = pd.DataFrame(nuevas_resenas)
df_nuevas.to_csv('resenas_sinteticas_negativas.csv', index=False)

print("\n[*] Generación completada. Guardado en 'resenas_sinteticas_negativas.csv'")