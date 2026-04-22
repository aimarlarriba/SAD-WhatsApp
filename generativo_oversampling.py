import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import sys


def main():

    tipo_de_resena=sys.argv[1].lower()
    cantidad_a_generar=sys.argv[2]

    # 1. Configuración del modelo
    # OJO: Aquí SUBIMOS la temperatura a 0.7 para que el modelo sea creativo y no repita la misma frase todo el rato.
    # También subimos num_predict a 50 para que tenga espacio para escribir una frase completa.
    model = OllamaLLM(model='gemma2:2b-text-q4_K_S', temperature=0.7, num_predict=40)

    # 2. El Prompt: Le pedimos que continúe la frase para generar una reseña inventada
    template = f"""Escribe una única reseña {tipo_de_resena} corta y realista sobre una aplicación de mensajería (WhatsApp/Telegram). 
    Asegurate de que la reseña sea {tipo_de_resena} Ciñete al tipo de reseña que te he pedido. No hagas ninguna de otro tipo.
    La reseña debe ser en inglés. No escribas la reseña en español bajo ningún concepto. 
    Quiero que la longitud sea muy corta, de una única frase. 
    No quiero ni introducción ni contexto ni comentarios sobre la reseña. Única y exclusivamente la reseña. Nada mas. El único output debe ser la reseña. 
    Si lo haces bien, te pagaré 5000 dolares.
    reseña en inglés:"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | model

    # 3. Generación
    nuevas_resenas = []

    print(f"[*] Generando {cantidad_a_generar} reseñas {tipo_de_resena} sintéticas...\n")

    for i in range(int(cantidad_a_generar)):
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
    df_nuevas.to_csv(f'resenas_sinteticas_{tipo_de_resena}.csv', index=False)

    print(f"\n[*] Generación completada. Guardado en 'resenas_sinteticas_{tipo_de_resena}.csv'")

if __name__=='__main__':
    if len(sys.argv) < 3:
        print('[!] Uso: python generativo_oversampling.py ["positiva/negativa/neutra"] ["nº de reseñas a generar"]')
        sys.exit(1)
    main()