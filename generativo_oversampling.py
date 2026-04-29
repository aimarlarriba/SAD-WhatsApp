import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import sys

def main():
    tipo_de_resena = sys.argv[1].lower()
    cantidad_a_generar = sys.argv[2]

    # 1. Ajustamos los ejemplos (Few-Shot) dinámicamente según lo que pida el usuario
    if tipo_de_resena == "positiva":
        ejemplos = """1. "This app is fantastic, I use it every single day to talk to my family."
2. "Great privacy features and the calls are super clear."
3. "I love the new update, the interface is so clean and fast."
4. \""""
        etiqueta = "positivo"
    elif tipo_de_resena == "neutra":
        ejemplos = """1. "It's just a normal messaging app, nothing special but it works."
2. "Standard features. It does the job."
3. "An average app for texting, neither good nor bad."
4. \""""
        etiqueta = "neutro"
    else:  # Si no es positiva ni neutra, asumimos que es negativa
        ejemplos = """1. "The app crashes constantly and takes up way too much storage space."
2. "I receive too much spam and the call quality is absolutely terrible."
3. "Updates take forever and the new interface is very confusing."
4. \""""
        etiqueta = "negativo"

    # 2. Construimos el Prompt final uniendo el encabezado y los ejemplos elegidos
    template = f"""Here is a list of very short and {tipo_de_resena} user reviews about a messaging app:
{ejemplos}"""

    prompt = PromptTemplate.from_template(template)

    # 3. Configuración del modelo
    model = OllamaLLM(model='gemma2:2b-text-q4_K_S', temperature=0.7, num_predict=50)
    chain = prompt | model

    # 4. Generación
    nuevas_resenas = []
    print(f"[*] Generando {cantidad_a_generar} reseñas {tipo_de_resena}s sintéticas...\n")

    for i in range(int(cantidad_a_generar)):
        respuesta = chain.invoke({}).strip()

        # Limpieza con el machetazo
        respuesta_limpia = respuesta.split('"')[0].split('\n')[0].strip()

        print(f"Inventada {i + 1}: {respuesta_limpia}")

        nuevas_resenas.append({
            'content': respuesta_limpia,
            'sentiment': etiqueta,  # Usamos la etiqueta exacta que necesita Aimar
            'source': 'Generado_LLM'
        })

    # 5. Guardado
    df_nuevas = pd.DataFrame(nuevas_resenas)
    df_nuevas.to_csv(f'resenas_sinteticas_{etiqueta}.csv', index=False)

    print(f"\n[*] Generación completada. Guardado en 'resenas_sinteticas_{etiqueta}.csv'")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('[!] Uso: python generativo_oversampling.py [positiva/negativa/neutra] [nº de reseñas]')
        sys.exit(1)
    main()