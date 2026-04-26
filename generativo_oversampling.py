import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import sys

def main():
    tipo_de_resena = sys.argv[1].lower()
    cantidad_a_generar = sys.argv[2]

    # 1. Configuración del modelo
    # Temperatura a 0.7 para que sea creativo y num_predict a 40 para frases cortas.
    model = OllamaLLM(model='gemma2:2b-text-q4_K_S', temperature=0.7, num_predict=40)

    # 2. El Prompt: Truco Few-Shot. Le damos una lista y le forzamos a escribir el elemento 4.
    template = f"""Here is a list of very short and {tipo_de_resena} user reviews about a messaging app:
    1. "The app crashes constantly and takes up way too much storage space."
    2. "I receive too much spam and the call quality is absolutely terrible."
    3. "Updates take forever and the new interface is very confusing."
    4. \""""

    prompt = PromptTemplate.from_template(template)

    # Subimos un pelín el num_predict para que acabe la frase, la cortaremos luego por código
    model = OllamaLLM(model='gemma2:2b-text-q4_K_S', temperature=0.7, num_predict=50)
    chain = prompt | model

    # 3. Generación
    nuevas_resenas = []
    print(f"[*] Generando {cantidad_a_generar} reseñas {tipo_de_resena}s sintéticas...\n")

    for i in range(int(cantidad_a_generar)):
        respuesta = chain.invoke({}).strip()

        # TRUCO DE LIMPIEZA: Como el modelo empezará escribiendo texto, le decimos
        # que corte la frase exactamente donde ponga las siguientes comillas (") o un salto de línea.
        # Así nos quitamos toda la basura que intente escribir después.
        respuesta_limpia = respuesta.split('"')[0].split('\n')[0].strip()

        print(f"Inventada {i + 1}: {respuesta_limpia}")

        nuevas_resenas.append({
            'content': respuesta_limpia,
            'sentiment': tipo_de_resena,
            'source': 'Generado_LLM'
        })

    # 4. Guardado
    df_nuevas = pd.DataFrame(nuevas_resenas)
    df_nuevas.to_csv(f'resenas_sinteticas_{tipo_de_resena}.csv', index=False)

    print(f"\n[*] Generación completada. Guardado en 'resenas_sinteticas_{tipo_de_resena}.csv'")

if __name__=='__main__':
    if len(sys.argv) < 3:
        print('[!] Uso: python generativo_oversampling.py [positiva/negativa/neutra] [nº de reseñas]')
        sys.exit(1)
    main()