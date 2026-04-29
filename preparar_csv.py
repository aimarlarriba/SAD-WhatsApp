import csv
import os


def obtener_sentimiento(score_raw):
    """Convierte el score (1-5) a la etiqueta requerida por el proyecto."""
    try:
        nota = int(str(score_raw).strip())
        if nota <= 2:
            return "negativo"
        elif nota == 3:
            return "neutro"
        else:
            return "positivo"
    except (ValueError, TypeError):
        return None


def procesar_ficheros(configuracion):
    total_datos = []
    cabecera = ["reviewID", "content", "sentiment", "gender", "location", "date", "source"]

    for ruta, etiqueta, formato in configuracion:
        if not os.path.exists(ruta):
            print(f"[!] No encontrado: {ruta}")
            continue

        print(f"[*] Procesando {etiqueta} (Formato: {formato})...")
        procesadas = 0

        with open(ruta, 'r', encoding='utf-8-sig', errors='replace') as f:
            lector = csv.reader(f, delimiter=',', quotechar='"')
            # Intentamos saltar cabecera (ajusta si los archivos generados no tienen)
            next(lector, None)

            for partes in lector:
                # Limpieza de columnas vacías al final
                while partes and partes[-1] == "":
                    partes.pop()

                if not partes:
                    continue

                if formato == "real":
                    # --- Lógica para Estructura Real (ID, contenido, score, etc.) ---
                    if len(partes) < 6:
                        continue

                    fecha = partes[-1]
                    localizacion = partes[-2]
                    genero = partes[-3]
                    score_val = partes[-4]
                    sentimiento = obtener_sentimiento(score_val)

                    if sentimiento:
                        # Reconstrucción del mensaje si hay comas mal escapadas
                        contenido = " ".join(partes[1:-4])
                        total_datos.append([
                            partes[0], contenido, sentimiento,
                            genero, localizacion, fecha, etiqueta
                        ])
                        procesadas += 1

                elif formato == "generado":
                    # --- Lógica para Estructura LLM (contenido, sentimiento, fuente) ---
                    # Estructura: [Contenido, Sentimiento, Fuente]
                    if len(partes) >= 3:
                        contenido = partes[0]
                        sentimiento = partes[1].strip().lower()
                        fuente = partes[2]

                        # Rellenamos con "N/A" los campos que no existen en este archivo
                        total_datos.append([
                            "N/A", contenido, sentimiento,
                            "N/A", "N/A", "N/A", fuente
                        ])
                        procesadas += 1

        print(f"    -> {procesadas} filas recuperadas.")

    # Guardar el resultado unificado
    with open('train_opiniones_balanceado.csv', 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow(cabecera)
        writer.writerows(total_datos)

    print(f"\n[OK] Archivo 'train_opiniones_balanceado.csv' generado con {len(total_datos)} filas.")


if __name__ == "__main__":
    # La configuración ahora incluye el tipo: 'real' o 'generado'
    archivos = [
        ('WhatsApp.csv', 'WhatsApp', 'real'),
        ('Telegram.csv', 'Telegram', 'real'),
        ('resenas_sinteticas_negativo.csv', 'LLM', 'generado'),
        ('resenas_sinteticas_neutro.csv', 'LLM', 'generado')
    ]
    procesar_ficheros(archivos)