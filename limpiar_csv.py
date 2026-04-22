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

    for ruta, etiqueta in configuracion:
        if not os.path.exists(ruta):
            print(f"[!] No encontrado: {ruta}")
            continue

        print(f"[*] Procesando {etiqueta}...")
        procesadas = 0

        with open(ruta, 'r', encoding='utf-8-sig', errors='replace') as f:
            lector = csv.reader(f, delimiter=',', quotechar='"')
            next(lector, None)  # Saltar cabecera

            for partes in lector:
                # 1. LIMPIEZA DE COLUMNAS VACÍAS AL FINAL
                # Si la fila termina en coma, partes[-1] será ''. Lo eliminamos.
                while partes and partes[-1] == "":
                    partes.pop()

                # Ahora la estructura es siempre: [ID, ...Mensaje..., Score, Gender, Location, Date]
                if len(partes) < 6:
                    continue

                # 2. ANCLAJE DESDE EL FINAL (Ahora sí es consistente)
                fecha = partes[-1]
                localizacion = partes[-2]
                genero = partes[-3]
                score_val = partes[-4]

                sentimiento = obtener_sentimiento(score_val)

                if sentimiento:
                    # 3. RECONSTRUCCIÓN: Todo lo que sobre entre el ID (0) y el Score (-4) es el mensaje
                    # Juntamos con espacio para no pegar palabras si faltan espacios tras comas
                    contenido = " ".join(partes[1:-4])

                    total_datos.append([
                        partes[0], contenido, sentimiento,
                        genero, localizacion, fecha, etiqueta
                    ])
                    procesadas += 1

        print(f"    -> {procesadas} filas recuperadas.")

    # Guardar para el siguiente paso (ML y Tableau)
    with open('train_opiniones.csv', 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        writer.writerow(cabecera)
        writer.writerows(total_datos)

    print(f"\n[OK] Archivo 'train_opiniones.csv' generado con {len(total_datos)} filas.")


if __name__ == "__main__":
    archivos = [('WhatsApp.csv', 'WhatsApp'), ('Telegram.csv', 'Telegram')]
    procesar_ficheros(archivos)