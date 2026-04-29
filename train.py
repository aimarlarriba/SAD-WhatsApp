# ==========================================
# ENTRENAMIENTO
# ==========================================
import re
import pandas as pd  # pandas (pd) es la librería principal para leer CSVs y manejar datos en forma de tabla (DataFrames).
import json  # json nos permite leer el archivo configuration.json donde guardamos los parámetros sin tocar el código.
import sys
import pickle  # pickle es la herramienta para "empaquetar" y guardar nuestro modelo ya entrenado (.sav).
import os
import emoji
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import shutil
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler  # Herramienta para balancear los datos si hay muchas más filas de una clase que de otra (recorta la clase mayoritaria).
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Partir datos en dos trozos: uno para estudiar train y otro para test/dev.
from sklearn.impute import SimpleImputer  # Herramienta que busca celdas vacías (nulos o NaN) y las rellena (con "median" "mean" "most_frequent").
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score  # Las fórmulas matemáticas para ponerle nota a nuestro modelo.
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder  # Herramientas de preprocesado: escalar números (Z-score), hacer cajas (bins) y pasar texto a números (LabelEncoder).
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# ---------------------------------------------------------
# FUNCIÓN PARA CALCULAR LAS NOTAS DEL MODELO
# ---------------------------------------------------------
def registrar_metrica(y_true, y_pred, nom, pars, avg):
    # y_true: Las respuestas correctas reales.
    # y_pred: Las respuestas que ha adivinado el modelo.
    # nom: Nombre del algoritmo (ej. "KNN").
    # pars: Los hiperparámetros usados (ej. "k=3, p=1").
    # avg: Forma de hacer la media ('binary' si son 2 clases, 'macro' si son 3 o más).

    # Accuracy: % total de aciertos
    acc = accuracy_score(y_true, y_pred)
    # Precision: De todos los que el modelo dijo que eran clase X, ¿cuántos lo eran de verdad? (zero_division=0 evita errores si da 0).
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    # Recall: De todos los que de verdad eran clase X, ¿cuántos logró encontrar el modelo?
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    # F1 Score: Es una mezcla (media armónica) entre Precision y Recall. Es la nota en la que más nos fijamos.
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

    # Devuelve un diccionario con toda la fila de resultados para el CSV y, aparte, el número suelto del F1 para comparar.
    return {
        "Combinación": f"{nom} ({pars})",
        "Accuracy": acc, "Precisión": prec, "Recall": rec, f"F_score_{avg}": f1
    }, f1


# ---------------------------------------------------------
# FUNCION PARA PROCESAMIENTO DE TEXTO
# ---------------------------------------------------------
def limpiar_texto_libre(texto, idioma, negation_words=None, stopwords_domain=None):
    texto = str(texto)
    texto = emoji.demojize(texto, language='en').replace(':', ' ')
    texto = str(texto).lower()
    texto = re.sub(r'[^a-z0-9áéíóúñ\s]', '', texto)

    try:
        stop_words = set(stopwords.words(idioma))
    except:
        stop_words = set()

    # Si nos pasan la lista de negaciones desde el JSON, las salvamos
    if negation_words:
        stop_words = stop_words - set(negation_words)

    # Si nos pasan stopwords extra desde el JSON, las añadimos
    if stopwords_domain:
        stop_words = stop_words.union(set(stopwords_domain))

    stemmer = PorterStemmer()
    tokens = word_tokenize(texto)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]

    return " ".join(tokens)


# ---------------------------------------------------------
# FUNCIONES DE ENTRENAMIENTO ESPECÍFICAS POR ALGORITMO
# ---------------------------------------------------------
def entrenar_knn(hp, X_train, y_train, X_dev, y_dev, avg):
    # hp: contiene los hiperparámetros que configuramos en el JSON.
    resultados = []  # Aquí iremos apuntando las notas de cada intento.

    # Estas tres variables forman el "podio". Empiezan vacías o en -1.
    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_comb_local = ""

    # Probamos distintos números de vecinos (K). El range salta de 2 en 2 para probar solo impares (1, 3, 5...).
    for k in range(hp["knn"]["k_min"], hp["knn"]["k_max"] + 1, 2):

        # Probamos la forma de medir la distancia (p=1 es distancia Manhattan, p=2 es Euclídea).
        for p in range(hp["knn"].get("p_min", 1), hp["knn"].get("p_max", 2) + 1):

            # Probamos si todos los vecinos valen igual (uniform) o si el más cercano tiene más peso (distance).
            for w in hp["knn"]["weights"]:

                # 1. ENTRENAMIENTO: Creamos el modelo con los parámetros actuales y le damos los datos para que aprenda (.fit)
                print(f"[*] Entrenando KNN con k={k}, p={p}, weights={w}...")
                clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w).fit(X_train, y_train)

                # 2. EXAMEN: Le pedimos que adivine los datos de validación (.predict) y calculamos su nota (res, val).
                res, val = registrar_metrica(y_dev, clf.predict(X_dev), "KNN", f"k={k},p={p},w={w}", avg)

                # Guardamos  en la lista general.
                resultados.append(res)

                # 3.Si esta nota (val) supera a la mejor que teníamos guardada, actualizamos.
                if val > mejor_f1_local:
                    mejor_f1_local = val
                    mejor_clf_local = clf
                    mejor_comb_local = res["Combinación"]

    # Devolvemos todas las notas y al mejor
    # (El 'None' está ahí porque KNN no usa discretizador, al contrario que Naive Bayes).
    return resultados, mejor_f1_local, mejor_clf_local, None, mejor_comb_local


def entrenar_arboles(hp, X_train, y_train, X_dev, y_dev, avg):
    resultados = []
    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_comb_local = ""

    # Probamos hasta qué nivel de profundidad dejamos crecer al árbol (para que no memorice demasiado).
    for d in hp["trees"]["max_depth"]:

        # Probamos cuántos datos como mínimo tienen que quedar en la última rama para dar por válida una regla.
        for ml in hp["trees"]["min_samples_leaf"]:

            # ENTRENAMIENTO: Creamos el árbol.
            # (El random_state=42 es una semilla para que, si hay empates o decisiones aleatorias, siempre elija lo mismo y sea repetible).
            print(f"[*] Entrenando Decision Tree con max_depth={d}, min_samples_leaf={ml}...")
            clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=ml, random_state=42).fit(X_train, y_train)

            # EXAMEN: Calculamos la nota del árbol con los datos de prueba.
            res, val = registrar_metrica(y_dev, clf.predict(X_dev), "Tree", f"d={d},ml={ml}", avg)
            resultados.append(res)

            # Si este árbol es el mejor hasta ahora, lo guardamos.
            if val > mejor_f1_local:
                mejor_f1_local = val
                mejor_clf_local = clf
                mejor_comb_local = res["Combinación"]

    return resultados, mejor_f1_local, mejor_clf_local, None, mejor_comb_local


def entrenar_rf(hp, X_train, y_train, X_dev, y_dev, avg):
    resultados = []

    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_comb_local = ""

    # Bucle 1: Elegimos cuántos árboles vamos a plantar (ej. 50, 100, 200).
    # Random Forest funciona poniendo a muchos árboles distintos a votar la respuesta final.
    for n in hp["random_forest"]["n_estimators"]:
        # Bucle 2: Elegimos hasta qué nivel de profundidad dejamos crecer a todos esos árboles.
        for d in hp["random_forest"]["max_depth"]:
            # 1. ENTRENAMIENTO: Creamos el bosque con 'n' árboles y profundidad 'd', y lo ponemos a estudiar (.fit).
            # (El random_state=42 vuelve a ser para que el factor "aleatorio" del bosque sea siempre igual si lo repites).
            print(f"[*] Entrenando Random Forest con n_estimators={n}, max_depth={d}...")
            clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42).fit(X_train, y_train)

            # 2. EXAMEN: Hacemos que el bosque intente adivinar los datos de prueba (.predict) y le calculamos la nota.
            res, val = registrar_metrica(y_dev, clf.predict(X_dev), "Random Forest", f"n={n},d={d}", avg)

            # Guardamos la nota en la lista general
            resultados.append(res)

            if val > mejor_f1_local:
                mejor_f1_local = val
                mejor_clf_local = clf
                mejor_comb_local = res["Combinación"]

    return resultados, mejor_f1_local, mejor_clf_local, None, mejor_comb_local


def entrenar_nb(hp, X_train_ns, y_train_ns, X_dev_imp, y_dev, avg, cat_indices):
    resultados = []
    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_prep_local = None
    mejor_comb_local = ""

    # Extraemos parámetros del JSON
    min_cat = hp["naive_bayes"].get("min_categories", None)
    alphas = hp["naive_bayes"].get("alphas", [0.1, 0.5, 1.0])

    # --- 1. MULTINOMIAL NAIVE BAYES (El más recomendado para TF-IDF) ---
    for a in alphas:
        print(f"[*] Entrenando MultinomialNB con alpha={a}...")
        clf_mult = MultinomialNB(alpha=a).fit(X_train_ns, y_train_ns)

        res, val = registrar_metrica(y_dev, clf_mult.predict(X_dev_imp), "MultinomialNB", f"alpha={a}", avg)
        resultados.append(res)

        if val > mejor_f1_local:
            mejor_f1_local = val
            mejor_clf_local = clf_mult
            mejor_prep_local = None
            mejor_comb_local = res["Combinación"]

    # --- 2. CATEGORICAL NAIVE BAYES ---
    for a in alphas:
        for bins in hp["naive_bayes"].get("n_bins", [5]):
            # Discretizamos los datos continuos en categorías
            print(f"[*] Entrenando CategoricalNB con bins={bins}, alpha={a}...")
            disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
            X_train_nb = disc.fit_transform(X_train_ns)

            clf_cat = CategoricalNB(alpha=a, min_categories=min_cat).fit(X_train_nb, y_train_ns)

            res, val = registrar_metrica(y_dev, clf_cat.predict(disc.transform(X_dev_imp)), "CategoricalNB",
                                         f"bins={bins},alpha={a}", avg)
            resultados.append(res)

            if val > mejor_f1_local:
                mejor_f1_local = val
                mejor_clf_local = clf_cat
                mejor_prep_local = disc
                mejor_comb_local = res["Combinación"]

    # --- 3. GAUSSIAN NAIVE BAYES ---
    print(f"[*] Entrenando GaussianNB...")
    clf_gau = GaussianNB().fit(X_train_ns, y_train_ns)

    res, val = registrar_metrica(y_dev, clf_gau.predict(X_dev_imp), "GaussianNB", "default", avg)
    resultados.append(res)

    if val > mejor_f1_local:
        mejor_f1_local = val
        mejor_clf_local = clf_gau
        mejor_prep_local = None
        mejor_comb_local = res["Combinación"]

    return resultados, mejor_f1_local, mejor_clf_local, mejor_prep_local, mejor_comb_local


def entrenar_lr(hp, X_train, y_train, X_dev, y_dev, avg):
    resultados = []
    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_comb_local = ""

    # Iteramos sobre los hiperparámetros definidos en el JSON
    for c in hp["logistic_regression"].get("C", [1.0]):
        for sol in hp["logistic_regression"].get("solver", ["lbfgs"]):
            try:
                # 1. ENTRENAMIENTO
                # Eliminamos 'penalty' para cumplir con las normas de sklearn 1.8
                print(f"[*] Entrenando Regresión Logística con C={c}, solver={sol}...")
                clf = LogisticRegression(C=c, solver=sol, max_iter=1000, random_state=42).fit(X_train, y_train)

                # 2. EXAMEN
                res, val = registrar_metrica(y_dev, clf.predict(X_dev), "Logistic Regression",
                                             f"C={c},sol={sol}", avg)
                resultados.append(res)

                # 3. ACTUALIZAR MEJOR LOCAL
                if val > mejor_f1_local:
                    mejor_f1_local = val
                    mejor_clf_local = clf
                    mejor_comb_local = res["Combinación"]

            except ValueError:
                # Si el solver no es compatible con el número de clases, lo salta sin romper el programa
                continue

    return resultados, mejor_f1_local, mejor_clf_local, None, mejor_comb_local


# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def train():
    # 1. VERIFICAR ARGUMENTOS DE TERMINAL
    # sys.argv es una lista para consola
    # sys.argv[0] es "train.py", sys.argv[1] es el "csv", sys.argv[2] es "config.json".
    if len(sys.argv) < 3:
        print("\n[!] Uso: python train.py <ruta_csv_o_nombre> <config.json>")
        sys.exit(1)  # Corta la ejecución si no le has dado los archivos

    f_data_input = sys.argv[1]
    f_conf = sys.argv[2]

    # 2. LEER JSON DE CONFIGURACIÓN
    with open(f_conf, 'r') as f:
        config = json.load(f)

    proyecto = config.get("project_name", "Proyecto_Generico")

    # --- AJUSTE DE RUTA DINÁMICA ---
    # Si el archivo no existe en la ruta actual, lo busca en la carpeta 'datos' del proyecto
    if not os.path.exists(f_data_input):
        posible_ruta = os.path.join("proyectos", proyecto, "datos", f_data_input)
        if os.path.exists(posible_ruta):
            f_data_input = posible_ruta
        else:
            print(f"[ERROR] No se encuentra el archivo: {f_data_input}")
            sys.exit(1)

    # Extraemos las variables para usarlas más fácil. Si no existe en el json, coge el valor por defecto (ej. "Proyecto_Generico")
    conf_pre = config['preprocessing']
    hp = config['hyperparameters']
    target = conf_pre['target_variable']  # Esta es la columna que queremos adivinar (ej. "Species" o "Clase").
    algoritmo_elegido = config.get('algorithm', 'todos')  # Permite correr solo knn, solo tree, o todos.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")  # Para crear carpetas con la fecha actual.

    # 3. CREAR CARPETAS
    base_path = os.path.join("proyectos", proyecto)
    data_path = os.path.join(base_path, "datos")
    best_path = os.path.join(base_path, "best_model")
    archive_path = os.path.join(base_path,
                                "archivo_versiones")  # Aquí guardamos modelos viejos si encontramos uno mejor.

    for folder in [data_path, best_path, archive_path]:
        os.makedirs(folder, exist_ok=True)  # exist_ok=True evita que de error si la carpeta ya existe.

    # 4. CARGAR DATOS
    df_full = pd.read_csv(f_data_input)  # Lee el excel/csv

    # LabelEncoder coge las clases en texto (ej. "Gato", "Perro") y las vuelve números (0, 1) que la máquina entiende.
    le = LabelEncoder()
    y_full = le.fit_transform(df_full[target].astype(str))  # target es la columna objetivo del JSON.

    # Leer preferencia del JSON
    pref_avg = config.get("average_strategy", "auto")

    # Determinar el promedio real
    num_clases = len(le.classes_)

    if pref_avg == "auto":
        avg = 'binary' if num_clases == 2 else 'macro'
    else:
        # Si el usuario pide macro/micro/weighted pero solo hay 2 clases,
        # sklearn lo permite, pero 'binary' suele ser el estándar.
        avg = pref_avg

    print(f"[*] Modo de evaluación: F1-Score ({avg})")

    # En el JSON podemos decir si queremos separar un trozo automático para Test. Si es 0, no separa nada.
    split_pct = conf_pre.get('test_split', 0)

    # 5. DIVISIÓN INICIAL (Opcional según JSON)
    if split_pct > 0:
        # stratify=y_full: asegura que si hay 80% gatos y 20% perros en el archivo original,
        # en el trozo de test haya también 80% gatos y 20% perros.
        df, df_test_auto = train_test_split(df_full, test_size=split_pct, random_state=42, stratify=y_full)

    else:
        df = df_full  # Si split_pct era 0, usamos todo el archivo para entrenar.

    # Guarda una copia de seguridad exacta de lo que vamos a usar para entrenar.
    nombre_csv = os.path.basename(f_data_input)
    f_data_final = os.path.join(data_path, f"entrenamiento_{nombre_csv}")
    df.to_csv(f_data_final, index=False)

    # 6. LIMPIEZA Y PREPROCESADO
    # Si en el json pusiste "drop_features": ["ID", "Nombre"], aquí borra esas columnas porque no sirven para predecir.
    df_clean = df.drop(columns=[c for c in conf_pre.get('drop_features', []) if c in df.columns])

    # --- FILTRAR CLASES CON POCOS REGISTROS ---
    # Para poder hacer split estratificado y usar SMOTE después, necesitamos un mínimo de muestras por clase.
    min_muestras = conf_pre.get('min_samples', 1)
    conteos_clases = df_clean[target].value_counts()
    clases_suficientes = conteos_clases[conteos_clases >= min_muestras].index
    df_clean = df_clean[df_clean[target].isin(clases_suficientes)].copy()

    # Reajustamos el encoder para que las etiquetas sean continuas (0, 1, 2...)
    le = LabelEncoder()
    y = le.fit_transform(df_clean[target].astype(str))

    # 7. DIVISIÓN TRAIN / DEV (Antes de procesar texto o dummies) ---
    X_raw = df_clean.drop(columns=[target])
    X_train_raw, X_dev_raw, y_train, y_dev = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

    # Actualizamos el promedio por si el número de clases cambió tras filtrar
    num_clases = len(le.classes_)
    if pref_avg == "auto":
        avg = 'binary' if num_clases == 2 else 'macro'

    # --- PROCESAMIENTO DE TEXTO (CORREGIDO) ---
    text_cfg = conf_pre.get('text_processing', {})
    text_columns = text_cfg.get('columns', [])
    vectorizador = None

    if text_cfg.get('enabled', False) and text_columns:
        print(f"[*] Procesando y limpiando columnas de texto: {text_columns}")
        print(f"[*] Método de procesado de texto utilizado: {text_cfg.get('method')}")

        # Obtener idioma del JSON (por defecto 'english')
        idioma = text_cfg.get('language', 'spanish')

        # Leer las listas del JSON (si no existen, devuelve lista vacía)
        p_neg = text_cfg.get('negation_words', [])
        s_dom = text_cfg.get('stopwords_domain', [])

        for col in text_columns:
            # PASO 1: Limpiar el texto (minúsculas, stopwords, stemmer)
            X_train_raw[col] = X_train_raw[col].apply(lambda x: limpiar_texto_libre(x, idioma, p_neg, s_dom))
            X_dev_raw[col] = X_dev_raw[col].apply(lambda x: limpiar_texto_libre(x, idioma, p_neg, s_dom))

        # PASO 2: Unir las columnas de texto limpias en una sola cadena para vectorizar
        train_text = X_train_raw[text_columns].apply(lambda x: ' '.join(x), axis=1)
        dev_text = X_dev_raw[text_columns].apply(lambda x: ' '.join(x), axis=1)

        # Vectorizar: FIT solo en TRAIN
        rango_ngramas = tuple(text_cfg.get('ngram_range', [1, 2]))
        if text_cfg.get('method') == "bow":
            vectorizador = CountVectorizer(ngram_range=rango_ngramas, max_df=0.90, min_df=3)
        else:
            vectorizador = TfidfVectorizer(ngram_range=rango_ngramas, max_df=0.90, min_df=3)

        X_train_vec = vectorizador.fit_transform(train_text)
        X_dev_vec = vectorizador.transform(dev_text)

        # Convertir a DataFrame
        X_train_text_df = pd.DataFrame(X_train_vec.toarray(), columns=vectorizador.get_feature_names_out(),
                                       index=X_train_raw.index)
        X_dev_text_df = pd.DataFrame(X_dev_vec.toarray(), columns=vectorizador.get_feature_names_out(),
                                     index=X_dev_raw.index)

        # Quitar columnas de texto originales y unir las vectorizadas
        X_train_raw = pd.concat([X_train_raw.drop(columns=text_columns), X_train_text_df], axis=1)
        X_dev_raw = pd.concat([X_dev_raw.drop(columns=text_columns), X_dev_text_df], axis=1)

    # Aplicar dummies en Train
    X_train = pd.get_dummies(X_train_raw, drop_first=True)

    # Aplicar dummies en Dev y sincronizar columnas con Train
    X_dev = pd.get_dummies(X_dev_raw, drop_first=True)
    X_dev = X_dev.reindex(columns=X_train.columns, fill_value=0)

    # Eliminar columnas constantes (basado solo en Train)
    cols_activas = [col for col in X_train.columns if X_train[col].nunique() > 1]
    X_train = X_train[cols_activas]
    X_dev = X_dev[cols_activas]

    # La 'y' es nuestra columna de soluciones pasada a números
    y = le.transform(df_clean[target].astype(str))

    # 8. IMPUTACIÓN (Rellenar huecos vacíos)
    imputer = None
    X_train_imp, X_dev_imp = X_train, X_dev  # Por defecto, los datos se quedan igual

    # Solo imputamos si el JSON lo pide explícitamente
    if conf_pre.get('missing_values') == "impute":
        imputer = SimpleImputer(strategy=conf_pre.get('impute_strategy', 'mean')).fit(X_train)
        X_train_imp = imputer.transform(X_train)
        X_dev_imp = imputer.transform(X_dev)

    # 9. ESCALADO (Z-Score)
    scaler = None  # Por defecto nada
    X_train_prep, X_dev_prep = X_train_imp, X_dev_imp  # Copiamos los datos tal cual

    # Si el json pide "standard", aplica la fórmula matemática para que todos los números estén en la misma escala (para knn)
    if conf_pre.get('scaling') == "standard":
        scaler = StandardScaler().fit(X_train_imp)
        X_train_prep = scaler.transform(X_train_imp)
        X_dev_prep = scaler.transform(X_dev_imp)

    # 10. BALANCEO DE CLASES
    # Si hay 90 sanos y 10 enfermos, undersampling borra 80 sanos al azar para que quede 10 y 10.
    estrategia_balanceo = conf_pre.get('sampling')

    # Opción 1: Undersampling (recorta la clase mayoritaria)
    if estrategia_balanceo == "undersampling":
        rus = RandomUnderSampler(random_state=42)
        X_train_model, y_train_model = rus.fit_resample(X_train_prep, y_train)
        X_train_ns, y_train_ns = rus.fit_resample(X_train_imp, y_train)

    # Opción 2: Oversampling/SMOTE (crea datos sintéticos para la clase minoritaria)
    elif estrategia_balanceo == "smote":
        smote = SMOTE(random_state=42)
        X_train_model, y_train_model = smote.fit_resample(X_train_prep, y_train)
        X_train_ns, y_train_ns = smote.fit_resample(X_train_imp, y_train)

    elif estrategia_balanceo == "adasyn":
        adasyn = ADASYN(random_state=42)
        X_train_model, y_train_model = adasyn.fit_resample(X_train_prep, y_train)
        X_train_ns, y_train_ns = adasyn.fit_resample(X_train_imp, y_train)

    # Opción 3: Ningún balanceo
    else:
        X_train_model, y_train_model = X_train_prep, y_train
        X_train_ns, y_train_ns = X_train_imp, y_train

    # 11.Llamada a algoritmos
    # Aquí vamos guardando al mejor de entre todos los algoritmos.
    resultados_globales = []
    mejor_f1_global = -1
    mejor_clf_global = None
    mejor_prep_global = None
    nombre_mejor_global = ""
    mejor_comb_global = ""

    # Si en el json pone "knn" o "todos", entra aquí. (Sigue la misma lógica para el resto).
    if algoritmo_elegido in ["knn", "todos"]:
        res, f1, clf, prep, comb = entrenar_knn(hp, X_train_model, y_train_model, X_dev_prep, y_dev, avg)
        resultados_globales.extend(res)
        if f1 > mejor_f1_global:  # Si este KNN superó la mejor nota global actual, se pone la corona.
            mejor_f1_global, mejor_clf_global, mejor_prep_global, nombre_mejor_global, mejor_comb_global = f1, clf, prep, "KNN", comb

    if algoritmo_elegido in ["tree", "todos"]:
        res, f1, clf, prep, comb = entrenar_arboles(hp, X_train_model, y_train_model, X_dev_prep, y_dev, avg)
        resultados_globales.extend(res)
        if f1 > mejor_f1_global:
            mejor_f1_global, mejor_clf_global, mejor_prep_global, nombre_mejor_global, mejor_comb_global = f1, clf, prep, "Tree", comb

    if algoritmo_elegido in ["rf", "todos"]:
        res, f1, clf, prep, comb = entrenar_rf(hp, X_train_model, y_train_model, X_dev_prep, y_dev, avg)
        resultados_globales.extend(res)
        if f1 > mejor_f1_global:
            mejor_f1_global, mejor_clf_global, mejor_prep_global, nombre_mejor_global, mejor_comb_global = f1, clf, prep, "Random Forest", comb

    if algoritmo_elegido in ["nb", "todos"]:
        indices_dummy = []

        # Llamamos directamente con los datos procesados (no escalados)
        # X_train_ns es la versión "No Scaled", ideal para Naive Bayes
        res, f1, clf, prep, comb = entrenar_nb(
            hp,
            X_train_ns,
            y_train_ns,
            X_dev_imp,
            y_dev,
            avg,
            indices_dummy
        )

        resultados_globales.extend(res)

        if f1 > mejor_f1_global:
            mejor_f1_global = f1
            mejor_clf_global = clf
            mejor_prep_global = prep
            nombre_mejor_global = "Naive Bayes"
            mejor_comb_global = comb

    if algoritmo_elegido in ["lr", "todos"]:
        res, f1, clf, prep, comb = entrenar_lr(hp, X_train_model, y_train_model, X_dev_prep, y_dev, avg)
        resultados_globales.extend(res)

        if f1 > mejor_f1_global:
            mejor_f1_global = f1
            mejor_clf_global = clf
            mejor_prep_global = prep
            nombre_mejor_global = "Logistic Regression"
            mejor_comb_global = comb

    # ==========================================
    # 12. GUARDADO Y ARCHIVADO (LÓGICA MEJOR/PEOR)
    # ==========================================
    ruta_obj_best = os.path.join(best_path, "preprocessing_objects.sav")
    ruta_model_best = os.path.join(best_path, "bestmodel.sav")
    ruta_csv_best = os.path.join(best_path, "ultimos_resultados.csv")

    # Definimos el nombre de la carpeta de este intento (se usará en ambos casos)
    nombre_intento = f"intento_F1_{mejor_f1_global:.4f}_{timestamp}"
    folder_historial = os.path.join(archive_path, nombre_intento)

    # Creamos el diccionario con todas las herramientas de esta sesión
    obj_final = {
        'target_variable': target,
        'imputer': imputer, 'scaler': scaler, 'label_encoder': le,
        'columns': X_train.columns, 'discretizer': mejor_prep_global,
        'algoritmo': nombre_mejor_global, 'f1_score': mejor_f1_global,
        'average_strategy': avg, 'combinacion_exacta': mejor_comb_global,
        'fecha': timestamp, 'project_name': proyecto,
        'vectorizador_texto': vectorizador, 'text_columns_original': text_columns,
        'language': idioma, 'drop_features': conf_pre.get('drop_features', []),
        'negation_words': text_cfg.get('negation_words', []),
        'stopwords_domain': text_cfg.get('stopwords_domain', [])
    }

    # 1. GUARDAR SIEMPRE EN EL HISTORIAL (Para tener registro de todos)
    os.makedirs(folder_historial, exist_ok=True)
    pickle.dump(mejor_clf_global, open(os.path.join(folder_historial, "model.sav"), 'wb'))
    pickle.dump(obj_final, open(os.path.join(folder_historial, "preprocessing.sav"), 'wb'))
    pd.DataFrame(resultados_globales).to_csv(os.path.join(folder_historial, "resultados.csv"), index=False)
    shutil.copy2(f_conf, os.path.join(folder_historial, "configuracion_usada.json"))

    # Guardar el test usado en este intento específico
    if split_pct > 0:
        df_test_auto.to_csv(os.path.join(folder_historial, "test.csv"), index=False)

    # 2. CONTROL DE RÉCORD (Actualizar best_model si es mejor)
    f1_actual = 0.0
    if os.path.exists(ruta_obj_best):
        with open(ruta_obj_best, 'rb') as f:
            f1_actual = pickle.load(f).get('f1_score', 0.0)

    if mejor_f1_global > f1_actual:
        # Si es un nuevo récord, movemos el "best" antiguo al historial antes de sobrescribir
        if os.path.exists(ruta_model_best):
            with open(ruta_obj_best, 'rb') as f:
                old_meta = pickle.load(f)
                old_f1 = old_meta.get('f1_score', 0.0)
                old_ts = old_meta.get('fecha', 'antiguo').replace(':', '-')

            # Nombre unificado para el récord superado
            backup_folder = os.path.join(archive_path, f"intento_F1_{old_f1:.4f}_{old_ts}")
            os.makedirs(backup_folder, exist_ok=True)
            shutil.move(ruta_model_best, os.path.join(backup_folder, "model.sav"))
            shutil.move(ruta_obj_best, os.path.join(backup_folder, "preprocessing.sav"))

        # Guardamos el nuevo récord en la carpeta principal
        pickle.dump(mejor_clf_global, open(ruta_model_best, 'wb'))
        pickle.dump(obj_final, open(ruta_obj_best, 'wb'))
        pd.DataFrame(resultados_globales).to_csv(ruta_csv_best, index=False)
        if split_pct > 0:
            df_test_auto.to_csv(os.path.join(best_path, "test.csv"), index=False)

        print(f"\n[!] NUEVO RÉCORD: {mejor_f1_global:.4f}. Actualizado en 'best_model'.")
    else:
        print(f"\n[-] No supera al mejor ({f1_actual:.4f}). Guardado en historial: {nombre_intento}")


# Esto asegura que la función train() solo arranque si ejecutas el archivo directamente.
if __name__ == "__main__":
    train()
