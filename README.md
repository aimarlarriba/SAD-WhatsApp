
# SAD-Clasificacion-Automatizada
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Este repositorio contiene la implementación un pipeline completo de procesamiento de Lenguaje Natural (NLP) y Machine Learning diseñado para auditar la salud competitiva de WhatsApp frente a Telegram, todo ello para la asignatura de **Sistemas de Ayuda a la Decisión (SAD)** de la **Universidad del País Vasco (UPV/EHU)**. El objetivo principal es ir más allá de la simple clasificación de sentimientos, logrando extraer inteligencia de negocio procesable sobre los motivos reales de retención y fuga de usuarios.  
 
El proyecto abarca desde la limpieza de datos en crudo hasta el modelado de tópicos no supervisado, culminando en una comparativa técnica entre algoritmos de clasificación tradicionales y modelos de Inteligencia Artificial Generativa. Los hallazgos se visualizan estratégicamente mediante dashboards en Tableau.

## Desarrollado por:
* **Urko Horas**
* **Lou Gómez**
* **Aimar Larriba**
* **Eneko Rodríguez**

---

## Estructura del Proyecto
El repositorio está estructurado para reflejar el ciclo de vida de los datos:

### 1. Preprocesamiento y Preparación de Datos  
*   **`preparar_csv.py`**: Automatiza la unificación y normalización de datasets (fuentes reales y sintéticas) en un CSV maestro para el entrenamiento de modelos de IA. El código estandariza las notas de cada opinión en etiquetas (positivo, negativo y neutro), limpia inconsistencias en el texto y homogeniza estructuras dispares, garantizando un corpus de datos limpio y listo para tareas de NLP.
  
### 2. Aprendizaje No Supervisado (Topic Modeling)  
*   **`clustering_lda.py`**: Implementación de Latent Dirichlet Allocation (LDA) para descubrir estructuras ocultas en los textos. Identifica los tópicos latentes (ej. "Lentitud", "Fallos de Cuenta", "Experiencia Multidispositivo").  
*   **`grafico_lda.py`**: Módulo de visualización para proyectar los clusters generados, permitiendo interpretar semánticamente de qué se quejan exactamente los usuarios de cada plataforma.  
  
### 3. Machine Learning Tradicional (Supervisado)
* **`train.py`**: Script que realiza la carga de datos, preproceso dinámico, partición estratificada, barrido de parámetros (Grid Search) y selección del mejor modelo.
* **`test.py`**: Programa para cargar el modelo ganador y clasificar nuevas instancias, manteniendo la consistencia del preproceso.
* **`configuration.json`**: Fichero centralizado de configuración (estrategias de preproceso y rangos de hiperparámetros).

### 4. Inteligencia Artificial Generativa y Aumentada  
*   **`generativo_fewShot.py`**: Implementación de clasificación generativa utilizando el LLM Gemma 2 (vía Ollama y LangChain). Utiliza técnicas de *Few-Shot Prompting* sobre el texto crudo para clasificar sentimientos sin necesidad de reentrenamiento, actuando como un evaluador de alto nivel.  
*   **`generativo_oversampling.py`**: Script experimental que utiliza IA generativa para la creación de datos sintéticos, buscando balancear clases minoritarias en el dataset sin recurrir a técnicas de duplicación clásica.  

### 5. Estructura de las carpetas
* **`proyectos/{project_name}/`**: 
  * **`datos/`**: Copias de seguridad de los datasets utilizados y tests automáticos.
  * **`best_model/`**: Contiene `bestmodel.sav`, `preprocessing_objects.sav`, `configuracion_usada.json`, `test.csv` (si se ha indicado su creación) y el informe de `ultimos_resultados.csv`.
  * **`archivo_versiones/`**: Histórico de modelos previos archivados al encontrar una mejora en el F-score.
  * **`predicciones_generadas/`**: CSVs resultantes de las ejecuciones de `test.py`.
  
```
      .
      ├── train.py                # Script de entrenamiento y optimización
      ├── test.py                 # Script de inferencia y evaluación
      ├── configuration.json      # Configuración de experimentos
      └── proyectos/
          └── {project_name}/     # Carpeta creada automáticamente
              ├── datos/          # Copias de seguridad y tests estratificados
              │   ├── entrenamiento_dataset.csv
              │   └── test_automatico_Iris.csv
              ├── best_model/     # El modelo con mejor F1-score hasta la fecha
              │   ├── bestmodel.sav
              │   ├── preprocessing_objects.sav
              │   ├── ultimos_resultados.csv
              │   └── predicciones_generadas/
              │       └── pred_KNN_F1_0.98_dataset.csv
              └── archivo_versiones/ # Historial de modelos
                  └── v_F1_0.9200_2026-03-20_10-30/
                      ├── bestmodel.sav
                      └── preprocessing_objects.sav
```

---

## Estructura de `configuration.json`
El archivo de configuración, el cual se muestra a continuación, actúa como el motor del experimento, permitiendo modificar el comportamiento de los scripts sin necesidad de editar el código fuente. 
```json
{  
  "project_name": "Nombre",  
  "algorithm": "todos",  
  "average_strategy": "macro",  
  "preprocessing": {  
    "test_split": 0.2,  
    "target_variable": "Target",  
    "drop_features": [],  
    "missing_values": "none",  
    "impute_strategy": "median",  
    "scaling": "none",  
    "sampling": "none",  
    "min_samples": 4,  
    "text_processing": {  
      "enabled": true,  
      "columns": ["content"],  
      "processing_type": "stem",  
      "method": "bow",  
      "language": "english",  
      "ngram_range": [1, 2],  
      "stopwords_domain": [],  
      "negation_words": []  
    }  
  },  
  "hyperparameters": {  
    "knn": {  
      "k_min": 3,  
      "k_max": 11,  
      "p_min": 1,  
      "p_max": 2,  
      "weights": ["uniform", "distance"]  
    },  
    "trees": {  
      "max_depth": [10, 20, 30, null],  
      "min_samples_leaf": [1, 2, 5]  
    },  
    "random_forest": {  
        "n_estimators": [100, 200, 300],  
        "max_depth": [10, 20, null]  
    },  
    "naive_bayes": {  
      "n_bins": [5, 10, 15],  
      "alphas": [0.01, 0.1, 0.5, 1.0],  
      "min_categories": null  
  },  
    "logistic_regression": {  
      "C": [10],  
      "solver": ["lbfgs", "saga"]  
    }  
  }  
}
```
Este se divide en tres bloques principales:

#### 1. Control de Ejecución
* **`project_name`**: Determina el nombre del proyecto.
* **`algorithm`**: Permite aislar un experimento o ejecutar la comparativa completa entre algoritmos para seleccionar el mejor modelo global.
    * **Valores**: `"knn"`, `"tree"`, `"rf"`, `"nb"`, `"lr"` o `"todos"`.
* **`average_strategy`**: Determina el tipo de F-Score a usar.
    * **Valores**: `"micro"`, `"macro"`, `"weighted"`, `"binary"` o `"auto"`.

#### 2. Preprocesado (`preprocessing`)
Configura las transformaciones que aseguran la calidad de los datos antes del entrenamiento:

* **`test_split`**: Extrae un porcentaje de la muestra inicial para generar un set de evaluación.
    * **Valores**: Float entre 0 y 1.
* **`target_variable`**: Nombre del atributo a predecir. Debe coincidir exactamente con el nombre de la columna objetivo en el archivo `.csv`.
    * **Valores**: String.
* **`drop_features`**: Nombres de las columnas irrelevantes o identificadores únicos a eliminar para evitar el sobreajuste. Deben coincidir exactamente con el nombre de la columna en el archivo `.csv`.
    * **Valores**: Lista de String `[]`.
* **`missing_values`**: Activa o desactiva la gestión de datos faltantes en el dataset
    * **Valores**: `"impute"` o `"none"`.
* **`impute_strategy`**: Define el criterio estadístico para rellenar los valores nulos.
    * **Valores**: `"mean"`, `"median"` o `"most_frequent"`.
* **`scaling`**: Activa o desactiva el escalado $Z$-score, fundamental para algoritmos basados en distancia como KNN.
    * **Valores**: `"standard"` o `"none"`.
* **`sampling`**: Define el método de balanceo de clases en el conjunto de entrenamiento para evitar sesgos hacia la clase mayoritaria.
    * **Valores**: `"undersampling"`, `"smote"`, `"adasyn"` o `"none"`.
* **`min_samples`**: Define el número mínimo de apariciones que debe tener una clase. Útil para datasets sin balanceo.
  * **Valores**: Integer
* **`text_processing`**: Diccionario que gestiona la vectorización y limpieza del lenguaje natural.
	* **`enabled`**: Activa/desactiva el procesamiento de texto.
		* **Valores**: `true` o `false`.
    * **`columns`**: Lista de columnas que contienen el texto a procesar.
	    * **Valores**: Lista de String `[]`.
    * **`processing_type`**: Técnica de normalización.
	    * **Valores**: `"stem"` o `"lemmatize"`.
    * **`method`**: Técnica de vectorización.
	    * **Valores**: `"bow"` o `"tfidf"`.
    * **`ngram_range`**: Define el rango de n-gramas (unigramas, bigramas) para capturar contexto.
	   * **Valores**: Lista de Integer `[]`.
    * **`stopwords_domain`**: Lista de palabras personalizadas a ignorar (nombres de la app, verbos comunes sin carga semántica).
	     * **Valores**: Lista de String `[]`.
    * **`negation_words`**: Palabras que deben conservarse para no perder el sentido negativo de las frases (ej. "no", "not").
	     * **Valores**: Lista de String `[]`.

#### 3. Hiperparámetros (`hyperparameters`)
Define los rangos para el barrido automático (Grid Search) y la optimización de los modelos:

* **`knn`**:
    * **`k_min` / `k_max`**: Define el rango de vecinos $k$ para el barrido.
        * **Valores**: Integer.
    * **`p_min` / `p_max`**: Define la métrica de distancia de Minkowski ($p=1$: Manhattan, $p=2$: Euclídea).
        * **Valores**: Integer.
    * **`weights`**: Determina la influencia de los vecinos según su cercanía. Se puede indicar sólo un método o ambos.
        * **Valores**: Lista de String `["uniform", "distance"]`. 
* **`trees`**: 
    * **`max_depth`**: Controla la profundidad máxima del árbol para evitar el *overfitting*. Se deben indicar los valores a probar, no el rango.
        * **Valores**: Lista de Integer `[]`. 
    * **`min_samples_leaf`**: Define el número mínimo de muestras requerido en un nodo terminal. Se deben indicar los valores a probar, no el rango.
        * **Valores**: Lista de Integer `[]`.
* **`random_forest`**:
    * **`n_estimators`**: Define el número de estimadores (árboles) que componen el Random Forest. Se deben indicar los valores a probar, no el rango.
        * **Valores**: Lista de Integer `[]`. 
    * **`max_depth`**: Controla la profundidad máxima del árbol para evitar el *overfitting*. Se deben indicar los valores a probar, no el rango.
        * **Valores**: Lista de Integer `[]`. 
* **`naive_bayes`**:
    * **`n_bins`**: Determina el número de intervalos para la discretización de variables continuas al usar la versión CategoricalNB. Se deben indicar los valores a probar.
        * **Valores**: Lista de Integer `[]`.
    * **`alphas`**: Define los valores del parámetro de suavizado de Laplace (Laplace smoothing) a probar en el barrido para evitar probabilidades nulas.
        * **Valores**: Lista de Float `[]`.
    * **`min_categories`**: Número mínimo de categorías esperadas por atributo. Si se desconoce, se deja vacío para que el algoritmo lo calcule automáticamente.
        * **Valores**: Integer o `null`.
* **`logistic_regression`**:
    * **`C`**: Define el parámetro de regularización (menor valor, mayor regularización).
        * **Valores**: Lista de Integer `[]`. 
    * **`solver`**:  Determina el algoritmo de optimización.
        * **Valores**: Lista de String `["lbfgs", "saga"]`. 

---

## Requisitos
El proyecto está desarrollado en Python 3.11. Para replicar el entorno de ejecución de forma sencilla, se recomienda el uso de un entorno virtual:
```bash
# Instalación de dependencias
pip install -r requirements.txt
```
***Nota:** _Se requiere tener `Ollama` instalado y el modelo `gemma2:2b-text-q4_K_S` descargado localmente para el módulo generativo_*.

---

## Modo de Empleo

### 1. Preparación de los datos
El script unifica archivos dispersos, limpia el texto y genera el archivo maestro necesario para el resto del pipeline. 
```bash
python preparar_csv.py
```

### 2. Entrenamiento y Barrido
El script de entrenamiento requiere dos argumentos por línea de comandos: el archivo de datos y el fichero de configuración.
```bash
python train.py TrainDev.csv configuration.json
```
El script comparará el F-score del mejor modelo actual en `best_model/`. Si el nuevo entrenamiento lo supera, se sustituirán los  datos por los del nuevo modelo.

***Nota:** El archivo `.csv` se puede indicar mediante su ruta directa o, si ya está `datos/` del proyecto correspondiente creado y se encuentra en esta, únicamente mediante su nombre*.

### 3. Clasificación de Instancias
Para predecir la clase de nuevas muestras, se utiliza el modelo guardado en el proyecto y carpeta correspondiente. Se debe indicar el nombre del proyecto, el nombre de la carpeta que contiene el modelo a probar y, si no ha sido generado de forma automática, el archivo de datos (cargado con datos nuevos) . 
```bash
python test.py NombreDelProyecto NombreCarpetaModelo (Test.csv)
```
***Nota:** El script de test aplica automáticamente el preprocesado (escalado, imputación) utilizando los parámetros aprendidos durante el entrenamiento, pero nunca aplica balanceo a los datos de test.*

***Nota:** El archivo `.csv` se puede indicar mediante su ruta directa o, si ha sido generado de forma automática y ya está en la carpeta del modelo a probar, únicamente mediante el nombre del proyecto y el modelo específico*.

### 4. Ejecución del Pipeline Generativo
Para ejecutar la clasificación basada en LLM (Gemma 2) mediante _Few-Shot Prompting_. 
```bash
python generativo_fewShot.py
```
***Nota:** Asegúrate de tener el servicio de `Ollama` corriendo en segundo plano.*.

### 5. Análisis de Tópicos
Para descubrir de qué hablan los usuarios y visualizar los resultados:
```bash
# Generar los clusters de tópicos 
python clustering_lda.py 

# Visualizar la distribución semántica 
python grafico_lda.p
```
---

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## Declaración de Asistencia de IA
Se ha hecho uso de herramientas de IA Generativa (Gemini) como asistente para:

* **Depuración de código LLM**: Resolución de problemas de formato y métricas en el pipeline de Inferencia Generativa (*Few-Shot Prompting* con LangChain y Ollama).  
* **Enfoque analítico**: Asistencia en la extracción de conclusiones estratégicas (benchmarking de competidores) para la narrativa visual de los dashboards de Tableau.  
* **Documentación**: Redacción, estructuración y formato Markdown del presente archivo README para reflejar el estado real de la arquitectura.

***Nota:** Todo el código ha sido validado y testeado manualmente para asegurar su integridad y cumplimiento con los objetivos de la asignatura.*