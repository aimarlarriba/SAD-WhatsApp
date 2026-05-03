# Análisis Estratégico y NLP: WhatsApp vs Telegram

## Descripción del Proyecto
Este repositorio contiene un pipeline completo de procesamiento de Lenguaje Natural (NLP) y Machine Learning diseñado para auditar la salud competitiva de WhatsApp frente a Telegram. El objetivo principal es ir más allá de la simple clasificación de sentimientos, logrando extraer inteligencia de negocio procesable sobre los motivos reales de retención y fuga de usuarios.

El proyecto abarca desde la limpieza de datos en crudo hasta el modelado de tópicos no supervisado, culminando en una comparativa técnica entre algoritmos de clasificación tradicionales y modelos de Inteligencia Artificial Generativa. Los hallazgos se visualizan estratégicamente mediante dashboards en Tableau.

## Arquitectura del Pipeline y Estructura de Archivos

El repositorio está estructurado para reflejar el ciclo de vida del dato:

### 1. Preprocesamiento y Preparación de Datos
*   **`preparar_csv.py`**: Script fundacional encargado de la ingesta de las reseñas crudas. Realiza la limpieza de texto, tokenización, eliminación de ruido y exportación del conjunto de datos validado (`train_opiniones.csv` y otros) para alimentar las siguientes fases.

### 2. Aprendizaje No Supervisado (Topic Modeling)
*   **`clustering_lda.py`**: Implementación de Latent Dirichlet Allocation (LDA) para descubrir estructuras ocultas en los textos. Identifica los tópicos latentes (ej. "Lentitud", "Fallos de Cuenta", "Experiencia Multidispositivo").
*   **`grafico_lda.py`**: Módulo de visualización para proyectar los clusters generados, permitiendo interpretar semánticamente de qué se quejan exactamente los usuarios de cada plataforma.

### 3. Machine Learning Tradicional (Supervisado)
*   **`train.py`**: Pipeline de entrenamiento de modelos clásicos. Aplica técnicas de vectorización (TF-IDF, Bag of Words) y evalúa un abanico de algoritmos estadísticos: Logistic Regression, Random Forest, Decision Trees, KNN y Categorical/Multinomial Naive Bayes.
*   **`test.py`**: Script de validación para medir el rendimiento (Accuracy, Precision, Recall, F1-Score) de los modelos entrenados frente a datos no vistos.

### 4. Inteligencia Artificial Generativa y Aumentada
*   **`generativo_fewShot.py`**: Implementación de clasificación generativa utilizando el LLM Gemma 2 (vía Ollama y LangChain). Utiliza técnicas de *Few-Shot Prompting* sobre el texto crudo para clasificar sentimientos sin necesidad de reentrenamiento, actuando como un evaluador de alto nivel.
*   **`generativo_oversampling.py`**: Script experimental que utiliza IA generativa para la creación de datos sintéticos, buscando balancear clases minoritarias en el dataset sin recurrir a técnicas de duplicación clásica.
*   **`configuration.json`**: Archivo centralizado con los hiperparámetros y rutas necesarias para los modelos.

## Resultados Técnicos y de Negocio

**A nivel técnico:**
El proyecto demostró que la IA Generativa sin ajuste fino (*zero/few-shot*) supera las capacidades de clasificación estática del texto. Mientras que el mejor modelo del pipeline tradicional (Regresión Logística con TF-IDF) alcanzó un Accuracy de **0.66**, el LLM (Gemma 2:2b) alcanzó un Accuracy y Macro-Fscore de **0.74** interactuando únicamente con texto crudo.

**A nivel de negocio:**
Los resultados del clustering exportados a Tableau desvelaron un fallo estructural asimétrico: 
Telegram sufre un colapso en la fase de 'onboarding' (barrera de entrada por fallos de cuenta). Sin embargo, el problema crítico de WhatsApp es la **Lentitud Operativa** de la app una vez instalada, un lastre técnico que asfixia directamente su principal ventaja competitiva (la experiencia multidispositivo).

## Instrucciones de Ejecución

1. **Instalación del entorno:**
   Ejecutar en consola: `pip install -r requirements.txt`
   *(Nota: Se requiere tener Ollama instalado y el modelo gemma2:2b-text-q4_K_S descargado localmente para el módulo generativo).*

2. **Preparación de los datos:**
   Ejecutar: `python preparar_csv.py`

3. **Ejecución del pipeline clásico:**
   Ejecutar: `python train.py` seguido de `python test.py`

4. **Ejecución del pipeline generativo (Opcional/Avanzado):**
   Ejecutar: `python generativo_fewShot.py`

---

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## Declaración de Asistencia de IA
Se ha hecho uso de herramientas de IA Generativa (Gemini) como asistente para:

* **Estructuración técnica**: Diseño de la lógica de persistencia, partición de datos y gestión de carpetas por proyecto.
* **Depuración de código**: Resolución de problemas en el preprocesado dinámico (One-Hot Encoding con `get_dummies`) y flujos de variables.
* **Documentación**: Redacción y formato Markdown de este archivo README.

***Nota:** Todo el código ha sido validado y testeado manualmente para asegurar su integridad y cumplimiento con los objetivos de la asignatura.*
