# LINK GOOGLE COLAB

https://colab.research.google.com/drive/1CaJzqQNhvV2bDtsbfSQZfoEimJK-VdXu#scrollTo=b33b0b5a

# Clasificación Automática de Modulaciones Digitales con Machine Learning

Este proyecto desarrolla un sistema de clasificación automática de modulaciones digitales a partir de señales IQ utilizando técnicas de Machine Learning. El objetivo principal es identificar el tipo de modulación de una señal recibida a partir de sus componentes en fase (`I`) y cuadratura (`Q`), considerando diferentes niveles de ruido representados mediante el SNR.

El trabajo utiliza el dataset público **RadioML 2016.10A**, ampliamente usado para tareas de clasificación automática de modulación. Se seleccionaron cinco tipos de modulación: `BPSK`, `QPSK`, `8PSK`, `QAM16` y `QAM64`.

---

## 1. Objetivo del proyecto

El objetivo del proyecto es comparar el desempeño de dos técnicas de aprendizaje automático para la clasificación de modulaciones digitales:

* Red Neuronal Multicapa: `MLPClassifier`.
* Regresión Logística con regularización L2.

Además, se busca evaluar si la extracción de características derivadas de las señales IQ mejora el desempeño del modelo frente al uso exclusivo de las muestras IQ crudas.

---

## 2. Dataset utilizado

El dataset utilizado es **RML2016.10A**, almacenado en formato `.pkl`.

Cada clave del dataset corresponde a una combinación de:

```text
(modulación, SNR)
```

Por ejemplo:

```text
('QPSK', 2)
```

Cada valor asociado a una clave tiene la forma:

```text
(1000, 2, 128)
```

Esto significa:

* `1000`: señales disponibles para esa combinación modulación-SNR.
* `2`: canales IQ, donde el primer canal corresponde a `I` y el segundo a `Q`.
* `128`: muestras temporales por señal.

En este proyecto se trabajó con:

```text
5 modulaciones × 20 niveles de SNR × 1000 señales = 100000 señales
```

Las modulaciones seleccionadas fueron:

```text
BPSK, QPSK, 8PSK, QAM16, QAM64
```

---

## 3. Estructura general del proyecto

El notebook sigue la siguiente secuencia:

1. Carga del dataset.
2. Exploratory Data Analysis (EDA).
3. Construcción del dataset final `X`, `y` y `SNR`.
4. Aplanamiento de señales IQ.
5. Codificación de etiquetas.
6. División train/test estratificada por modulación y SNR.
7. Feature Extraction.
8. Definición de pipelines.
9. Entrenamiento de modelos iniciales.
10. Optimización de hiperparámetros con repeated k-fold cross-validation.
11. Evaluación de modelos optimizados.
12. Análisis por SNR.
13. Comparación estadística entre modelos.
14. Curva de aprendizaje.
15. Visualización con t-SNE.
16. Conclusiones.

---

## 4. Análisis exploratorio de datos

El EDA permitió comprender la estructura del dataset antes del modelado. Se revisaron:

* Modulaciones disponibles.
* Niveles de SNR disponibles.
* Cantidad de muestras por combinación modulación-SNR.
* Forma de las señales IQ.
* Valores mínimos, máximos, media y desviación estándar de las señales.
* Presencia de valores nulos o infinitos.
* Señales IQ en el dominio del tiempo.
* Constelaciones IQ.
* Distribución de amplitud, fase, potencia y componentes I/Q.
* Comportamiento de características derivadas según modulación y SNR.

El análisis mostró que el dataset está balanceado, ya que cada combinación modulación-SNR contiene 1000 señales. También se evidenció que el nivel de SNR influye en la separabilidad de las modulaciones: a menor SNR, mayor ruido y mayor dificultad para clasificar correctamente.

---

## 5. Feature Extraction

A partir de las señales IQ se crearon características derivadas para enriquecer la representación de entrada de los modelos.

Inicialmente, cada señal tiene forma:

```text
(2, 128)
```

Luego se aplana para obtener:

```text
256 características = 128 valores I + 128 valores Q
```

Posteriormente, se generaron características derivadas:

### Características creadas

* Amplitud:

```text
amplitud = sqrt(I² + Q²)
```

* Potencia instantánea:

```text
potencia = I² + Q²
```

* Fase:

```text
fase = arctan2(Q, I)
```

* Fase circular:

```text
sin(fase)
cos(fase)
```

La fase se representa mediante seno y coseno porque es una variable circular. Los valores cercanos a `-π` y `π` representan direcciones similares, aunque numéricamente parezcan opuestos.

Además, se calcularon estadísticas resumen por señal:

* Media de I.
* Desviación estándar de I.
* Media de Q.
* Desviación estándar de Q.
* Media, desviación estándar, máximo y mínimo de amplitud.
* Media, desviación estándar, máximo y mínimo de potencia.
* Media y desviación estándar de `sin(fase)`.
* Media y desviación estándar de `cos(fase)`.

La matriz final con características queda con:

```text
784 características por señal
```

Distribuidas así:

```text
IQ crudo       = 256 columnas
amplitud       = 128 columnas
potencia       = 128 columnas
sin(fase)      = 128 columnas
cos(fase)      = 128 columnas
estadísticas   = 16 columnas
Total          = 784 columnas
```

---

## 6. División train/test

La división del dataset se realizó con una proporción 80/20:

```text
Entrenamiento: 80000 señales
Prueba:        20000 señales
```

La partición se hizo de forma estratificada combinando modulación y SNR. Para ello se creó una clave de estratificación:

```text
modulación_SNR
```

Por ejemplo:

```text
BPSK_-20
QPSK_18
QAM64_0
```

Esto permite que los conjuntos de entrenamiento y prueba mantengan una distribución equilibrada tanto de clases como de niveles de ruido.

El SNR no se incluyó como característica de entrada del modelo. Se conservó como variable auxiliar para estratificar los datos y para analizar posteriormente el rendimiento del modelo según el nivel de ruido.

---

## 7. Pipelines de Machine Learning

Se utilizaron pipelines de `scikit-learn` para integrar en un solo flujo:

1. Escalado de variables.
2. Selección de características.
3. Entrenamiento del modelo.

Cada pipeline incluye:

* `StandardScaler`: estandariza las características.
* `SelectKBest`: selecciona las características más relevantes.
* Clasificador: MLP o Regresión Logística L2.

La selección de características se realizó con:

```text
SelectKBest(score_func=f_classif)
```

`f_classif` aplica una prueba estadística tipo ANOVA F-test para evaluar qué características tienen mayor relación con la clase objetivo.

El uso de pipelines evita fuga de información, ya que el escalado y la selección de características se ajustan únicamente con los datos de entrenamiento en cada partición de validación cruzada.

---

## 8. Modelos implementados

### 8.1 Red Neuronal MLP

La MLP fue utilizada como modelo principal debido a su capacidad para aprender relaciones no lineales entre las características de entrada.

Configuración general:

```text
hidden_layer_sizes = (256, 128)
activation = relu
solver = adam
alpha = regularización L2
early_stopping = True
```

La regularización L2 mediante `alpha` ayuda a controlar el sobreajuste. Valores mayores de `alpha` implican mayor regularización.

### 8.2 Regresión Logística L2

La Regresión Logística L2 fue utilizada como modelo base lineal para comparar contra la MLP.

Configuración general:

```text
penalty = l2
solver = lbfgs
C = inverso de la regularización
```

En Regresión Logística, `C` funciona como el inverso de la regularización. Valores altos de `C` reducen la regularización, mientras que valores bajos aumentan la penalización.

---

## 9. Optimización de hiperparámetros

La optimización de hiperparámetros se realizó mediante:

```text
GridSearchCV + RepeatedStratifiedKFold
```

Se utilizó repeated k-fold cross-validation con:

```text
5 folds × 2 repeticiones = 10 evaluaciones
```

Esto permite evaluar cada combinación de hiperparámetros de forma más robusta que con una sola partición.

Se optimizaron principalmente:

### Para MLP

* Número de características seleccionadas `k`.
* Arquitectura de capas ocultas.
* Regularización L2 `alpha`.
* Tasa de aprendizaje inicial.

### Para Regresión Logística

* Número de características seleccionadas `k`.
* Parámetro `C`.

---

## 10. Resultados principales

Los resultados obtenidos muestran que la MLP supera a la Regresión Logística L2.

| Modelo                                        | Accuracy en prueba |
| --------------------------------------------- | -----------------: |
| MLP segura con IQ crudo                       |             0.3454 |
| MLP con características IQ derivadas          |             0.4316 |
| Regresión Logística L2 con características IQ |             0.3436 |

La MLP con características derivadas obtuvo el mejor desempeño general en el conjunto de prueba.

También se realizó una comparación mediante validación cruzada:

| Modelo                            | Accuracy promedio CV |
| --------------------------------- | -------------------: |
| MLP optimizada                    |               0.4040 |
| Regresión Logística L2 optimizada |               0.3442 |

La prueba de Wilcoxon mostró una diferencia estadísticamente significativa entre ambos modelos:

```text
p-value = 0.001953125
```

Esto indica que la MLP presenta un desempeño significativamente superior al modelo lineal evaluado.

---

## 11. Análisis por SNR

El análisis por SNR mostró que el desempeño del modelo depende directamente del nivel de ruido.

En niveles bajos de SNR, el accuracy disminuye porque la señal está más degradada por el ruido. En niveles medios y altos de SNR, el modelo alcanza mejores resultados, lo que confirma que el ruido afecta la separabilidad de las modulaciones.

Este análisis es importante porque permite evaluar la robustez del modelo frente a diferentes condiciones de canal.

---

## 12. Curva de aprendizaje

Se generó una curva de aprendizaje para analizar el comportamiento del modelo frente al tamaño del conjunto de entrenamiento.

La curva mostró una diferencia entre el error de entrenamiento y el error de validación, lo que evidencia cierto grado de sobreajuste. Sin embargo, el uso de regularización L2, selección de características y parada temprana ayudó a controlar parcialmente este problema.

---

## 13. Visualización con t-SNE

Se utilizó t-SNE para proyectar el espacio de características en dos dimensiones.

La visualización mostró superposición entre clases, especialmente en modulaciones con patrones similares o en presencia de ruido. Esto explica por qué el problema no es trivial y por qué un modelo lineal como la Regresión Logística tiene menor desempeño que la MLP.

---

## 14. Conclusiones

La extracción de características derivadas a partir de señales IQ permitió mejorar el desempeño del modelo respecto al uso exclusivo de IQ crudo. Al combinar las muestras originales con amplitud, potencia, fase circular y estadísticas resumen, se obtuvo una representación más informativa de la señal.

La MLP optimizada superó a la Regresión Logística L2 tanto en el conjunto de prueba como en validación cruzada. La prueba estadística de Wilcoxon confirmó que la diferencia entre ambos modelos es significativa.

El análisis por SNR evidenció que el ruido afecta directamente el desempeño del clasificador. En SNR bajos, las modulaciones son más difíciles de separar, mientras que en SNR medios y altos el modelo obtiene mejores resultados.

Como trabajo futuro, se recomienda evaluar modelos especializados en señales temporales, como CNN 1D, LSTM o arquitecturas híbridas, ya que podrían aprovechar mejor la estructura secuencial de las componentes I/Q.

---

## 15. Requisitos

El proyecto fue desarrollado en Python usando principalmente:

```text
numpy
pandas
matplotlib
scikit-learn
scipy
```

Instalación sugerida:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## 16. Cómo ejecutar el proyecto

1. Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
```

2. Entrar a la carpeta del proyecto:

```bash
cd ML-Digital-Modulations-classifier
```

3. Crear una carpeta `data` y colocar dentro el archivo:

```text
RML2016.10a_dict.pkl
```

4. Abrir el notebook:

```text
notebook/digital-modulations-classifier.ipynb
```

5. Ejecutar las celdas en orden.

---

## 17. Estructura sugerida del repositorio

```text
ML-Digital-Modulations-classifier/
│
├── data/
│   └── RML2016.10a_dict.pkl
│
├── notebook/
│   └── digital-modulations-classifier.ipynb
│
├── README.md
│
└── requirements.txt
```

Nota: el archivo del dataset puede no incluirse directamente en GitHub debido a su tamaño. En ese caso, debe indicarse en el repositorio cómo obtenerlo y dónde ubicarlo.

---

## 18. Autor

Proyecto desarrollado como parte de una actividad académica de Machine Learning aplicada a señales IQ y clasificación automática de modulaciones digitales.

Autor: `[Gustavo Berrú C]`
