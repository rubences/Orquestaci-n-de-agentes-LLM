# Orquestacion de agentes LLM con Python

Este repositorio contiene una implementacion en Python para orquestar un flujo RAG
con LangChain + FAISS + IBM watsonx Granite.

Referencia del tutorial base:
<https://www.ibm.com/es-es/think/tutorials/llm-agent-orchestration-with-langchain-and-granite#228874318>

Referencia adicional (LangGraph y futuro de agentes IA):
<https://medium.com/@jddam/gu%C3%ADa-completa-langgraph-y-el-futuro-de-los-agentes-de-ia-en-2025-2f34ceaa456f>

## Archivos principales

- `llm_agent_orchestration.ipynb`: Notebook paso a paso.
- `llm_agent_orchestration.py`: Script Python ejecutable.

## Requisitos

Python 3.10+ recomendado.

Instala dependencias:

```bash
pip install --upgrade pip
pip install langchain langchain-ibm langchain-community faiss-cpu pandas sentence-transformers
```

## Variables y credenciales

- `WML_APIKEY`: API key de IBM watsonx.
- `PROJECT_ID`: ID del proyecto watsonx.

Si no defines variables de entorno, el script pedira los datos en consola.
Tambien puedes pasar la API key por argumento con `--apikey`.

El `project_id` por defecto en este repo es:

`d7e326f5-1a40-4612-9a0b-5888b4f4034a`

## Ejecucion

1. Descarga el archivo de texto de ejemplo `aosh.txt` (The Adventures of Sherlock Holmes)
  desde Project Gutenberg y colocalo en la raiz del repo.
2. Ejecuta:

```bash
python llm_agent_orchestration.py --input-file aosh.txt --output-csv output.csv
```

## Ejecucion usando CSV (texto fuente + consultas)

Formato recomendado de CSV de texto (`documentos.csv`):

```csv
text
"Sherlock Holmes es un detective consultor..."
"Dr. Watson narra varias aventuras..."
```

Formato recomendado de CSV de consultas (`consultas.csv`):

```csv
query
"Quien es Sherlock Holmes?"
"Cual es el papel de Watson?"
```

Comando:

```bash
python llm_agent_orchestration.py \
  --project-id d7e326f5-1a40-4612-9a0b-5888b4f4034a \
  --input-csv documentos.csv \
  --input-csv-text-column text \
  --queries-csv consultas.csv \
  --queries-csv-column query \
  --output-csv output.csv
```

Si quieres pasar la API key por argumento (no recomendado para produccion):

```bash
python llm_agent_orchestration.py --apikey "TU_API_KEY" --input-csv documentos.csv --queries-csv consultas.csv
```

## Ejemplo con consultas personalizadas

```bash
python llm_agent_orchestration.py \
  --input-file aosh.txt \
  --output-csv resultados.csv \
  --queries "Who is Sherlock Holmes?" "What is Watson's role?"
```

## Salida

El script genera un CSV con columnas:

- `Thought`
- `Action`
- `Action Input`
- `Observation`
- `Final Answer`

## Llamada a herramientas con Ollama (local)

La llamada a herramientas en LLM permite que el modelo use funciones externas para resolver
tareas que van mas alla de su contexto interno: consultar datos en tiempo real, ejecutar
calculos, o inspeccionar archivos locales.

En este proyecto se incluye un flujo local con Ollama para buscar informacion en archivos
de texto/PDF e imagenes dentro de una carpeta del sistema de archivos. Esto es util cuando
quieres privacidad y ejecucion local sin depender de APIs remotas.

Script incluido:

- `ollama_tool_calling_local_fs.py`

### Que hace este script

1. Define dos herramientas: `Search inside text files` para `.txt` y `.pdf`, y `Search inside image files` para imagenes.

2. Registra las herramientas para que Ollama pueda invocarlas via `tools`.

3. Procesa los `tool_calls` devueltos por el modelo.

4. Genera respuesta final devolviendo archivos encontrados.

### Requisitos para esta parte

Instalar Ollama desde:
<https://ollama.com/download>

Comprobar instalacion:

```bash
ollama -v
```

Instalar librerias Python:

```bash
pip install ollama pymupdf
```

Descargar modelos:

```bash
ollama pull granite3.2:8b
ollama pull granite3.2-vision
```

### Estructura esperada

Coloca tus archivos en una carpeta local, por ejemplo `./files`:

- `.txt`
- `.pdf`
- `.jpg`, `.jpeg`, `.png`

### Ejecucion Ollama local

Modo interactivo:

```bash
python ollama_tool_calling_local_fs.py --files-dir ./files --start-ollama
```

Modo no interactivo:

```bash
python ollama_tool_calling_local_fs.py --files-dir ./files --query "informacion sobre perros" --start-ollama
```

Referencia conceptual del tutorial original (IBM):
<https://www.ibm.com/es-es/think/tutorials/llm-agent-orchestration-with-langchain-and-granite>

## Resolucion de problemas con sistemas multiagente (crewAI)

Se agrego un ejemplo completo de colaboracion multiagente para analisis de
transcripciones de call center con tres roles:

1. `Transcript Analyzer`
2. `Quality Assurance Specialist`
3. `Report Generator`

La implementacion vive en:

- `multiagent-collab-cs-call-center-analysis/`

### Estructura

```text
multiagent-collab-cs-call-center-analysis/
├── data/
│   └── transcript.txt
├── main.py
├── requirements-crewai.txt
└── src/customer_service_analyzer/
  ├── __init__.py
  ├── crew.py
  ├── config/
  │   ├── agents.yaml
  │   └── tasks.yaml
  └── tools/
    ├── custom_tool.py
    └── tool_helper.py
```

### Paso 1. Configura el entorno

```bash
cd multiagent-collab-cs-call-center-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-crewai.txt
```

### Paso 2. Credenciales

Define estas variables en tu `.env` (o exportalas en la shell):

- `WATSONX_APIKEY`
- `WATSONX_PROJECT_ID`
- `WATSONX_URL` (por ejemplo `https://us-south.ml.cloud.ibm.com`)
- `SERPER_API_KEY`
- `CREWAI_MODEL` (opcional, por defecto: `watsonx/ibm/granite-3-8b-instruct`)

### Paso 3. Ejecuta el sistema

```bash
python main.py
```

Resultado esperado:

1. El crew ejecuta tareas secuenciales (`transcript_analysis` -> `quality_evaluation` -> `report_generation`).
2. Se imprime el resultado final en consola.
3. Se genera `report.md` con el informe consolidado.

### Personalizacion rapida

1. Cambia roles u objetivos en `src/customer_service_analyzer/config/agents.yaml`.
2. Ajusta descripciones de tareas en `src/customer_service_analyzer/config/tasks.yaml`.
3. Agrega herramientas en `src/customer_service_analyzer/tools/custom_tool.py`.

## Varias estrategias de colaboracion multiagente

Los agentes pueden coordinarse de distintas formas segun el tipo de problema,
el nivel de incertidumbre y los requisitos de control.

### 1. Colaboracion basada en reglas

- Se basa en reglas explicitas (por ejemplo, logica if-then, maquinas de estado o reglas declarativas).
- Prioriza consistencia, trazabilidad y control del comportamiento.
- Es ideal para procesos estructurados y de baja variabilidad.

Ventajas:

- Alta previsibilidad y eficiencia operativa.
- Facil auditoria de decisiones.

Desventajas:

- Baja adaptabilidad ante escenarios nuevos.
- Escalado complejo cuando crecen casos excepcionales.

### 2. Colaboracion basada en roles

- Cada agente tiene responsabilidades especializadas (analista, evaluador, generador de reporte, etc.).
- Los agentes colaboran compartiendo contexto y resultados intermedios.
- Permite descomponer tareas complejas en subproblemas expertos.

Ventajas:

- Modularidad y reutilizacion de componentes.
- Mejor separacion de responsabilidades.

Desventajas:

- Dependencia de una buena integracion/orquestacion.
- Posibles cuellos de botella entre etapas.

### 3. Colaboracion basada en modelos

- Los agentes usan modelos internos del entorno, de otros agentes y del objetivo comun.
- Se apoyan en inferencia probabilistica y/o aprendizaje para decidir bajo incertidumbre.
- Es util cuando el contexto cambia o es parcialmente observable.

Ventajas:

- Alta flexibilidad y decisiones mas contextuales.
- Mejor respuesta a escenarios dinamicos.

Desventajas:

- Mayor complejidad de diseño y validacion.
- Coste computacional superior.

### Tabla comparativa de estrategias

| Estrategia | Como funciona | Casos de uso ideales | Complejidad | Coste computacional | Ventaja principal | Riesgo principal |
| --- | --- | --- | --- | --- | --- | --- |
| Basada en reglas | Reglas explicitas (if-then, estados, logica fija) | Procesos estables, compliance, flujos repetitivos | Baja | Bajo | Consistencia y control | Poca adaptabilidad |
| Basada en roles | Agentes especializados por responsabilidad y contexto compartido | Pipelines modulares, tareas por expertos, analisis por etapas | Media | Medio | Modularidad y especializacion | Dependencia de integracion |
| Basada en modelos | Agentes con modelos internos e inferencia bajo incertidumbre | Entornos dinamicos, informacion incompleta, planificacion adaptativa | Alta | Alto | Flexibilidad contextual | Mayor complejidad tecnica |

Guia rapida de seleccion:

- Si priorizas control y trazabilidad: colaboracion basada en reglas.
- Si priorizas productividad por especializacion: colaboracion basada en roles.
- Si priorizas adaptacion en incertidumbre: colaboracion basada en modelos.

## Marcos de referencia

1. IBM Bee Agent Framework

- Enfoque modular para construir procesos multiagente escalables.
- Incluye componentes para herramientas, memoria y monitoreo.
- Destaca por capacidades orientadas a produccion y extensibilidad.

1. LangChain Agents

- Facilita agentes con uso dinamico de herramientas.
- Permite flujos de razonamiento de varios pasos y toma de decisiones contextual.
- Amplio ecosistema de integraciones.

1. OpenAI Swarm

- Coordina agentes especializados mediante handoffs.
- Favorece transiciones fluidas entre agentes por tarea.
- Buen ajuste para arquitecturas ligeras y modulares.

### Tabla comparativa de marcos (enfoque empresarial)

| Marco | Facilidad de inicio | Control de orquestacion | Observabilidad | Coste operativo estimado | Mejor para |
| --- | --- | --- | --- | --- | --- |
| IBM Bee Agent Framework | Media | Alto | Alto | Medio | Sistemas multiagente modulares con foco productivo |
| LangChain Agents | Alta | Medio-Alto | Medio | Medio | Integraciones rapidas y flujos con herramientas heterogeneas |
| OpenAI Swarm | Alta | Medio | Medio | Medio-Alto | Handoffs entre agentes especializados y arquitecturas ligeras |
| crewAI | Alta | Alto (secuencial/paralelo por tareas) | Medio-Alto | Medio | Equipos por roles y automatizacion de procesos de negocio |

Criterio practico de eleccion:

- Si priorizas rapidez de prototipado: LangChain Agents o crewAI.
- Si priorizas control modular de nivel productivo: Bee Agent Framework.
- Si priorizas handoffs simples entre agentes: OpenAI Swarm.

## Uso de LangGraph para crear agentes ReAct

LangGraph permite construir agentes ReAct con un grafo de ejecucion ciclico.
En su forma basica, el flujo incluye:

1. Nodo de modelo (razona y decide accion).
2. Nodo de herramientas (ejecuta la accion solicitada).

En escenarios mas complejos se agrega un nodo adicional de salida estructurada.
El estado del grafo funciona como memoria de trabajo para preservar contexto
entre iteraciones del agente.

Referencia recomendada:
<https://medium.com/@jddam/gu%C3%ADa-completa-langgraph-y-el-futuro-de-los-agentes-de-ia-en-2025-2f34ceaa456f>

### Requisitos previos

- Cuenta de IBM Cloud con acceso a watsonx.ai.
- Python 3.10+ (ideal 3.11 para ciertos entornos con Poetry).
- Credenciales de watsonx.ai y, si aplica, un `space_id` de despliegue.

### Flujo de implementacion recomendado

- Generar credenciales en watsonx.ai: `watsonx_apikey`, `watsonx_url`, `space_id`.

- Preparar entorno local:

```bash
git clone git@github.com:IBM/ibmdotcom-tutorials.git
cd react-agent-langgraph-it-support/base/langgraph-react-agent/
pipx install --python 3.11 poetry
source $(poetry -q env use 3.11 && poetry env info --path)/bin/activate
poetry install
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

- Completar `config.toml` con credenciales y modelo (`model_id`).

- Conectar fuente de datos (por ejemplo IBM Cloud Object Storage) para lectura/escritura de tickets.

- Crear herramientas con `@tool` (ejemplos tipicos: `find_tickets`, `create_ticket`, `get_todays_date`).

- Registrar herramientas y construir el agente con `create_react_agent` usando un modelo compatible con tool calling.

- Ejecutar y probar por una de estas vias: localmente, desplegado en watsonx.ai (chat preview), o desde scripts de despliegue/consulta en IDE.

### Buenas practicas

- Mantener prompts de sistema orientados a tareas y validacion de parametros.
- Exponer trazas de tool calling para depuracion.
- Versionar herramientas y contratos de entrada/salida.
- Usar estado persistente para continuidad de conversaciones largas.

### Implementacion incluida en este repositorio

Se agrego una implementacion ejecutable en:

- `langgraph-react-it-support/`

Contenido principal:

- `langgraph-react-it-support/chat_react_agent.py`
- `langgraph-react-it-support/src/langgraph_react_agent/agent.py`
- `langgraph-react-it-support/src/langgraph_react_agent/tools.py`
- `langgraph-react-it-support/data/tickets.csv`

Instalacion:

```bash
cd langgraph-react-it-support
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-langgraph.txt
```

Configuracion:

1. Copia `langgraph-react-it-support/.env.example` a `langgraph-react-it-support/.env`.
2. Completa `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` y `WATSONX_URL`.

Opcional (backend en IBM Cloud Object Storage en lugar de CSV local):

- `LANGGRAPH_USE_COS=true`
- `COS_ENDPOINT`
- `COS_INSTANCE_CRN`
- `COS_BUCKET_NAME`
- `COS_CSV_FILE_NAME` (por defecto `tickets.csv`)

Ejecucion:

```bash
cd langgraph-react-it-support
source .venv/bin/activate
python chat_react_agent.py
```

Prueba rapida no interactiva (smoke test):

```bash
cd langgraph-react-it-support
source .venv/bin/activate
python smoke_test.py
```

Consultas de prueba:

- "List all current tickets"
- "Create a ticket for email outage with high urgency"
- "What is today's date?"

Notas de backend de datos:

- Si `LANGGRAPH_USE_COS=false`, los tickets se guardan en `LANGGRAPH_TICKETS_FILE` (CSV local).
- Si `LANGGRAPH_USE_COS=true`, las herramientas leen/escriben en el bucket COS configurado.

## Soluciones empresariales: watsonx Orchestrate

En escenarios empresariales, watsonx Orchestrate permite combinar:

- registro de habilidades/agentes,
- analisis de intencion,
- orquestacion de flujo (secuencias, ramas, reintentos, paralelo),
- memoria/contexto compartido,
- asistencia LLM,
- supervision humana en el ciclo.

Este modelo ayuda a operar flujos complejos de extremo a extremo con control,
trazabilidad y colaboracion entre agentes y personas.

## Predicciones futuras

- Emergera inteligencia colectiva mas robusta en equipos de agentes.
- La calidad del sistema se medira por precision, relevancia, eficiencia,
  explicabilidad y coherencia global.
- La orquestacion multiagente permitira mayor automatizacion en problemas
  multidimensionales y flujos de varios pasos.
