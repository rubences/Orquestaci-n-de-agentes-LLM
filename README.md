# Orquestacion de agentes LLM con Python

Este repositorio contiene una implementacion en Python para orquestar un flujo RAG
con LangChain + FAISS + IBM watsonx Granite.

Referencia del tutorial base:
<https://www.ibm.com/es-es/think/tutorials/llm-agent-orchestration-with-langchain-and-granite#228874318>

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
