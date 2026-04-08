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
