"""LLM Agent Orchestration with LangChain + IBM Granite (Python script).

This script reproduces the workflow from the notebook:
1. Read a text file (for example, aosh.txt).
2. Split text into chunks.
3. Build a FAISS vector index.
4. Query relevant context.
5. Use IBM Granite (watsonx.ai) to generate answers.
6. Save results in CSV.
"""

from __future__ import annotations

import argparse
import getpass
import os

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxLLM

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

DEFAULT_PROJECT_ID = "d7e326f5-1a40-4612-9a0b-5888b4f4034a"
DEFAULT_WML_URL = "https://us-south.ml.cloud.ibm.com"
DEFAULT_MODEL_ID = "ibm/granite-3-8b-instruct"


def get_credentials(wml_url: str, project_id: str | None, apikey: str | None) -> tuple[dict[str, str], str]:
    """Load watsonx credentials from env vars or prompt for secure input."""
    api_key = apikey or os.getenv("WML_APIKEY")
    if not api_key:
        api_key = getpass.getpass("Ingresa tu WML API key: ")

    resolved_project_id = project_id or os.getenv("PROJECT_ID") or DEFAULT_PROJECT_ID

    credentials = {
        "url": wml_url,
        "apikey": api_key,
    }
    return credentials, resolved_project_id


def init_llm(credentials: dict[str, str], project_id: str, model_id: str) -> WatsonxLLM:
    """Initialize IBM Granite LLM via watsonx.ai."""
    return WatsonxLLM(
        model_id=model_id,
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=project_id,
        params={"max_new_tokens": 150},
    )


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_text_from_csv(file_path: str, text_column: str) -> str:
    """Extract text from a CSV by concatenating values from a text column."""
    df = pd.read_csv(file_path)
    if text_column not in df.columns:
        raise ValueError(
            f"La columna '{text_column}' no existe en {file_path}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    text_series = df[text_column].dropna().astype(str)
    if text_series.empty:
        raise ValueError(
            f"La columna '{text_column}' en {file_path} no contiene texto utilizable."
        )
    return "\n".join(text_series.tolist())


def load_queries_from_csv(file_path: str, query_column: str) -> list[str]:
    """Load query list from a CSV column."""
    df = pd.read_csv(file_path)
    if query_column not in df.columns:
        raise ValueError(
            f"La columna '{query_column}' no existe en {file_path}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    queries = df[query_column].dropna().astype(str).str.strip().tolist()
    queries = [q for q in queries if q]
    if not queries:
        raise ValueError(
            f"La columna '{query_column}' en {file_path} no contiene consultas validas."
        )
    return queries


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split long text into chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def create_vector_index(chunks: list[str]) -> FAISS:
    """Create a FAISS vector index from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)


def query_index_with_granite_dynamic(vector_store: FAISS, query: str, llm: WatsonxLLM) -> dict[str, str]:
    """Retrieve relevant chunks and generate a Granite answer."""
    print("\n> Entering new AgentExecutor chain...")
    thought = f"The query '{query}' requires context from the book to provide an accurate response."
    print(f" Thought: {thought}")

    action = "Search FAISS Vector Store"
    print(f" Action: {action}")
    print(f" Action Input: \"{query}\"")

    results = vector_store.similarity_search(query, k=3)
    observation = "\n".join([result.page_content for result in results])
    print(f" Observation:\n{observation}\n")

    prompt = f"Context:\n{observation}\n\nQuestion: {query}\nAnswer:"
    print(" Thought: Combining retrieved context with the query to generate a detailed answer.")
    final_answer = llm.invoke(prompt).strip()
    print(f" Final Answer: {final_answer}")
    print("\n> Finished chain.")

    return {
        "Thought": thought,
        "Action": action,
        "Action Input": query,
        "Observation": observation,
        "Final Answer": final_answer,
    }


def dynamic_output_to_dataframe(
    vector_store: FAISS,
    queries: list[str],
    llm: WatsonxLLM,
    csv_filename: str,
) -> pd.DataFrame:
    """Run multiple queries and persist the output as CSV."""
    output_data: list[dict[str, str]] = []

    for query in queries:
        output = query_index_with_granite_dynamic(vector_store, query, llm)
        output_data.append(output)

    df = pd.DataFrame(output_data)
    print("\nFinal DataFrame:")
    print(df)

    df.to_csv(csv_filename, index=False)
    print(f"\nOutput saved to {csv_filename}")
    return df


def default_queries() -> list[str]:
    return [
        "What is the plot of 'A Scandal in Bohemia'?",
        "Who is Dr. Watson, and what role does he play in the stories?",
        "Describe the relationship between Sherlock Holmes and Irene Adler.",
        "What methods does Sherlock Holmes use to solve cases?",
    ]


def run_workflow(args: argparse.Namespace) -> pd.DataFrame:
    credentials, project_id = get_credentials(args.wml_url, args.project_id, args.apikey)
    llm = init_llm(credentials, project_id, args.model_id)

    if args.input_csv:
        text = extract_text_from_csv(args.input_csv, args.input_csv_text_column)
    else:
        text = extract_text_from_txt(args.input_file)

    chunks = split_text_into_chunks(text, args.chunk_size, args.chunk_overlap)
    vector_store = create_vector_index(chunks)

    if args.queries_csv:
        queries = load_queries_from_csv(args.queries_csv, args.queries_csv_column)
    elif args.queries:
        queries = args.queries
    else:
        queries = default_queries()

    return dynamic_output_to_dataframe(vector_store, queries, llm, args.output_csv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM agent orchestration with LangChain and IBM Granite")
    parser.add_argument("--input-file", default="aosh.txt", help="Path to text source file")
    parser.add_argument("--input-csv", default=None, help="Path to CSV source file")
    parser.add_argument(
        "--input-csv-text-column",
        default="text",
        help="CSV column containing source text when using --input-csv",
    )
    parser.add_argument("--output-csv", default="output.csv", help="Path to output CSV")
    parser.add_argument("--project-id", default=None, help="IBM watsonx project ID")
    parser.add_argument("--apikey", default=None, help="IBM watsonx API key (or use WML_APIKEY env var)")
    parser.add_argument("--wml-url", default=DEFAULT_WML_URL, help="IBM watsonx ML URL")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model ID")
    parser.add_argument("--chunk-size", type=int, default=500, help="Text splitter chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Text splitter overlap")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Optional list of custom queries. If omitted, default queries are used.",
    )
    parser.add_argument("--queries-csv", default=None, help="Path to CSV file with queries")
    parser.add_argument(
        "--queries-csv-column",
        default="query",
        help="Column name for queries when using --queries-csv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.input_csv and not os.path.exists(args.input_csv):
        raise FileNotFoundError(
            f"No se encontro el CSV de entrada: {args.input_csv}. "
            "Usa una ruta valida o ejecuta con --input-file."
        )

    if not args.input_csv and not os.path.exists(args.input_file):
        raise FileNotFoundError(
            f"No se encontro el archivo de entrada: {args.input_file}. "
            "Descarga aosh.txt desde Project Gutenberg, usa --input-file o --input-csv."
        )

    if args.queries_csv and not os.path.exists(args.queries_csv):
        raise FileNotFoundError(
            f"No se encontro el CSV de consultas: {args.queries_csv}. "
            "Usa una ruta valida o pasa consultas con --queries."
        )

    run_workflow(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
