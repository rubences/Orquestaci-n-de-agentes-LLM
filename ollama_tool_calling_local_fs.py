"""Tool calling con Ollama sobre sistema de archivos local.

Flujo:
1. Define herramientas para buscar en archivos de texto/PDF e imagenes.
2. Registra dichas herramientas para que el modelo pueda llamarlas.
3. Ejecuta la consulta del usuario y procesa tool_calls.
4. Devuelve solo nombres de archivo encontrados.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Callable

import ollama
import pymupdf

TEXT_MODEL = "granite3.2:8b"
VISION_MODEL = "granite3.2-vision"


def search_text_files(keyword: str, files_dir: Path) -> str:
    """Busca una palabra clave en .txt y .pdf usando el modelo de texto."""
    for entry in files_dir.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue

        if entry.suffix.lower() == ".pdf":
            document_text = ""
            doc = pymupdf.open(entry)
            for page in doc:
                document_text += page.get_text()
            doc.close()

            prompt = (
                "Respond only 'yes' or 'no', do not add any additional information. "
                f"Is the following text about {keyword}? {document_text}"
            )
            res = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
            if "yes" in res["message"]["content"].lower():
                return str(entry)

        elif entry.suffix.lower() == ".txt":
            file_content = entry.read_text(encoding="utf-8", errors="ignore")
            prompt = (
                "Respond only 'yes' or 'no', do not add any additional information. "
                f"Is the following text about {keyword}? {file_content}"
            )
            res = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
            if "yes" in res["message"]["content"].lower():
                return str(entry)

    return "None"


def search_image_files(keyword: str, files_dir: Path) -> str:
    """Busca una palabra clave en la descripcion de imagenes locales usando modelo vision."""
    image_ext = {".jpg", ".jpeg", ".png"}

    for entry in files_dir.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in image_ext:
            continue

        res = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image in short sentences. Use simple phrases first and then describe it more fully.",
                    "images": [str(entry)],
                }
            ],
        )
        if keyword.lower() in res["message"]["content"].lower():
            return str(entry)

    return "None"


def build_tools() -> list[dict]:
    """Define el schema de herramientas para la llamada de funciones en Ollama."""
    return [
        {
            "type": "function",
            "function": {
                "name": "Search inside text files",
                "description": "This tool searches in PDF or plaintext files in the local file system for mentions of the keyword.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Generate one keyword from the user request to search in text files",
                        }
                    },
                    "required": ["keyword"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Search inside image files",
                "description": "This tool searches photos or image files in the local file system for the keyword.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Generate one keyword from the user request to search in image files",
                        }
                    },
                    "required": ["keyword"],
                },
            },
        },
    ]


def ensure_ollama_running() -> None:
    """Inicia ollama serve si no esta corriendo."""
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_tool_calling(user_input: str, files_dir: Path) -> str:
    available_functions: dict[str, Callable[..., str]] = {
        "Search inside text files": lambda keyword: search_text_files(keyword, files_dir),
        "Search inside image files": lambda keyword: search_image_files(keyword, files_dir),
    }

    messages = [{"role": "user", "content": user_input}]
    response = ollama.chat(model=TEXT_MODEL, messages=messages, tools=build_tools())

    output: list[str] = []

    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            function_to_call = available_functions.get(tool_call.function.name)
            if not function_to_call:
                continue

            tool_res = function_to_call(**tool_call.function.arguments)
            if str(tool_res) != "None":
                output.append(str(tool_res))

        messages.append(response.message)
        prompt = (
            "If the tool output contains one or more file names, then give the user only the filename found. "
            "Do not add additional details. If the tool output is empty ask the user to try again. "
            "Here is the tool output: "
        )
        messages.append({"role": "tool", "content": prompt + ", ".join(output)})

        final_response = ollama.chat(model=TEXT_MODEL, messages=messages)
        return final_response.message.content

    return "No tool calls returned from model"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tool calling local con Ollama")
    parser.add_argument("--files-dir", default="./files", help="Carpeta local con txt/pdf/jpg/png")
    parser.add_argument("--query", default=None, help="Consulta del usuario")
    parser.add_argument("--start-ollama", action="store_true", help="Inicia `ollama serve` antes de ejecutar")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    files_dir = Path(args.files_dir)
    if not files_dir.exists() or not files_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de archivos: {files_dir}")

    if args.start_ollama:
        ensure_ollama_running()

    user_input = args.query or input("What would you like to search for? ").strip()
    if not user_input:
        raise ValueError("Debes proporcionar una consulta con --query o por input interactivo.")

    result = run_tool_calling(user_input, files_dir)
    print("Final response:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
