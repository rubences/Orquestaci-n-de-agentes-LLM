# A2A tutorial local

Este directorio implementa una demo de protocolo A2A con dos componentes:

- Servidor A2A compatible con JSON-RPC 2.0 y AgentCard.
- Cliente de chat que envia mensajes al servidor y muestra respuestas.

## Estructura

- beeai-a2a-server/beeai_chat_server.py
- beeai-a2a-client/beeai_chat_client.py

## Ejecutar (2 terminales)

Terminal servidor:

```bash
cd a2a_tutorial/beeai-a2a-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python beeai_chat_server.py
```

Terminal cliente:

```bash
cd a2a_tutorial/beeai-a2a-client
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python beeai_chat_client.py
```

## Ver AgentCard

Abre:

- http://127.0.0.1:9999/.well-known/agent-card.json

## Endpoint SSE de ejemplo

- http://127.0.0.1:9999/stream?q=weather%20in%20Tokyo
