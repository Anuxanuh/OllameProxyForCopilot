import logging
from pathlib import Path

from fastapi import FastAPI

from proxy_core import ProxyState
from proxy_core.routes import (
    create_chat_router,
    create_embed_router,
    create_models_router,
    create_system_router,
    register_request_trace_middleware,
)

app = FastAPI(title="Ollama-Compatible Proxy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ollama_proxy")

CONFIG_PATH = Path(__file__).with_name("proxy_config.json")
PORT = 11434
OLLAMA_VERSION = "0.6.4"

STATE = ProxyState(CONFIG_PATH, logger)

register_request_trace_middleware(app, logger)
app.include_router(create_system_router(OLLAMA_VERSION))
app.include_router(create_models_router(STATE, logger))
app.include_router(create_chat_router(STATE, logger))
app.include_router(create_embed_router(STATE, logger))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ollama_proxy:app", host="0.0.0.0", port=PORT, reload=False)
