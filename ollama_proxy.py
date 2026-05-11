import logging
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

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

def configure_logging() -> None:
    logs_dir = Path(__file__).with_name("Logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        filename=str(logs_dir / "ollama_proxy.log"),
        when="H",
        interval=1,
        backupCount=168,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d_%H.log"
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


configure_logging()
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
