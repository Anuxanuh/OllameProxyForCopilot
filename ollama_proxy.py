import argparse
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

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s")
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

    # Dedicated full-body trace logger: file only, no console noise.
    file_trace_logger = logging.getLogger("ollama_proxy.filetrace")
    file_trace_logger.setLevel(logging.INFO)
    file_trace_logger.handlers.clear()
    file_trace_logger.addHandler(file_handler)
    file_trace_logger.propagate = False


configure_logging()
logger = logging.getLogger("ollama_proxy")

CONFIG_PATH = Path(__file__).with_name("proxy_config.json")
PORT = 11434
OLLAMA_VERSION = "0.6.4"

# Allow overriding CONFIG_PATH and PORT via command-line arguments.
_arg_parser = argparse.ArgumentParser(description="Ollama-Compatible Proxy", add_help=False)
_arg_parser.add_argument("--config", type=Path, default=None, help="Path to proxy config JSON")
_arg_parser.add_argument("--port", type=int, default=None, help="Listen port")
_cli_args, _ = _arg_parser.parse_known_args()
if _cli_args.config:
    CONFIG_PATH = _cli_args.config
if _cli_args.port:
    PORT = _cli_args.port

STATE = ProxyState(CONFIG_PATH, logger)

register_request_trace_middleware(app, logger)
app.include_router(create_system_router(OLLAMA_VERSION))
app.include_router(create_models_router(STATE, logger))
app.include_router(create_chat_router(STATE, logger))
app.include_router(create_embed_router(STATE, logger))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ollama_proxy:app", host="0.0.0.0", port=PORT, reload=False)
