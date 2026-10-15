import os

import uvicorn

from llm_mitm.config import load_config
from llm_mitm.logging_setup import configure_logging


def main():
    config = load_config(os.environ.get("LLM_MITM_CONFIG", "config.yaml"))
    log_config = configure_logging(config.server.json_logs, config.server.log_level)
    uvicorn.run(
        "llm_mitm.app:app",
        host=config.server.host,
        port=config.server.port,
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
