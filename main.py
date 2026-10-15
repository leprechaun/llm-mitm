import os

import uvicorn

from llm_mitm.config import load_config
from llm_mitm.logging_setup import configure_logging


def main():
    config = load_config(os.environ.get("LLM_MITM_CONFIG", "config.yaml"))
    configure_logging(config.server.json_logs, config.server.log_level)
    uvicorn.run(
        "llm_mitm.app:app",
        host=config.server.host,
        port=config.server.port,
        # We've already wired every logger uvicorn cares about ourselves --
        # None skips uvicorn's own dictConfig call, which would otherwise
        # overwrite the handlers configure_logging() just set up.
        log_config=None,
    )


if __name__ == "__main__":
    main()
