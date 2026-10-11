import os

import uvicorn

from llm_mitm.config import load_config


def main():
    config = load_config(os.environ.get("LLM_MITM_CONFIG", "config.yaml"))
    uvicorn.run("llm_mitm.app:app", host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
