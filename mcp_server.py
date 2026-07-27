"""Entry-point shim. The server itself lives in src/mcp_server.py.

This exists so an already-installed claude_desktop_config.json keeps working
verbatim after the move into src/ — the config points at `mcp_server.py` at the
repo root, and rewriting a config that lives outside this repo is exactly the
kind of breakage nobody notices until the tools go quiet.

    {
      "mcpServers": {
        "ff-edge": {
          "command": "uv",
          "args": [
            "--directory", "/absolute/path/to/ff-edge",
            "run", "python", "mcp_server.py"
          ]
        }
      }
    }
"""

from __future__ import annotations

from src.mcp_server import mcp

if __name__ == "__main__":
    mcp.run()
