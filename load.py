"""
Download all files under the `algo/` directory from a Snowflake Workspace
to your local filesystem.

Workspace files are exposed as a stage-like path:
    snow://workspace/USER$.PUBLIC.DEFAULT$/versions/live/<path>
(USER$ is a literal identifier, not your username.)

This script lists every file under `algo/` and downloads each one,
preserving the directory structure locally.
"""

import os
from pathlib import Path

from snowflake.snowpark import Session


# ---------------------------------------------------------------------------
# Configuration -- fill these in for your environment
# ---------------------------------------------------------------------------
CONNECTION_PARAMS = {
    "account":   "HXRTJEP-RLC89789",  # org-account format
    "user":      "VG",              # your Snowflake username
    "password":  "Assasiancreed123#", # or use "authenticator": "externalbrowser"
    "role":      "ACCOUNTADMIN",
    "warehouse": "COMPUTE_WH",
    "database":  "SNOWFLAKE",       # any accessible db; not used for workspace I/O
    "schema":    "PUBLIC",
}

# Workspace stage root (literal). Do NOT replace USER$ with your username.
WORKSPACE_ROOT = "snow://workspace/USER$.PUBLIC.DEFAULT$/versions/live"

# Source directory inside the workspace (relative to versions/live/).
WORKSPACE_SUBDIR = "algo/"

# Local destination directory.
LOCAL_DEST_DIR = "./algo_download"


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------
def download_workspace_dir(
    session: Session,
    workspace_root: str,
    subdir: str,
    local_dest: str,
) -> None:
    list_path = f"'{workspace_root}/{subdir}'"

    rows = session.sql(f"LIST {list_path}").collect()
    if not rows:
        print(f"No files found under {list_path}")
        return

    Path(local_dest).mkdir(parents=True, exist_ok=True)

    for row in rows:
        # row["name"] looks like "/versions/live/algo/foo/bar.py"
        full_name = row["name"].lstrip("/")
        marker = f"versions/live/{subdir}"
        idx = full_name.find(marker)
        rel_path = (
            full_name[idx + len(marker):] if idx >= 0
            else full_name.split(subdir, 1)[-1]
        )

        local_file = Path(local_dest) / rel_path
        local_file.parent.mkdir(parents=True, exist_ok=True)

        get_src = f"'{workspace_root}/{subdir}{rel_path}'"
        get_dst = f"file://{local_file.parent.resolve()}/"

        print(f"Downloading {rel_path} -> {local_file}")
        session.sql(f"GET {get_src} {get_dst}").collect()


def main() -> None:
    session = Session.builder.configs(CONNECTION_PARAMS).create()
    try:
        download_workspace_dir(
            session=session,
            workspace_root=WORKSPACE_ROOT,
            subdir=WORKSPACE_SUBDIR,
            local_dest=LOCAL_DEST_DIR,
        )
        print(f"Done. Files saved to {os.path.abspath(LOCAL_DEST_DIR)}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
