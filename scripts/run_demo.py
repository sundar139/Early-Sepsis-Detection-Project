from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(
        ["streamlit", "run", "src/early_sepsis/demo/app.py"],
        check=True,
    )


if __name__ == "__main__":
    main()
