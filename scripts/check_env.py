"""Environment validation for required imports."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


@dataclass(frozen=True)
class ImportCheckResult:
    """Represents the result of an import check."""

    module: str
    success: bool
    error: str | None = None


IMPORT_NAME_MAP: dict[str, str] = {
    "chromadb": "chromadb",
    "langchain": "langchain",
    "langgraph": "langgraph",
    "openai": "openai",
    "pydantic": "pydantic",
    "pypdf": "pypdf",
    "python-dotenv": "dotenv",
    "rank-bm25": "rank_bm25",
}


def read_dependencies(pyproject_path: Path) -> list[str]:
    """Read dependencies from pyproject.toml."""

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        return []
    return [str(item) for item in dependencies]


def check_import(module_name: str) -> ImportCheckResult:
    """Attempt to import a module and return result."""

    try:
        __import__(module_name)
    except Exception as exc:  # noqa: BLE001 - report any import failure
        return ImportCheckResult(module=module_name, success=False, error=str(exc))
    return ImportCheckResult(module=module_name, success=True)


def main() -> int:
    """Validate required dependencies can be imported."""

    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    dependencies = read_dependencies(pyproject_path)
    required_modules = [
        IMPORT_NAME_MAP[name]
        for name in dependencies
        if name in IMPORT_NAME_MAP
    ]

    results = [check_import(module) for module in required_modules]
    failures = [result for result in results if not result.success]

    if failures:
        print(f"{RED}Dependency import check failed:{RESET}", file=sys.stderr)
        for failure in failures:
            message = failure.error or "Unknown error"
            print(f"{RED}- {failure.module}: {message}{RESET}", file=sys.stderr)
        return 1

    print(f"{GREEN}All required dependencies imported successfully.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
