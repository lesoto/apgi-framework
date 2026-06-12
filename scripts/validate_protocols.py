"""Validate all protocol JSON files against protocols/schemas/protocol.schema.json."""

import json
import pathlib
import sys

import jsonschema

ROOT = pathlib.Path(__file__).parent.parent
SCHEMA_PATH = ROOT / "protocols" / "schemas" / "protocol.schema.json"
PROTOCOLS_GLOB = sorted((ROOT / "protocols").glob("*.json"))


def validate_protocols(
    schema_path: pathlib.Path = SCHEMA_PATH,
    protocols_glob: list[pathlib.Path] = PROTOCOLS_GLOB,
) -> int:
    """Validate all protocol JSON files and return a process exit code."""

    schema = json.loads(schema_path.read_text())
    failed = []
    for f in protocols_glob:
        try:
            jsonschema.validate(json.loads(f.read_text()), schema)
            print(f"  OK   {f.name}")
        except jsonschema.ValidationError as e:
            print(f"  FAIL {f.name}: {e.message}")
            failed.append(f.name)

    if failed:
        print(f"\nValidation failed for: {failed}", file=sys.stderr)
        return 1
    print("All protocols valid.")
    return 0


def main() -> None:
    raise SystemExit(validate_protocols())


if __name__ == "__main__":
    main()
