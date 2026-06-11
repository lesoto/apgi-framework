"""Validate all protocol JSON files against protocols/schemas/protocol.schema.json."""

import json
import pathlib
import sys

import jsonschema

ROOT = pathlib.Path(__file__).parent.parent
SCHEMA_PATH = ROOT / "protocols" / "schemas" / "protocol.schema.json"
PROTOCOLS_GLOB = sorted((ROOT / "protocols").glob("*.json"))

schema = json.loads(SCHEMA_PATH.read_text())
failed = []
for f in PROTOCOLS_GLOB:
    try:
        jsonschema.validate(json.loads(f.read_text()), schema)
        print(f"  OK   {f.name}")
    except jsonschema.ValidationError as e:
        print(f"  FAIL {f.name}: {e.message}")
        failed.append(f.name)

if failed:
    print(f"\nValidation failed for: {failed}", file=sys.stderr)
    sys.exit(1)
print("All protocols valid.")
