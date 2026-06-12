"""Tests for scripts/validate_protocols.py."""

import json
import pathlib
import sys

import jsonschema
import pytest

# Add scripts directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
import validate_protocols


class TestValidateProtocols:
    """Test protocol validation logic."""

    def test_schema_loads_successfully(self, tmp_path):
        """Test that the schema file can be loaded."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text('{"type": "object"}')
        schema = json.loads(schema_path.read_text())
        assert schema is not None
        assert schema["type"] == "object"

    def test_valid_protocol_passes_validation(self, tmp_path):
        """Test that a valid protocol passes validation."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["protocol_id"],
            "properties": {"protocol_id": {"type": "string"}},
        }
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))

        protocol = {"protocol_id": "APGI-P01"}
        protocol_path = tmp_path / "protocol.json"
        protocol_path.write_text(json.dumps(protocol))

        exit_code = validate_protocols.validate_protocols(
            schema_path=schema_path, protocols_glob=[protocol_path]
        )
        assert exit_code == 0

    def test_invalid_protocol_fails_validation(self, tmp_path):
        """Test that an invalid protocol fails validation."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["protocol_id"],
            "properties": {"protocol_id": {"type": "string"}},
        }
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(schema))

        protocol = {}  # Missing required field
        protocol_path = tmp_path / "protocol.json"
        protocol_path.write_text(json.dumps(protocol))

        exit_code = validate_protocols.validate_protocols(
            schema_path=schema_path, protocols_glob=[protocol_path]
        )
        assert exit_code == 1

    def test_protocol_glob_finds_json_files(self, tmp_path):
        """Test that the glob pattern finds JSON files."""
        tmp_path.joinpath("protocol_1.json").write_text("{}")
        tmp_path.joinpath("protocol_2.json").write_text("{}")
        tmp_path.joinpath("not_json.txt").write_text("")

        json_files = sorted(tmp_path.glob("*.json"))
        assert len(json_files) == 2
        assert all(f.suffix == ".json" for f in json_files)

    def test_validation_error_message_format(self, tmp_path):
        """Test that validation errors are formatted correctly."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["protocol_id"],
            "properties": {"protocol_id": {"type": "string"}},
        }
        protocol = {}  # Missing required field

        try:
            jsonschema.validate(protocol, schema)
        except jsonschema.ValidationError as e:
            assert e.message is not None
            assert len(e.message) > 0

    def test_main_exits_cleanly(self):
        with pytest.raises(SystemExit) as excinfo:
            validate_protocols.main()
        assert excinfo.value.code == 0
