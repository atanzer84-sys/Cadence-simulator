import pytest
import logging
import configs.config_parsing as cp


# ----------------------------------------------------------------------
# parse_simple_kv
# ----------------------------------------------------------------------

# Tests: parse_simple_kv
# Behavior: raises FileNotFoundError and logs ERROR when file missing
def test_parse_simple_kv_missing_file_raises(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    missing = tmp_path / "nope.cfg"

    with pytest.raises(FileNotFoundError):
        cp.parse_simple_kv(missing)

    assert any(rec.levelname == "ERROR" for rec in caplog.records)


# Tests: parse_simple_kv
# Behavior: ignores blank lines and comment lines
def test_parse_simple_kv_ignores_blank_and_comments(tmp_path):
    content = """
    # comment
    a = 1

    # another
    b = 2
    """
    p = tmp_path / "cfg.txt"
    p.write_text(content)

    out = cp.parse_simple_kv(p)
    assert out == {"a": "1", "b": "2"}


# Tests: parse_simple_kv
# Behavior: strips inline comments after values
def test_parse_simple_kv_strips_inline_comments(tmp_path):
    content = "a = 1   # inline comment"
    p = tmp_path / "cfg.txt"
    p.write_text(content)

    out = cp.parse_simple_kv(p)
    assert out == {"a": "1"}


# Tests: parse_simple_kv
# Behavior: ignores lines without '='
def test_parse_simple_kv_ignores_lines_without_equals(tmp_path):
    content = """
    a = 1
    this line has no equals
    b = 2
    """
    p = tmp_path / "cfg.txt"
    p.write_text(content)

    out = cp.parse_simple_kv(p)
    assert out == {"a": "1", "b": "2"}


# Tests: parse_simple_kv
# Behavior: trims whitespace around keys and values
def test_parse_simple_kv_trims_whitespace(tmp_path):
    content = "   a   =    123   "
    p = tmp_path / "cfg.txt"
    p.write_text(content)

    out = cp.parse_simple_kv(p)
    assert out == {"a": "123"}


# Tests: parse_simple_kv
# Behavior: last duplicate key wins
def test_parse_simple_kv_duplicate_keys_last_wins(tmp_path):
    content = """
    a = 1
    a = 2
    """
    p = tmp_path / "cfg.txt"
    p.write_text(content)

    out = cp.parse_simple_kv(p)
    assert out == {"a": "2"}


# Tests: parse_simple_kv
# Behavior: value is split at first '#' only
def test_parse_simple_kv_value_with_multiple_hashes(tmp_path):
    content = "a = 1#2#3"
    p = tmp_path / "cfg.txt"
    p.write_text(content)

    out = cp.parse_simple_kv(p)
    assert out == {"a": "1"}


# Tests: parse_simple_kv
# Behavior: splits at first '=' only
def test_parse_simple_kv_value_contains_equals(tmp_path):
    content = "a = 1=2=3"
    p = tmp_path / "cfg.txt"
    p.write_text(content)

    out = cp.parse_simple_kv(p)
    assert out == {"a": "1=2=3"}


# ----------------------------------------------------------------------
# as_int
# ----------------------------------------------------------------------

# Tests: as_int
# Behavior: parses valid integers
def test_as_int_valid():
    assert cp.as_int("10", key="x") == 10
    assert cp.as_int(5, key="x") == 5


# Tests: as_int
# Behavior: invalid int logs ERROR mentioning the key name
def test_as_int_invalid(caplog):
    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError) as exc:
        cp.as_int("not_int", key="abc")

    assert "abc" in str(exc.value)
    assert any(
        rec.levelname == "ERROR" and "abc" in rec.message
        for rec in caplog.records
    )

# ----------------------------------------------------------------------
# as_float
# ----------------------------------------------------------------------

# Tests: as_float
# Behavior: parses valid floats
def test_as_float_valid():
    assert cp.as_float("1.5", key="x") == 1.5
    assert cp.as_float(2, key="x") == 2.0


# Tests: as_float
# Behavior: invalid float logs ERROR mentioning the key name
def test_as_float_invalid(caplog):
    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError) as exc:
        cp.as_float("nope", key="f")

    assert "f" in str(exc.value)
    assert any(
        rec.levelname == "ERROR" and "f" in rec.message
        for rec in caplog.records
    )


# ----------------------------------------------------------------------
# as_bool
# ----------------------------------------------------------------------

# Tests: as_bool
# Behavior: accepts common true/false spellings
@pytest.mark.parametrize("val, expected", [
    ("1", True), ("true", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("off", False), ("", False),
])
def test_as_bool_valid(val, expected):
    assert cp.as_bool(val, key="b") is expected


# Tests: as_bool
# Behavior: trims whitespace around tokens
@pytest.mark.parametrize("raw, expected", [
    (" true ", True),
    ("  yes  ", True),
    (" off ", False),
    (" 0 ", False),
])
def test_as_bool_whitespace_wrapped(raw, expected):
    assert cp.as_bool(raw, key="b") is expected


# Tests: as_bool
# Behavior: invalid boolean logs ERROR mentioning the key name
def test_as_bool_invalid(caplog):
    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError) as exc:
        cp.as_bool("maybe", key="flag")

    assert "flag" in str(exc.value)
    assert any(
        rec.levelname == "ERROR" and "flag" in rec.message
        for rec in caplog.records
    )


# ----------------------------------------------------------------------
# as_optional_int
# ----------------------------------------------------------------------

# Tests: as_optional_int
# Behavior: returns None for blank/none
@pytest.mark.parametrize("val", ["", "none", "None", None])
def test_as_optional_int_none(val):
    assert cp.as_optional_int(val) is None


# Tests: as_optional_int
# Behavior: whitespace-wrapped 'none' returns None
@pytest.mark.parametrize("raw", [" none ", "  NONE  ", "   None   "])
def test_as_optional_int_whitespace_none(raw):
    assert cp.as_optional_int(raw) is None


# Tests: as_optional_int
# Behavior: parses valid ints
def test_as_optional_int_valid():
    assert cp.as_optional_int("10") == 10
    assert cp.as_optional_int(5) == 5

# Tests: as_optional_int
# Behavior: invalid int logs ERROR mentioning the value
def test_as_optional_int_invalid(caplog):
    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError):
        cp.as_optional_int("bad")

    assert any(
        rec.levelname == "ERROR" and "bad" in rec.message
        for rec in caplog.records
    )


# ----------------------------------------------------------------------
# as_optional_float
# ----------------------------------------------------------------------

# Tests: as_optional_float
# Behavior: returns None for blank/none
@pytest.mark.parametrize("val", ["", "none", "None", None])
def test_as_optional_float_none(val):
    assert cp.as_optional_float(val) is None


# Tests: as_optional_float
# Behavior: whitespace-wrapped 'none' returns None
@pytest.mark.parametrize("raw", [" none ", "  NONE  ", "   None   "])
def test_as_optional_float_whitespace_none(raw):
    assert cp.as_optional_float(raw) is None


# Tests: as_optional_float
# Behavior: parses valid floats
def test_as_optional_float_valid():
    assert cp.as_optional_float("1.5") == 1.5
    assert cp.as_optional_float(2) == 2.0


# ----------------------------------------------------------------------
# as_optional_str
# ----------------------------------------------------------------------

# Tests: as_optional_str
# Behavior: returns None for blank/none
@pytest.mark.parametrize("val", ["", "none", "None", None])
def test_as_optional_str_none(val):
    assert cp.as_optional_str(val) is None


# Tests: as_optional_str
# Behavior: whitespace-wrapped 'none' returns None
@pytest.mark.parametrize("raw", [" none ", "  NONE  ", "   None   "])
def test_as_optional_str_whitespace_none(raw):
    assert cp.as_optional_str(raw) is None


# Tests: as_optional_str
# Behavior: returns stripped string
def test_as_optional_str_valid():
    assert cp.as_optional_str("  abc  ") == "abc"


# ----------------------------------------------------------------------
# as_optional_lower_str
# ----------------------------------------------------------------------

# Tests: as_optional_lower_str
# Behavior: lowercases non-None strings
def test_as_optional_lower_str_valid():
    assert cp.as_optional_lower_str(" ABC ") == "abc"


# Tests: as_optional_lower_str
# Behavior: whitespace-wrapped 'none' returns None
@pytest.mark.parametrize("raw", [" none ", "  NONE  ", "   None   "])
def test_as_optional_lower_str_whitespace_none(raw):
    assert cp.as_optional_lower_str(raw) is None


# ----------------------------------------------------------------------
# sanitize_target_name
# ----------------------------------------------------------------------

# Tests: sanitize_target_name
# Behavior: trims whitespace and strips matching quotes
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Star", "Star"),
        (" Star ", "Star"),
        ("'Star'", "Star"),
        ('"Star"', "Star"),
        ("' Star '", "Star"),
        ('" Star "', "Star"),
    ],
)
def test_sanitize_target_name_basic(raw, expected):
    assert cp.sanitize_target_name(raw) == expected


# Tests: sanitize_target_name
# Behavior: allows empty or whitespace-only names
@pytest.mark.parametrize("raw", ["", "   ", "''", '""'])
def test_sanitize_target_name_empty(raw):
    assert cp.sanitize_target_name(raw) == ""

# Tests: as_optional_float
# Behavior: invalid float raises ValueError (no logging expected)
def test_as_optional_float_invalid():
    with pytest.raises(ValueError):
        cp.as_optional_float("not_a_float")

# Tests: sanitize_target_name
# Behavior: mismatched or single-sided quotes are not stripped
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("'Star", "'Star"),      # leading only
        ('Star"', 'Star"'),      # trailing only
        ("'Star\"", "'Star\""),  # mismatched
        ("\"Star'", "\"Star'"),  # mismatched reversed
    ],
)
def test_sanitize_target_name_mismatched_quotes(raw, expected):
    assert cp.sanitize_target_name(raw) == expected

# Tests: _ensure_min_le_max
# Behavior: does nothing when min_val <= max_val
def test_ensure_min_le_max_valid():
    from configs.global_config import _ensure_min_le_max
    # Should not raise
    _ensure_min_le_max(1, 2, key_min="a", key_max="b")
    _ensure_min_le_max(5, 5, key_min="a", key_max="b")

# Tests: _ensure_min_le_max
# Behavior: raises ValueError when min_val > max_val and message contains both keys
def test_ensure_min_le_max_invalid():
    from configs.global_config import _ensure_min_le_max
    with pytest.raises(ValueError) as exc:
        _ensure_min_le_max(10, 5, key_min="minExp", key_max="maxExp")
    msg = str(exc.value)
    assert "minExp" in msg
    assert "maxExp" in msg
    assert "<=" in msg
