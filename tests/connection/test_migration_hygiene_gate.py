from pathlib import Path
from collections.abc import Iterable

BANNED_TOKENS = [
    "_input_streams",
    "_output_streams",
    "write_port_conditions",
    "map_port_conditions_to_balance_terms",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def migrated_target_paths() -> list[Path]:
    root = repo_root()
    allowlist = [
        root / "src" / "modular_simulation" / "connection" / "__init__.py",
        root / "src" / "modular_simulation" / "connection" / "process_interface.py",
        root / "src" / "modular_simulation" / "connection" / "topology.py",
        root / "src" / "modular_simulation" / "connection" / "network.py",
        root / "tests" / "connection" / "test_topology_domain.py",
        root / "src" / "modular_simulation" / "connection" / "hydraulic_compile.py",
        root / "tests" / "connection" / "test_graph_compile_validation.py",
        root / "src" / "modular_simulation" / "connection" / "process_binding.py",
        root / "tests" / "connection" / "test_process_binding.py",
        root / "tests" / "connection" / "test_process_interface_adapter.py",
        root / "tests" / "connection" / "test_process_balance_mapping.py",
        root / "tests" / "connection" / "test_connection_integration_physics.py",
        root / "tests" / "connection" / "test_hydraulic_solver_solvability.py",
        root / "tests" / "connection" / "test_connection_layer_sequence.py",
        root / "tests" / "connection" / "test_connection_layer_picard_gate.py",
        root / "examples" / "connection_layer" / "two_tank_reaction_comparison.py",
        root / "examples" / "connection_layer" / "pump_causality_demo.py",
    ]
    return [p for p in allowlist if p.exists()]


def tokens_in_text(text: str) -> list[str]:
    return [tok for tok in BANNED_TOKENS if tok in text]


def read_tokens_from_path(p: Path) -> list[str]:
    try:
        data = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return tokens_in_text(data)


def scan_paths(paths: Iterable[Path | str]) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for item in paths:
        p = item if isinstance(item, Path) else Path(item)
        if not p.exists():
            continue
        for tok in read_tokens_from_path(p):
            findings.append((str(p), tok))
    return findings


def test_clean_case_no_banned_tokens():
    findings = scan_paths(migrated_target_paths())
    assert findings == [], f"Found banned tokens in migrated targets: {findings}"


def test_detection_case_finds_banned_tokens(tmp_path: Path) -> None:
    temp = tmp_path / "migrate_sample.py"
    _ = temp.write_text(
        ("def migrate(v):\n    token = _input_streams\n    return token\n"),
        encoding="utf-8",
    )
    findings = scan_paths(migrated_target_paths() + [temp])
    assert any(p == str(temp) and tok == "_input_streams" for (p, tok) in findings), (
        f"Temp file not reported in findings: {findings}"
    )
