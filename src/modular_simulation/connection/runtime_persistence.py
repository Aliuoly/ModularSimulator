from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from math import isfinite
from modular_simulation.connection.network import (
    ConnectionNetwork,
    QueuedReconfigurationTransaction,
)
from modular_simulation.connection.runtime import ConnectionRuntimeOrchestrator

RUNTIME_SNAPSHOT_SCHEMA_MARKER = "connection.runtime.snapshot"
RUNTIME_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _ValidatedQueuedRequest:
    request_id: str
    idempotency_key: str | None
    mutations: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ValidatedRuntimeSnapshot:
    simulation_time_s: float
    macro_step_index: int
    graph_revision: str | None
    next_edge_index: int
    next_reconfiguration_index: int
    next_revision_index: int
    queued_requests: tuple[_ValidatedQueuedRequest, ...]
    committed_idempotency_keys: tuple[str, ...]
    process_runtime_state: dict[str, object]


def save_runtime_snapshot_contract(
    *,
    runtime: ConnectionRuntimeOrchestrator,
    network: ConnectionNetwork,
    process_runtime_state: Mapping[str, object],
) -> dict[str, object]:
    compiled = network.compiled
    graph_revision = None if compiled is None else str(compiled.graph_revision)
    queue_payload: list[dict[str, object]] = []
    for queued in network.queued_reconfigurations:
        queue_payload.append(
            {
                "request_id": queued.request_id,
                "idempotency_key": queued.idempotency_key,
                "mutations": [
                    _normalize_mapping_payload(mutation) for mutation in queued.mutations
                ],
            }
        )

    return {
        "schema_marker": RUNTIME_SNAPSHOT_SCHEMA_MARKER,
        "schema_version": RUNTIME_SNAPSHOT_SCHEMA_VERSION,
        "runtime": {
            "simulation_time_s": runtime.simulation_time_s,
            "macro_step_index": runtime.macro_step_index,
            "macro_step_active": runtime.macro_step_active,
        },
        "process_runtime_state": _normalize_mapping_payload(process_runtime_state),
        "connection_runtime_state": {
            "graph_revision": graph_revision,
            "next_edge_index": network.next_edge_index,
            "next_reconfiguration_index": network.next_reconfiguration_index,
            "next_revision_index": network.next_revision_index,
            "queued_reconfigurations": queue_payload,
            "queued_request_ids": [queued["request_id"] for queued in queue_payload],
            "committed_idempotency_keys": list(runtime.committed_idempotency_keys),
        },
    }


def validate_runtime_snapshot_contract(snapshot: Mapping[str, object]) -> ValidatedRuntimeSnapshot:
    _require_exact_keys(
        "snapshot payload",
        snapshot,
        (
            "schema_marker",
            "schema_version",
            "runtime",
            "process_runtime_state",
            "connection_runtime_state",
        ),
    )
    schema_marker = snapshot["schema_marker"]
    if schema_marker != RUNTIME_SNAPSHOT_SCHEMA_MARKER:
        raise ValueError(
            "invalid runtime snapshot payload: field 'schema_marker' does not match expected marker"
        )

    schema_version = snapshot["schema_version"]
    if schema_version != RUNTIME_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "invalid runtime snapshot payload: field 'schema_version' does not match expected version"
        )

    runtime_section = _require_mapping(snapshot["runtime"], "runtime")
    _require_exact_keys(
        "runtime",
        runtime_section,
        ("simulation_time_s", "macro_step_index", "macro_step_active"),
    )
    simulation_time_s = _require_non_negative_finite_number(
        runtime_section["simulation_time_s"],
        "runtime.simulation_time_s",
    )
    macro_step_index = _require_non_negative_int(
        runtime_section["macro_step_index"],
        "runtime.macro_step_index",
    )
    macro_step_active = runtime_section["macro_step_active"]
    if macro_step_active is not False:
        raise ValueError(
            "invalid runtime snapshot payload: field 'runtime.macro_step_active' must be false at barrier"
        )

    process_runtime_state = _require_mapping(
        snapshot["process_runtime_state"], "process_runtime_state"
    )
    normalized_process_state = _normalize_mapping_payload(process_runtime_state)

    connection_section = _require_mapping(
        snapshot["connection_runtime_state"],
        "connection_runtime_state",
    )
    _require_exact_keys(
        "connection_runtime_state",
        connection_section,
        (
            "graph_revision",
            "next_edge_index",
            "next_reconfiguration_index",
            "next_revision_index",
            "queued_reconfigurations",
            "queued_request_ids",
            "committed_idempotency_keys",
        ),
    )
    graph_revision = connection_section["graph_revision"]
    if graph_revision is not None:
        graph_revision = _require_non_empty_string(
            graph_revision,
            "connection_runtime_state.graph_revision",
        )

    next_edge_index = _require_non_negative_int(
        connection_section["next_edge_index"],
        "connection_runtime_state.next_edge_index",
    )
    next_reconfiguration_index = _require_non_negative_int(
        connection_section["next_reconfiguration_index"],
        "connection_runtime_state.next_reconfiguration_index",
    )
    next_revision_index = _require_non_negative_int(
        connection_section["next_revision_index"],
        "connection_runtime_state.next_revision_index",
    )

    committed_idempotency_keys = _require_string_sequence(
        connection_section["committed_idempotency_keys"],
        "connection_runtime_state.committed_idempotency_keys",
    )

    queued_reconfiguration_values = connection_section["queued_reconfigurations"]
    if isinstance(queued_reconfiguration_values, str) or not isinstance(
        queued_reconfiguration_values, Sequence
    ):
        raise ValueError(
            "invalid runtime snapshot payload: field 'connection_runtime_state.queued_reconfigurations' must be a sequence"
        )

    validated_requests: list[_ValidatedQueuedRequest] = []
    ordered_request_ids: list[str] = []
    for request_index, request in enumerate(queued_reconfiguration_values, start=1):
        request_mapping = _require_mapping(
            request,
            "connection_runtime_state.queued_reconfigurations" + f"[{request_index}]",
        )
        _require_exact_keys(
            f"connection_runtime_state.queued_reconfigurations[{request_index}]",
            request_mapping,
            ("request_id", "idempotency_key", "mutations"),
        )
        request_id = _require_non_empty_string(
            request_mapping["request_id"],
            "connection_runtime_state.queued_reconfigurations" + f"[{request_index}].request_id",
        )
        idempotency_key = request_mapping["idempotency_key"]
        if idempotency_key is not None:
            idempotency_key = _require_non_empty_string(
                idempotency_key,
                "connection_runtime_state.queued_reconfigurations"
                + f"[{request_index}].idempotency_key",
            )

        mutations_value = request_mapping["mutations"]
        if isinstance(mutations_value, str) or not isinstance(mutations_value, Sequence):
            raise ValueError(
                "invalid runtime snapshot payload: field "
                + "'connection_runtime_state.queued_reconfigurations"
                + f"[{request_index}].mutations' must be a sequence"
            )
        normalized_mutations: list[dict[str, object]] = []
        for mutation_index, mutation in enumerate(mutations_value, start=1):
            mutation_mapping = _require_mapping(
                mutation,
                "connection_runtime_state.queued_reconfigurations"
                + f"[{request_index}].mutations[{mutation_index}]",
            )
            operation = mutation_mapping.get("operation")
            _ = _require_non_empty_string(
                operation,
                "connection_runtime_state.queued_reconfigurations"
                + f"[{request_index}].mutations[{mutation_index}].operation",
            )
            normalized_mutations.append(_normalize_mapping_payload(mutation_mapping))

        validated_requests.append(
            _ValidatedQueuedRequest(
                request_id=request_id,
                idempotency_key=idempotency_key,
                mutations=tuple(normalized_mutations),
            )
        )
        ordered_request_ids.append(request_id)

    queued_request_ids = _require_string_sequence(
        connection_section["queued_request_ids"],
        "connection_runtime_state.queued_request_ids",
    )
    if tuple(ordered_request_ids) != queued_request_ids:
        raise ValueError(
            "invalid runtime snapshot payload: field 'connection_runtime_state.queued_request_ids' "
            + "must match queued_reconfigurations ordering"
        )

    return ValidatedRuntimeSnapshot(
        simulation_time_s=simulation_time_s,
        macro_step_index=macro_step_index,
        graph_revision=graph_revision,
        next_edge_index=next_edge_index,
        next_reconfiguration_index=next_reconfiguration_index,
        next_revision_index=next_revision_index,
        queued_requests=tuple(validated_requests),
        committed_idempotency_keys=committed_idempotency_keys,
        process_runtime_state=normalized_process_state,
    )


def restore_runtime_snapshot_contract(
    *,
    runtime: ConnectionRuntimeOrchestrator,
    network: ConnectionNetwork,
    process_runtime_state_sink: MutableMapping[str, object],
    snapshot: Mapping[str, object],
    process_runtime_validator: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
) -> None:
    if runtime.macro_step_active:
        raise RuntimeError(
            "runtime snapshot restore is unavailable during active macro-step execution"
        )

    validated_snapshot = validate_runtime_snapshot_contract(snapshot)
    current_graph_revision = None if network.compiled is None else network.compiled.graph_revision
    if validated_snapshot.graph_revision != current_graph_revision:
        raise ValueError(
            "invalid runtime snapshot payload: field 'connection_runtime_state.graph_revision' "
            + "must match currently active compiled graph revision"
        )

    normalized_process_state = validated_snapshot.process_runtime_state
    if process_runtime_validator is not None:
        normalized_process_state = _normalize_mapping_payload(
            process_runtime_validator(validated_snapshot.process_runtime_state)
        )

    runtime_backup = (
        runtime.simulation_time_s,
        runtime.macro_step_index,
        runtime.macro_step_active,
        runtime.committed_idempotency_keys,
    )
    queue_backup = tuple(network.queued_reconfigurations)
    edge_index_backup = network.next_edge_index
    reconfiguration_index_backup = network.next_reconfiguration_index
    revision_index_backup = network.next_revision_index
    process_state_backup = dict(process_runtime_state_sink)

    staged_queue = [
        QueuedReconfigurationTransaction(
            request_id=item.request_id,
            idempotency_key=item.idempotency_key,
            mutations=item.mutations,
        )
        for item in validated_snapshot.queued_requests
    ]

    try:
        runtime.set_runtime_state(
            simulation_time_s=validated_snapshot.simulation_time_s,
            macro_step_index=validated_snapshot.macro_step_index,
            macro_step_active=False,
            committed_idempotency_keys=validated_snapshot.committed_idempotency_keys,
        )

        network.set_reconfiguration_counters(
            next_edge_index=validated_snapshot.next_edge_index,
            next_reconfiguration_index=validated_snapshot.next_reconfiguration_index,
            next_revision_index=validated_snapshot.next_revision_index,
        )
        network.replace_queued_reconfigurations(staged_queue)

        process_runtime_state_sink.clear()
        process_runtime_state_sink.update(normalized_process_state)
    except Exception:
        runtime.set_runtime_state(
            simulation_time_s=runtime_backup[0],
            macro_step_index=runtime_backup[1],
            macro_step_active=runtime_backup[2],
            committed_idempotency_keys=runtime_backup[3],
        )

        network.replace_queued_reconfigurations(queue_backup)
        network.set_reconfiguration_counters(
            next_edge_index=edge_index_backup,
            next_reconfiguration_index=reconfiguration_index_backup,
            next_revision_index=revision_index_backup,
        )

        process_runtime_state_sink.clear()
        process_runtime_state_sink.update(process_state_backup)
        raise


def _require_exact_keys(
    context: str,
    payload: Mapping[str, object],
    expected_keys: tuple[str, ...],
) -> None:
    actual_keys = tuple(payload.keys())
    if set(actual_keys) != set(expected_keys):
        raise ValueError(
            f"invalid runtime snapshot payload: field '{context}' keys must be exactly {expected_keys!r}"
        )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"invalid runtime snapshot payload: field '{field_name}' must be a mapping"
        )
    return {str(key): field_value for key, field_value in value.items()}


def _require_non_negative_finite_number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"invalid runtime snapshot payload: field '{field_name}' must be a finite non-negative number"
        )
    normalized = float(value)
    if normalized < 0.0 or not isfinite(normalized):
        raise ValueError(
            f"invalid runtime snapshot payload: field '{field_name}' must be a finite non-negative number"
        )
    return normalized


def _require_non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(
            f"invalid runtime snapshot payload: field '{field_name}' must be a non-negative integer"
        )
    return value


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"invalid runtime snapshot payload: field '{field_name}' must be a non-empty string"
        )
    return value.strip()


def _require_string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(
            f"invalid runtime snapshot payload: field '{field_name}' must be a sequence of strings"
        )
    normalized_values: list[str] = []
    for item in value:
        normalized_values.append(_require_non_empty_string(item, field_name))
    return tuple(normalized_values)


def _normalize_mapping_payload(payload: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key in sorted(payload.keys()):
        value = payload[key]
        normalized_key = str(key)
        if isinstance(value, Mapping):
            normalized[normalized_key] = _normalize_mapping_payload(value)
            continue
        if isinstance(value, str):
            normalized[normalized_key] = value
            continue
        if isinstance(value, Sequence):
            normalized[normalized_key] = _normalize_sequence_payload(value)
            continue
        normalized[normalized_key] = value
    return normalized


def _normalize_sequence_payload(payload: Sequence[object]) -> list[object]:
    normalized: list[object] = []
    for value in payload:
        if isinstance(value, Mapping):
            normalized.append(_normalize_mapping_payload(value))
            continue
        if isinstance(value, str):
            normalized.append(value)
            continue
        if isinstance(value, Sequence):
            normalized.append(_normalize_sequence_payload(value))
            continue
        normalized.append(value)
    return normalized


__all__ = [
    "RUNTIME_SNAPSHOT_SCHEMA_MARKER",
    "RUNTIME_SNAPSHOT_SCHEMA_VERSION",
    "ValidatedRuntimeSnapshot",
    "restore_runtime_snapshot_contract",
    "save_runtime_snapshot_contract",
    "validate_runtime_snapshot_contract",
]
