"""
core/path.py — Path configuration model.

A "path" in this project is a logical monitoring target: a named (host, port)
pair that the monitoring layer will repeatedly probe to measure health.

What a path is NOT
──────────────────
  - NOT a real VPN tunnel or WAN link (those belong to later stages)
  - NOT a network interface name (eth0, tun0, etc.)
  - NOT a routing table entry

What a path IS
──────────────
  - A human-readable name for a monitoring target  (e.g. "primary", "backup")
  - A host to connect to                           (e.g. "10.0.0.1", "127.0.0.1")
  - A TCP port to probe                            (e.g. 9001)

In a real SD-WAN system, each Path would map to a distinct WAN interface.
In this prototype, multiple paths may simply point to different local TCP server
ports. The monitoring code does not care — it only sees (host, port).

Why a dataclass?
  The path definition is pure configuration data. A frozen dataclass:
    - Self-documents its fields
    - Validates at construction time (via __post_init__)
    - Is immutable (prevents accidental mutation of configuration)
    - Is hashable (can be used as a dict key)

Dependency direction
────────────────────
  This module has NO imports from core/ and performs NO network I/O.
  It is a data-only module at the bottom of the dependency graph.
  Everything else can import it; it imports nothing from the project.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Path:
    """
    A logical monitoring path: a named TCP (host, port) target.

    Attributes
    ----------
    name:
        Human-readable label for this path.
        Used to identify measurement results.
        Must be a non-empty string.
        Examples: "primary", "backup", "lte", "mpls"

    host:
        Hostname or IPv4 address to probe.
        Examples: "10.0.0.1", "127.0.0.1", "edge-router.internal"

    port:
        TCP port to probe. Must be in the range 1–65535.
        (Port 0 is reserved and cannot be connected to.)

    Why frozen=True?
        Path objects are configuration. Mutating a path after it has been
        handed to the monitoring layer would create confusing, race-condition-
        prone behavior. Immutability makes configuration safe to pass around.

    Example
    -------
        primary = Path(name="primary", host="10.0.0.1", port=5201)
        backup  = Path(name="backup",  host="10.0.0.2", port=5201)
    """

    name: str
    host: str
    port: int

    def __post_init__(self) -> None:
        """
        Validate path configuration at construction time.

        Raises
        ------
        ValueError
            If name is empty, host is empty, or port is out of range.

        Why validate here and not in the monitoring layer?
          Configuration errors should be caught at the point of construction,
          not silently discovered during a monitoring run. Early validation
          gives a clear error message with the offending value.
        """
        if not self.name or not self.name.strip():
            raise ValueError(f"Path name must be a non-empty string, got {self.name!r}")
        if not self.host or not self.host.strip():
            raise ValueError(f"Path host must be a non-empty string, got {self.host!r}")
        if not (1 <= self.port <= 65535):
            raise ValueError(
                f"Path port must be in the range 1–65535, got {self.port!r}"
            )

    def __str__(self) -> str:
        return f"Path({self.name!r} → {self.host}:{self.port})"
