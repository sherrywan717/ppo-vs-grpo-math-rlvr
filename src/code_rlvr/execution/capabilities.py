"""Detect execution isolation without running untrusted code."""

import json
import shutil
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CapabilityReport:
    docker: bool
    podman: bool
    nsjail: bool
    firejail: bool
    bubblewrap: bool
    safe_backend: str | None

    @property
    def can_execute_untrusted(self) -> bool:
        return self.safe_backend is not None


def detect_capabilities() -> CapabilityReport:
    available = {name: shutil.which(name) is not None for name in (
        "docker", "podman", "nsjail", "firejail", "bwrap"
    )}
    # Presence alone is insufficient. No backend is trusted until a dedicated
    # adapter verifies namespace, syscall, filesystem, network, and resource limits.
    return CapabilityReport(
        docker=available["docker"],
        podman=available["podman"],
        nsjail=available["nsjail"],
        firejail=available["firejail"],
        bubblewrap=available["bwrap"],
        safe_backend=None,
    )


def main() -> int:
    report = detect_capabilities()
    print(json.dumps(asdict(report) | {"can_execute_untrusted": report.can_execute_untrusted}, indent=2))
    return 0 if report.can_execute_untrusted else 2


if __name__ == "__main__":
    raise SystemExit(main())

