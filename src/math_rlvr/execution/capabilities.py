"""LEGACY/OUT-OF-SCOPE: code execution is retained for history and must not be used."""

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ToolCapability:
    present: bool
    usable: bool
    detail: str


@dataclass(frozen=True)
class CapabilityReport:
    bubblewrap: ToolCapability
    nsjail: ToolCapability
    firejail: ToolCapability
    unshare: ToolCapability
    safe_backend: str | None

    @property
    def can_execute_untrusted(self) -> bool:
        return self.safe_backend is not None


def _probe(name: str, arguments: list[str]) -> ToolCapability:
    executable = shutil.which(name)
    if executable is None:
        return ToolCapability(False, False, "executable not found")
    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return ToolCapability(True, False, f"probe failed: {error}")
    detail = (result.stderr or result.stdout).strip().splitlines()
    summary = detail[-1][:200] if detail else f"exit code {result.returncode}"
    return ToolCapability(True, result.returncode == 0, summary)


def detect_capabilities() -> CapabilityReport:
    # Probes execute only trusted no-op commands. They never execute generated code.
    bubblewrap = _probe(
        "bwrap", ["--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "/bin/true"]
    )
    nsjail = _probe("nsjail", ["--mode", "o", "--", "/bin/true"])
    firejail = _probe("firejail", ["--quiet", "--noprofile", "/bin/true"])
    unshare = _probe("unshare", ["--user", "--map-root-user", "/bin/true"])
    # Presence alone is insufficient. No backend is trusted until a dedicated
    # adapter verifies namespace, syscall, filesystem, network, and resource limits.
    return CapabilityReport(
        bubblewrap=bubblewrap,
        nsjail=nsjail,
        firejail=firejail,
        unshare=unshare,
        safe_backend=None,
    )


def main() -> int:
    report = detect_capabilities()
    payload = asdict(report) | {"can_execute_untrusted": report.can_execute_untrusted}
    print(json.dumps(payload, indent=2))
    return 0 if report.can_execute_untrusted else 2


if __name__ == "__main__":
    raise SystemExit(main())
