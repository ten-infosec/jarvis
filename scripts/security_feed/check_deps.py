"""Check this repo's pinned Python dependencies against OSV (known vulnerabilities and malicious-package reports).

Usage:  python scripts/security_feed/check_deps.py [requirements.txt ...]
Default files: requirements.txt, requirements-cloud.txt, requirements-dev.txt at the repo root.
Exit code: 0 = nothing found, 1 = advisories found, 2 = lookup failed.
Only package names and versions are sent to OSV (public data); nothing is installed or executed.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402  (reuses the allowlisted fetch helper)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FILES = ["requirements.txt", "requirements-cloud.txt", "requirements-dev.txt"]
OSV_BATCH = "https://api.osv.dev/v1/querybatch"
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)==([^\s;#]+)")


# ===== Parse =====
def read_pins(path):
    """Returns ([(name, version)], [unpinned lines]); comments and blank lines are skipped."""
    pins, loose = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        m = PIN.match(line)
        (pins if m else loose).append(m.groups() if m else line)
    return pins, loose


# ===== Lookup =====
def query(pins, fetch=collect.fetch):
    """One batch request for all packages; returns {(name, version): [vuln ids]}."""
    body = {"queries": [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in pins]}
    hosts = {"api.osv.dev"}
    results = json.loads(fetch(OSV_BATCH, hosts, body)).get("results", [])
    # OSV answers in the same order as the queries
    return {pin: [v["id"] for v in r.get("vulns", [])] for pin, r in zip(pins, results)}


# ===== Main =====
def main(argv):
    files = [Path(a) for a in argv] or [ROOT / f for f in DEFAULT_FILES]
    status = 0
    for f in files:
        pins, loose = read_pins(f)
        print(f"== {f.name}: {len(pins)} pinned packages")
        for line in loose:
            print(f"   not checked (unpinned): {line}")
        try:
            found = query(pins)
        except Exception as ex:  # network/format problem: report, do not claim "clean"
            print(f"   LOOKUP FAILED: {ex!r}")
            status = max(status, 2)
            continue
        for (name, ver), ids in found.items():
            if ids:
                status = max(status, 1)
                kind = "MALICIOUS" if any(i.startswith("MAL-") for i in ids) else "vulnerable"
                print(f"   {kind}: {name}=={ver}  " + ", ".join(f"https://osv.dev/vulnerability/{i}" for i in ids))
        if not any(found.values()):
            print("   no known advisories")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
