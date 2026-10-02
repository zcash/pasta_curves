"""Checking the hand-written proofs against the generated skeletons.

A proof file interleaves generated skeletons with hand-written annotation blocks, delimited by
`-- BEGIN <label>` and `-- END <label>` lines. Once the blocks are removed (and blank lines
dropped), each skeleton the file is registered for must appear verbatim exactly once, starting at
its `-- generated skeleton for `<name>`:` marker.
"""

import re
import sys
from pathlib import Path

from .skeleton import skeleton


def _strip_annotations(path, text):
    """Remove checked annotation blocks and reject malformed marker structure."""
    remaining = []
    active = None
    ok = True
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^\s*-- (BEGIN|END)(?:\s+(.*?))?\s*$", line)
        if not marker:
            if active is None:
                remaining.append(line)
            continue
        kind, label = marker.group(1), marker.group(2) or ""
        if kind == "BEGIN":
            if active is not None:
                print(f"{path}:{line_number}: nested BEGIN inside `{active[0]}`", file=sys.stderr)
                ok = False
            else:
                active = (label, line_number)
        elif active is None:
            print(f"{path}:{line_number}: END without BEGIN", file=sys.stderr)
            ok = False
        else:
            if label != active[0]:
                print(
                    f"{path}:{line_number}: END `{label}` does not match BEGIN `{active[0]}` "
                    f"at line {active[1]}",
                    file=sys.stderr,
                )
                ok = False
            active = None
    if active is not None:
        print(f"{path}:{active[1]}: unterminated BEGIN `{active[0]}`", file=sys.stderr)
        ok = False
    return ok, [line for line in remaining if line.strip()]


def check_spec(path, routines):
    """Require exactly one current skeleton for every routine in this file's manifest."""
    path = Path(path)
    try:
        source = path.read_text()
    except OSError as error:
        print(f"{path}: cannot read proof file: {error.strerror}", file=sys.stderr)
        return False
    ok, remaining = _strip_annotations(path, source)
    expected_names = {routine.name for routine in routines}
    markers = [
        (index, match.group(1))
        for index, line in enumerate(remaining)
        if (match := re.match(r"^\s*-- generated skeleton for `([^`]+)`:", line))
    ]
    unexpected = sorted({name for _, name in markers if name not in expected_names})
    for name in unexpected:
        print(f"{path}: unexpected generated skeleton `{name}`", file=sys.stderr)
        ok = False
    for routine in routines:
        generated = [line for line in skeleton(routine) if line.strip()]
        starts = [index for index, name in markers if name == routine.name]
        if len(starts) != 1:
            print(
                f"{path}: expected one skeleton of {routine.name}, found {len(starts)}",
                file=sys.stderr,
            )
            ok = False
            continue
        start = starts[0]
        found = remaining[start : start + len(generated)]
        if found != generated:
            for index, expected in enumerate(generated):
                actual = found[index] if index < len(found) else "<eof>"
                if actual != expected:
                    print(
                        f"{path}: skeleton of {routine.name} diverges at skeleton line {index}:",
                        file=sys.stderr,
                    )
                    print(f"  expected: {expected}", file=sys.stderr)
                    print(f"  found:    {actual}", file=sys.stderr)
                    break
            ok = False
    return ok


def find_routine(specification, available):
    """Resolve ``ARCH:NAME`` to its routine."""
    if ":" not in specification:
        raise ValueError(f"expected ARCH:NAME, not {specification}")
    architecture, name = specification.split(":", 1)
    manifests = available
    if architecture not in manifests:
        raise ValueError(f"unknown architecture {architecture}")
    matches = [routine for routine in manifests[architecture] if routine.name == name]
    if len(matches) != 1:
        raise ValueError(f"no routine {architecture}:{name}")
    return matches[0]


def parse_spec_manifest(specification, manifest, available, root):
    """Resolve one explicitly registered proof file to its required routine skeletons."""
    requested_architecture = None
    path_text = specification
    if ":" in specification:
        requested_architecture, path_text = specification.split(":", 1)
    path = Path(path_text)
    try:
        key = path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"spec path is outside the repository: {path}") from error
    if key not in manifest:
        raise ValueError(f"unregistered proof skeleton manifest: {key}")
    architecture, routine_names = manifest[key]
    if requested_architecture is not None and requested_architecture != architecture:
        raise ValueError(f"manifest {key} belongs to {architecture}, not {requested_architecture}")
    if architecture not in available:
        raise ValueError(f"manifest {key} names unknown architecture {architecture}")
    routines = available[architecture]
    if routine_names is None:
        selected = routines
    else:
        by_name = {routine.name: routine for routine in routines}
        missing = [name for name in routine_names if name not in by_name]
        if missing:
            raise ValueError(f"manifest {key} names missing routines: {', '.join(missing)}")
        selected = [by_name[name] for name in routine_names]
    return path, selected


def check_specs(manifest, unproved_routines, available, root, strict=False):
    """Check every registered file and account for every generated routine exactly once.

    Explicitly unproved routines are reported, never counted as checked skeletons. Strict
    mode rejects them as well. Kernel checking remains the responsibility of the Lean build.
    """
    inventory = {
        (arch, routine.name) for arch, routines in available.items() for routine in routines
    }
    covered, unproved = {}, set()
    ok = True
    for filename, (arch, names) in manifest.items():
        if arch not in available:
            print(f"{filename}: unknown architecture {arch}", file=sys.stderr)
            ok = False
            continue
        by_name = {routine.name: routine for routine in available[arch]}
        selected = tuple(by_name) if names is None else names
        routines = []
        for name in selected:
            key = (arch, name)
            if key not in inventory:
                print(f"{filename}: unknown routine {arch}:{name}", file=sys.stderr)
                ok = False
                continue
            if key in covered:
                print(
                    f"{arch}:{name}: duplicate coverage in {covered[key]} and {filename}",
                    file=sys.stderr,
                )
                ok = False
            covered[key] = filename
            routines.append(by_name[name])
        if not check_spec(root / filename, routines):
            ok = False
    for arch, names in unproved_routines.items():
        if arch not in available:
            print(f"unproved list: unknown architecture {arch}", file=sys.stderr)
            ok = False
        for name in names:
            key = (arch, name)
            if key not in inventory:
                print(f"unproved list: unknown routine {arch}:{name}", file=sys.stderr)
                ok = False
            if key in unproved:
                print(f"unproved list: duplicate routine {arch}:{name}", file=sys.stderr)
                ok = False
            unproved.add(key)
    for arch, name in sorted(set(covered) & unproved):
        print(f"{arch}:{name}: both covered and explicitly unproved", file=sys.stderr)
        ok = False
    for arch, name in sorted(inventory - set(covered) - unproved):
        print(f"{arch}:{name}: no Spec coverage or explicit unproved entry", file=sys.stderr)
        ok = False
    for arch, name in sorted(unproved & inventory):
        print(f"unproved: {arch}:{name}")
    if strict and unproved:
        print(
            "strict Spec coverage requires every routine to have a checked skeleton",
            file=sys.stderr,
        )
        ok = False
    print(
        f"Spec coverage: {len(covered)} registered, {len(unproved & inventory)} unproved; "
        f"{'checks passed' if ok else 'FAILED'}"
    )
    return ok
