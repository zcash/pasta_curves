#!/usr/bin/env python3
"""Generate the Lean transcriptions of the crate's inline assembly, and check the proofs' skeletons.

Reads the inline `asm!` blocks in `src/asm/aarch64.rs` and `src/asm/x86_64.rs`, and writes

- `lean/PastaCurves/<Architecture>/Transcription.lean`: each block as a Lean definition over
  its instruction semantics, one `let` per instruction result, in the block's order, with the
  instruction as a trailing comment;
- `lean/PastaCurves/Vectors.lean`, `KnownAnswers.lean`, and `FieldTypes.lean`: the hardware
  reference vectors, the backend tests' known answers, and the field types' constants, as data
  and examples checked by kernel evaluation.

The work is done by the compiler in `asm2lean/` (see its `__init__.py` for its stages), on the
configuration in `pasta/`: which blocks, how their repeated rounds fold, how the proofs name
things, and which proof file proves which routine. This file is the command line.

Run from anywhere in the repository:

    python3 lean/scripts/gen.py                         # write every generated file
    python3 lean/scripts/gen.py --check                 # compare them without writing
    python3 lean/scripts/gen.py --skeleton ARCH:NAME    # print one proof skeleton
    python3 lean/scripts/gen.py --check-spec [ARCH:]FILE
    python3 lean/scripts/gen.py --check-specs [--strict]

`lean/scripts/check.sh` runs `--check`, `--check-specs`, and the generator's tests.
Python 3.10+; stdlib only.
"""

import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asm2lean import specs
from asm2lean.skeleton import skeleton
from pasta import aarch64_blocks, data, x86_64_blocks
from pasta.paths import ROOT
from pasta.specs import SPEC_MANIFEST, UNPROVED_ROUTINES


def generated_outputs():
    """Every generated path and its expected contents, without writing anything."""
    return data.generated_outputs() + [
        (aarch64_blocks.OUTPUT, aarch64_blocks.text()),
        (x86_64_blocks.OUTPUT, x86_64_blocks.text()),
    ]


def architecture_routines():
    """The routines of each architecture's transcription, which the proofs are about."""
    return {"AArch64": aarch64_blocks.programs(), "X86_64": x86_64_blocks.programs()}


def find_routine(specification):
    return specs.find_routine(specification, architecture_routines())


def parse_spec_manifest(specification):
    return specs.parse_spec_manifest(specification, SPEC_MANIFEST, architecture_routines(), ROOT)


def check_specs(strict=False):
    return specs.check_specs(
        SPEC_MANIFEST, UNPROVED_ROUTINES, architecture_routines(), ROOT, strict=strict
    )


def check_output(path, expected):
    """Compare one generated file without writing it, printing a unified diff."""
    display = path.relative_to(ROOT)
    if not path.exists():
        print(f"{display} does not exist", file=sys.stderr)
        return False
    actual = path.read_text()
    if actual == expected:
        return True
    diff = difflib.unified_diff(
        actual.splitlines(True),
        expected.splitlines(True),
        fromfile=str(display),
        tofile="generated",
    )
    sys.stderr.writelines(diff)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="compare every output without writing")
    action.add_argument(
        "--check-spec",
        metavar="[ARCH:]FILE",
        help="check the architecture's registered proof skeletons",
    )
    action.add_argument("--skeleton", metavar="ARCH:NAME", help="print one proof skeleton")
    action.add_argument(
        "--check-specs",
        action="store_true",
        help="check all Spec files and account for every generated routine",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="with --check-specs, reject explicitly unproved routines",
    )
    args = parser.parse_args(argv)
    if args.strict and not args.check_specs:
        parser.error("--strict requires --check-specs")
    if args.check_specs:
        return 0 if check_specs(strict=args.strict) else 1

    if args.skeleton:
        try:
            routine = find_routine(args.skeleton)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        print("\n".join(skeleton(routine)))
        return 0
    if args.check_spec:
        try:
            path, routines = parse_spec_manifest(args.check_spec)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        ok = specs.check_spec(path, routines)
        print(f"{path}: skeletons {'current' if ok else 'STALE'}")
        return 0 if ok else 1

    outputs = generated_outputs()
    if args.check:
        # Every file is compared, so that each stale one prints its diff.
        checks = [check_output(path, expected) for path, expected in outputs]
        ok = all(checks)
        print(f"generated transcriptions: {'current' if ok else 'STALE'}")
        return 0 if ok else 1

    for path, expected in outputs:
        path.write_text(expected)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
