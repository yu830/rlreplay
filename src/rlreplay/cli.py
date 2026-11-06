"""Local evidence inspection and explicitly selected replay operations."""

import argparse
import json
import sys

from rlreplay.artifacts import read
from rlreplay.contracts.grpo import check
from rlreplay.replay import diff, export_test, run_verl, semantic_replay

EXIT_CODES = {
    "PASS": 0,
    "CONTRACT_VIOLATION": 1,
    "SUSPECTED_MISMATCH": 2,
    "INCONCLUSIVE": 3,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rlreplay")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "check", "replay", "diff", "export-test"):
        sub = commands.add_parser(command)
        sub.add_argument("capsule")
        if command == "replay":
            sub.add_argument(
                "--runner", choices=("semantic", "verl"), default="semantic"
            )
            sub.add_argument("--allow-revision-change", action="store_true")
        elif command == "diff":
            sub.add_argument("other")
        elif command == "export-test":
            sub.add_argument("directory")
    args = parser.parse_args(argv)
    try:
        evidence = read(args.capsule)
        if args.command == "inspect":
            output = {
                "profile": evidence.to_dict()["profile"],
                "provenance": evidence.provenance,
                "samples": len(evidence.samples),
                "complete_groups": evidence.complete_groups,
            }
        elif args.command == "check":
            output = {"finding": check(evidence).to_dict()}
        elif args.command == "replay":
            if args.runner == "verl":
                replayed = run_verl(
                    evidence, allow_revision_change=args.allow_revision_change
                )
                output = {
                    "runner": "verl",
                    "finding": check(replayed).to_dict(),
                    "comparison": diff(evidence, replayed),
                }
            else:
                output = semantic_replay(evidence)
        elif args.command == "diff":
            output = diff(evidence, read(args.other))
        else:
            output = {"test": str(export_test(args.capsule, args.directory))}
    except (OSError, ValueError, TypeError, ImportError) as error:
        print(json.dumps({"state": "ERROR", "reason": str(error)}), file=sys.stderr)
        return 4
    print(json.dumps(output, indent=2, allow_nan=False))
    return EXIT_CODES[output["finding"]["verdict"]] if "finding" in output else 0
