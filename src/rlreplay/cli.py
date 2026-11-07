"""Local evidence inspection and explicitly selected replay operations."""

import argparse
import json
import sys

from rlreplay import __version__
from rlreplay.artifacts import read
from rlreplay.contracts.grpo import check
from rlreplay.replay import diff, export_test, run_verl, semantic_replay

EXIT_CODES = {
    "PASS": 0,
    "CONTRACT_VIOLATION": 1,
    "SUSPECTED_MISMATCH": 2,
    "INCONCLUSIVE": 3,
}


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _parse_args(argv):
    parser = _ArgumentParser(prog="rlreplay")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "check", "replay", "diff", "export-test"):
        sub = commands.add_parser(command)
        sub.add_argument("capsule")
        if command == "replay":
            sub.add_argument(
                "--runner", choices=("semantic", "verl", "trl"), default="semantic"
            )
            sub.add_argument("--allow-revision-change", action="store_true")
            sub.add_argument(
                "--trainer-factory", help="explicit local module:function for TRL"
            )
        elif command == "diff":
            sub.add_argument("other")
        elif command == "export-test":
            sub.add_argument("directory")
            sub.add_argument("--runner", choices=("verl", "trl"), default="verl")
            sub.add_argument(
                "--trainer-factory", help="explicit local module:function for TRL"
            )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        args = _parse_args(argv)
        if args.command in ("replay", "export-test") and args.runner == "trl":
            from rlreplay.trl_replay import (
                export_trl_test,
                read_trl,
                run_trl,
                trainer_from_factory,
            )

            if args.command == "replay":
                bundle = read_trl(args.capsule)
                trainer = trainer_from_factory(args.trainer_factory)
                output = run_trl(
                    bundle, trainer, allow_revision_change=args.allow_revision_change
                )
            else:
                output = {
                    "test": str(
                        export_trl_test(
                            args.capsule,
                            args.directory,
                            trainer_factory=args.trainer_factory,
                        )
                    )
                }
            code = (
                EXIT_CODES[output["finding"]["verdict"]] if "finding" in output else 0
            )
            print(json.dumps(output, indent=2, allow_nan=False))
            return code
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
        code = EXIT_CODES[output["finding"]["verdict"]] if "finding" in output else 0
        print(json.dumps(output, indent=2, allow_nan=False))
        return code
    except Exception as error:  # noqa: BLE001 - terminal boundary for ordinary failures
        print(
            json.dumps(
                {"state": "ERROR", "reason": str(error), "type": type(error).__name__}
            ),
            file=sys.stderr,
        )
        return 4
