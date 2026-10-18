"""The ``mf`` command: one entry point for every workflow."""

import argparse
from types import ModuleType

from machine_failure.cli import add_verbosity_options, command, configure_logging
from machine_failure.data import download
from machine_failure.experiments import (
    comparison,
    final_evaluation,
    final_model,
    interpretation,
    predict,
    thresholds,
    tune,
)

# In the order of the workflow. Each module has HELP, add_arguments and run.
COMMANDS: dict[str, ModuleType] = {
    "download": download,
    "compare": comparison,
    "tune": tune,
    "thresholds": thresholds,
    "train": final_model,
    "evaluate": final_evaluation,
    "predict": predict,
    "interpret": interpretation,
}


def build_parser() -> argparse.ArgumentParser:
    """The ``mf`` parser, with one subcommand per workflow."""
    parser = argparse.ArgumentParser(
        prog="mf",
        description="Machine failure classification: one command per workflow. "
        "Run 'mf COMMAND --help' for the options of a command; every command "
        "accepts -v/--verbose and -q/--quiet.",
    )
    subcommands = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND"
    )
    common = argparse.ArgumentParser(add_help=False)
    add_verbosity_options(common)
    for name, module in COMMANDS.items():
        subparser = subcommands.add_parser(
            name,
            parents=[common],
            help=module.HELP,
            description=module.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(subparser)
        subparser.set_defaults(run=module.run)
    return parser


@command
def main(argv: list[str] | None = None) -> None:
    """Run the ``mf`` subcommand given in ``argv`` (default: the command line)."""
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet)
    args.run(args)
