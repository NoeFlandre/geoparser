"""Conventions shared by the scripts that take command-line arguments.

Each script describes its command line in ``build_parser()``, which reads and
writes nothing. ``main(argv=None)`` parses the arguments and returns the exit
status, and the module ends with ``raise SystemExit(main())``.

The exit statuses are:

- ``EXIT_OK`` (0): the command completed and every check passed.
- ``EXIT_FAILURE`` (1): the command ran and a check or gate failed.
- ``EXIT_ERROR`` (2): the input could not be read or validated. argparse also
  exits with 2 for a malformed command line.

Scripts that CI runs as plain files, such as ``python3 scripts/changelog.py``,
cannot import this module, so they spell out the same values.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2
