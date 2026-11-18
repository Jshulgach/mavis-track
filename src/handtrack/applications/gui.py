"""Generic GUI entrypoint that dispatches to the selected backend."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from ._backend_selection import resolve_backend

_BACKEND_MODULES = {
    "webcam": "handtrack.applications.webcam_gui",
    "optitrack": "handtrack.applications.optitrack_gui",
}


def _run_module_entrypoint(module_name: str, passthrough_args: list[str]) -> int:
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if main is None:
        raise RuntimeError(
            f"Module '{module_name}' does not expose a main() entrypoint."
        )

    previous_argv = sys.argv
    sys.argv = [previous_argv[0], *passthrough_args]
    try:
        result = main()
    finally:
        sys.argv = previous_argv

    return 0 if result is None else int(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the advanced hands dashboard for the selected backend."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "optitrack", "webcam"),
        default=None,
        help=(
            "camera backend to use; defaults to HANDTRACK_BACKEND if set, "
            "otherwise auto"
        ),
    )
    gui_mode = parser.add_mutually_exclusive_group()
    gui_mode.add_argument("--advanced-hands", dest="advanced_hands", action="store_true",
                          help="use the default multi-camera hands dashboard")
    gui_mode.add_argument("--studio", dest="advanced_hands", action="store_false",
                          help="open the single-camera/video Studio explicitly")
    parser.set_defaults(advanced_hands=True)
    args, passthrough = parser.parse_known_args(argv)

    selected_backend = ("webcam" if not args.advanced_hands and args.backend in (None, "auto")
                        else resolve_backend(args.backend))
    if not args.advanced_hands and selected_backend != "webcam":
        parser.error("Studio uses webcam/video sources; use --backend webcam or omit --studio")
    module_name = ("handtrack.applications.mediapipe_gui" if not args.advanced_hands
                   else _BACKEND_MODULES[selected_backend])
    return _run_module_entrypoint(module_name, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
