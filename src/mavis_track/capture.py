"""Client for the local MAVIS camera recording interface."""

import json
import math
import os
import socket
import sys
import time
from uuid import uuid4


def capture_request(action, *, port=5015, trial_name=None, shared_output_root=None, timeout=2.0):
    """Send one camera-only request to a local dashboard or headless capture service.

    A timeout leaves the outcome unknown. Recover by querying status using the
    same trial name before deciding whether to retry a start or stop.
    """
    if action not in ("status", "start", "stop"):
        raise ValueError("Expected status, start or stop")
    if action != "status" and not trial_name:
        raise ValueError("Start and stop require a trial name")
    if not 1 <= port <= 65535 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Use port 1–65535 and a positive finite timeout")
    request_id = uuid4().hex
    packet = {"mode": "camera", "action": action, "request_id": request_id}
    if trial_name is not None:
        packet["trial_name"] = trial_name
    if shared_output_root is not None:
        packet["shared_output_root"] = str(shared_output_root)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.connect(("127.0.0.1", port))
            client.send(json.dumps(packet).encode("utf-8"))
            deadline = time.monotonic() + timeout
            while (remaining := deadline - time.monotonic()) > 0:
                client.settimeout(remaining)
                response = json.loads(client.recv(8192))
                if not isinstance(response, dict) or response.get("request_id") != request_id:
                    continue
                if response.get("ok") is not True:
                    raise RuntimeError(response.get("error") or "MAVIS rejected the request")
                if "camera_ready" not in response:
                    raise RuntimeError("The running MAVIS instance lacks camera-only recording control. Upgrade it.")
                return response
    except (OSError, ValueError) as exc:
        raise ConnectionError(f"No usable MAVIS reply on port {port}. Check the listener and query status.") from exc
    raise ConnectionError(f"No matching MAVIS reply on port {port}. Query status before retrying.")


def add_arguments(parser):
    parser.add_argument("action", choices=("status", "start", "stop"))
    parser.add_argument("--port", type=int, help="loopback UDP port (MAVIS_CAPTURE_CONTROL_PORT or 5015)")
    parser.add_argument("--trial-name", help="name of the trial to start or stop")
    parser.add_argument("--shared-output-root", help="require this MAVIS output root for a shared trial")
    parser.add_argument("--timeout", type=float, default=2., help="reply timeout in seconds")


def run(args):
    try:
        port = args.port if args.port is not None else int(os.environ.get("MAVIS_CAPTURE_CONTROL_PORT", "5015"))
        result = capture_request(args.action, port=port, trial_name=args.trial_name,
                                 shared_output_root=args.shared_output_root, timeout=args.timeout)
    except (ConnectionError, RuntimeError, ValueError) as exc:
        print(f"MAVIS control: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
