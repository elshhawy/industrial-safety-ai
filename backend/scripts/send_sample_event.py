"""Sample event sender for testing AI-to-Backend integration."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send sample AI safety event to backend")
    parser.add_argument("--url", default="http://127.0.0.1:8010", help="Backend base URL")
    parser.add_argument(
        "--payload",
        default="ppe_violation_event.json",
        choices=[
            "normal_person_event.json",
            "ppe_violation_event.json",
            "danger_zone_violation_event.json",
            "zone_entry_event.json",
            "zone_exit_event.json",
        ],
        help="Example payload file to send",
    )
    parser.add_argument("--event-id", default=None, help="Custom event_id (default: auto-generated UUID)")
    parser.add_argument("--duplicate", action="store_true", help="Send the event twice to verify duplicate rejection")
    args = parser.parse_args()

    file_path = EXAMPLES_DIR / args.payload
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    event_id = args.event_id or f"evt_cli_{uuid4()}"
    payload["event_id"] = event_id

    endpoint = args.url.rstrip("/") + "/api/v1/events"
    print(f"--> Sending {args.payload} (event_id={event_id}) to {endpoint}")

    def send():
        req = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=10) as resp:
                print(f"<-- Status: {resp.status}")
                body = json.loads(resp.read().decode("utf-8"))
                print(f"<-- Response: {json.dumps(body, indent=2)}")
                return resp.status, body
        except HTTPError as exc:
            print(f"<-- Error HTTP {exc.code}")
            err_body = exc.read().decode("utf-8")
            print(f"<-- Detail: {err_body}")
            return exc.code, err_body

    send()

    if args.duplicate:
        print(f"\n--> Resending duplicate event_id={event_id} (expecting HTTP 409)...")
        status, _ = send()
        if status == 409:
            print("--> Success: Duplicate properly rejected with HTTP 409!")
        else:
            print(f"--> Error: Expected HTTP 409, got {status}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
