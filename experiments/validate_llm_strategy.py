"""Ask DeepSeek for one bounded hypothesis, then score it by local BLE replay."""

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from replay_public_ble import main as replay


PROMPT = (
    "Propose one testable change to LocAgent using only these validation metrics. "
    "The current adaptive rule switches to the 15-second window after one positive "
    "motion check if anchor geometry permits, otherwise it uses 45 seconds. "
    "Choose 2 to 5 consecutive positive checks to reduce false switches. A negative "
    "or missing check resets the count. Aim to reduce moving CEP90 while preserving "
    "stationary CEP90 and solve rate. Return only a JSON object, for example "
    '{"motion_confirmations": 2, "hypothesis": "Consecutive checks may reduce jitter."}'
)


def parse_proposal(value):
    if not isinstance(value, dict):
        raise ValueError("proposal must be a JSON object")
    count = value.get("motion_confirmations")
    hypothesis = value.get("hypothesis")
    if type(count) is not int or not 2 <= count <= 5:
        raise ValueError("motion_confirmations must be an integer from 2 to 5")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("proposal must include a nonempty hypothesis")
    return {"motion_confirmations": count, "hypothesis": hypothesis.strip()}


def propose(metrics, model, api_key, base_url):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(metrics)},
        ],
        "max_tokens": 4096,
    }
    if base_url.rstrip("/") == "https://api.deepseek.com":
        payload["response_format"] = {"type": "json_object"}
        payload["thinking"] = {"type": "disabled"}
    content = chat_completion(payload, api_key, base_url)
    try:
        content = content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = content.partition("\n")[2][:-3]
        return parse_proposal(json.loads(content))
    except ValueError as exc:
        raise ValueError(f"Invalid DeepSeek proposal: {exc}") from exc


def chat_completion(payload, api_key, base_url):
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LocAgent/0.1",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            reply = json.load(response)
    except HTTPError as exc:
        message = f"API returned HTTP {exc.code}"
        if exc.code == 403:
            message += " (gateway denied the request; check access with the gateway provider)"
        raise ValueError(message) from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"DeepSeek request failed: {exc}") from exc
    try:
        choice = reply["choices"][0]
        content = choice["message"].get("content")
        if isinstance(content, str) and content.strip():
            return content
        if choice.get("finish_reason") == "length":
            raise ValueError("API returned no assistant text before reaching the output limit")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ValueError("API response did not contain a Chat Completions message") from None
    raise ValueError("API returned an empty assistant message")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path, nargs="?")
    parser.add_argument("validation", type=Path, nargs="?")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--proposal", type=Path, help="Replay a saved proposal without calling the API")
    mode.add_argument("--check-api", action="store_true", help="Test the API without replaying BLE data")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables/llm_validation"))
    args = parser.parse_args()
    if not args.check_api and (args.calibration is None or args.validation is None):
        parser.error("calibration and validation files are required for replay")
    api_key = None
    if args.proposal is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            parser.error("set DEEPSEEK_API_KEY or pass --proposal")

    try:
        if args.check_api:
            chat_completion(
                {"model": args.model, "messages": [{"role": "user", "content": "Reply with OK."}]},
                api_key,
                args.base_url,
            )
            print(f"API returned an assistant message for {args.model}.")
            return
        if args.proposal is None:
            baseline = replay(args.calibration, args.validation, args.output_dir / "baseline.csv")
            proposal = propose(baseline, args.model, api_key, args.base_url)
            source = args.model
        else:
            proposal = parse_proposal(json.loads(args.proposal.read_text(encoding="utf-8-sig")))
            source = "local file"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "proposal.json").write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        metrics = replay(
            args.calibration, args.validation, args.output_dir / "decisions.csv",
            motion_confirmations=proposal["motion_confirmations"],
        )
        result = {"source": source, "proposal": proposal, "validation_metrics": metrics}
        (args.output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
