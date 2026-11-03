"""Standard-library-only worker. It is mounted read-only into the provider container."""

import contextlib
import hashlib
import importlib.util
import json
import sys


def main():
    spec = importlib.util.spec_from_file_location("domain_provider", "/app/provider.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    if not callable(getattr(module, "provide", None)):
        raise ValueError("missing provide function")
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin.buffer:
        try:
            if len(line) > 64 * 1024 * 1024:
                raise ValueError("request exceeds 64 MB")
            request = json.loads(line)
            overrides = []
            for row in request["rows"]:
                key = json.dumps(
                    [request["seed"], request["kind"], request["cohort"], row["id"]],
                    separators=(",", ":"),
                )
                seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
                with contextlib.redirect_stdout(sys.stderr):
                    result = module.provide(request["kind"], request["cohort"], [row], seed)
                if (
                    not isinstance(result, list)
                    or len(result) != 1
                    or not isinstance(result[0], dict)
                ):
                    raise ValueError("provide must return one property dictionary per input row")
                overrides.extend(result)
            print(json.dumps({"overrides": overrides}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"error": f"{type(error).__name__}: {str(error)[:1000]}"}), flush=True)


if __name__ == "__main__":
    main()
