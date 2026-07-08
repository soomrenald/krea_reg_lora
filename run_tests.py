from __future__ import annotations

import inspect

import tests.test_core as test_core


def main() -> None:
    failures: list[str] = []
    for name, fn in sorted(vars(test_core).items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                failures.append(name)
                print(f"FAIL {name}: {exc}")
                trace = inspect.trace()
                if trace and trace[-1].code_context:
                    print(trace[-1].code_context[0].strip())
    if failures:
        raise SystemExit(f"{len(failures)} tests failed: {', '.join(failures)}")
    print("All tests passed")


if __name__ == "__main__":
    main()
