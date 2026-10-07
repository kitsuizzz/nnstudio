#!/usr/bin/env python3
"""The browser boundary: no NaN/±inf may reach a JSON response.

Python writes float('inf') as `Infinity` (not JSON), and JSON.parse refuses at
the minus sign of `-Infinity` — "No number after minus sign in JSON at
position …". Every HTTP response goes through nn_server.json_safe.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nn_server  # noqa: E402

FAILED = []


def check(name, fn):
    try:
        fn()
        print("  ok   " + name)
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        print("  FAIL " + name + " \u2014 " + str(e))


def nonfinite_becomes_null():
    out = nn_server.json_safe({
        "mean": float("-inf"), "min": float("nan"), "max": float("inf"),
        "ok": 1.5, "count": 3, "text": "fine",
        "peek": [float("-inf"), 2.0, None],
        "box": {"x": float("inf"), "y": -2.0},
        "list": (1.0, float("nan")),
    })
    assert out["mean"] is None and out["min"] is None and out["max"] is None, out
    assert out["ok"] == 1.5 and out["count"] == 3 and out["text"] == "fine", out
    assert out["peek"][0] is None and out["peek"][1] == 2.0 and out["peek"][2] is None
    assert out["box"] == {"x": None, "y": -2.0}, out["box"]
    assert out["list"] == [1.0, None], out["list"]
    text = json.dumps(out, allow_nan=False)     # strict: raises on any survivor
    assert "Infinity" not in text and "NaN" not in text, text


check("NaN and \u00b1inf become null, everything else rides through",
      nonfinite_becomes_null)

if FAILED:
    print("json_safe: %d failure(s): %s" % (len(FAILED), ", ".join(FAILED)))
    raise SystemExit(1)
print("json_safe: all assertions passed")
