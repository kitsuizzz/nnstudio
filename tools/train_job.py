#!/usr/bin/env python3
"""Train a design — the training half of the agent-job family.

    tools/train_job.py --task <id> --design <design.json> [--checkpoint <dir>]
                        [--steps 0] [--name <run>]

It runs `nn_torch.DesignTrainer` and streams events as lines to stdout (the
wrapper tees them into the task log). `agents/<task>.cmd` is its remote
control: the server writes {"cmd": "save" | "detach"} and the loop obeys on
the next step. `agents/<task>.state.json` is the machine-readable progress
the training box polls."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="train")
    ap.add_argument("--design", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--steps", type=int, default=0, help="0 = until detach")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=0)
    a = ap.parse_args()

    import nn_torch

    with open(a.design, encoding="utf-8") as fh:
        canvas = json.load(fh)
    cfg = {"name": a.name or a.task, "checkpoints": os.path.join(ROOT, "checkpoints")}
    if a.batch:
        cfg["batch"] = a.batch
    if a.lr:
        cfg["lr"] = a.lr
    if a.save_every:
        cfg["save_every"] = a.save_every

    state_path = os.path.join(ROOT, "agents", a.task + ".state.json")
    cmd_path = os.path.join(ROOT, "agents", a.task + ".cmd")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    if os.path.isfile(cmd_path):
        os.remove(cmd_path)

    def write_state(**kw):
        kw.setdefault("step", 0)
        kw.setdefault("loss", None)
        kw.setdefault("saved_at", None)
        kw["task"] = a.task
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(kw, fh)
        os.replace(tmp, state_path)

    def command():
        try:
            with open(cmd_path, encoding="utf-8") as fh:
                cmd = json.load(fh).get("cmd")
            os.remove(cmd_path)
            return cmd
        except (OSError, ValueError):
            return None

    print("training %s — %s" % (a.task, a.design))
    t0 = time.time()
    try:
        tr = nn_torch.DesignTrainer(canvas, cfg, device=a.device,
                                    checkpoint=a.checkpoint or None)
    except Exception as e:  # noqa: BLE001
        write_state(error=str(e))
        print("training failed to start: %s" % e)
        return 1

    def on_event(ev):
        if ev.get("type") == "loss":
            write_state(step=ev["step"], loss=ev["loss"],
                        saved_at=ev.get("saved_at"),
                        rows=len(tr.rows), elapsed=time.time() - t0)
            print("step %d  loss %.4f" % (ev["step"], ev["loss"]))
        elif ev.get("type") == "checkpoint":
            write_state(step=tr.step,
                        loss=tr.history[-1] if tr.history else None,
                        saved_at=ev.get("path"),
                        rows=len(tr.rows), elapsed=time.time() - t0)
            print("checkpoint %s" % ev.get("path"))

    steps = a.steps
    try:
        while True:
            if steps:
                batch = min(steps, 200)
                steps -= batch
            else:
                batch = 200
            res = tr.run(steps=batch, on_event=on_event, command=command)
            if res.get("state") == "detached":
                print("detached at step %d — checkpoint %s"
                      % (res["step"], res["checkpoint"]))
                break
            if steps <= 0 and a.steps:
                path = tr.save()
                print("finished %d steps — checkpoint %s" % (tr.step, path))
                break
    except Exception as e:  # noqa: BLE001
        write_state(step=tr.step,
                    loss=tr.history[-1] if tr.history else None,
                    error=str(e))
        print("training failed: %s" % e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
