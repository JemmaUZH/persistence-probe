from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal MineRL reset/step frame capture smoke test.")
    parser.add_argument("--env", default="MineRLBasaltFindCave-v0")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("scratch/minerl_smoke"))
    args = parser.parse_args()

    import gym
    import minerl  # noqa: F401

    args.output.mkdir(parents=True, exist_ok=True)
    env = gym.make(args.env)
    try:
        obs = env.reset()
        print(f"reset ok: {args.env}")
        print(f"obs type: {type(obs)}")
        if hasattr(obs, "keys"):
            print(f"obs keys: {sorted(obs.keys())}")

        for idx in range(args.steps):
            action = env.action_space.noop()
            obs, reward, done, info = env.step(action)
            pov = obs["pov"] if isinstance(obs, dict) and "pov" in obs else None
            if pov is not None:
                Image.fromarray(pov).save(args.output / f"{idx:06d}.png")
            print(f"step {idx}: reward={reward} done={done}")
            if done:
                break
    finally:
        env.close()

    print(f"wrote frames to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
