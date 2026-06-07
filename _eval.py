import sys
import glob
import numpy as np
import torch
from env import SpikesEnv
from train import QNetwork

N = 500
paths = sys.argv[1:] or sorted(glob.glob("*.pt"))

env = SpikesEnv(frame_skip=5, curriculum=False)
obs_dim = env.observation_space.shape[0]
n_act = env.action_space.n

print(f"Ocena greedy, {N} epizodów od startu 0:\n")
results = []
for p in paths:
    net = QNetwork(obs_dim, n_act)
    net.load_state_dict(torch.load(p, weights_only=True))
    net.eval()
    scores = []
    for _ in range(N):
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                a = int(net(torch.tensor(obs, dtype=torch.float32)).argmax())
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                scores.append(info["survived"])
                break
    avg = float(np.mean(scores))
    results.append((avg, p))
    print(f"{p:<20} avg {avg:5.2f}   max {max(scores):>3}")

print()
best_avg, best_p = max(results)
print(f"NAJLEPSZY: {best_p}  (avg {best_avg:.2f})")
