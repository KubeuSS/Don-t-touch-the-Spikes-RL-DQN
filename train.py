import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import SpikesEnv

LR             = 2e-5
GAMMA          = 0.995
BATCH_SIZE     = 128      
BUFFER_SIZE    = 50000
MIN_BUFFER     = 128
TAU            = 0.005   # współczynnik soft update target network
EPS_START      = 0.2
EPS_END        = 0.02
EPS_DECAY      = 0.999
TOTAL_STEPS    = 200_000
LOG_EVERY      = 1_000
SAVE_EVERY     = 10_000 


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32),
            torch.tensor(a,            dtype=torch.long),
            torch.tensor(r,            dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d,            dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class QNetwork(nn.Module):

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 64),  nn.ReLU(),
            nn.Linear(64, 128),      nn.ReLU(),
        )
        self.value     = nn.Linear(128, 1)          # V(s)
        self.advantage = nn.Linear(128, n_actions)  # A(s,a)

    def forward(self, x):
        x = self.backbone(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=-1, keepdim=True)  # Q(s,a)


def train():
    env = SpikesEnv(frame_skip=5, curriculum=True)
    obs_dim  = env.observation_space.shape[0]  # 11
    n_actions = env.action_space.n              # 2

    q_net      = QNetwork(obs_dim, n_actions)
    target_net = QNetwork(obs_dim, n_actions)

    if os.path.exists("model_final.pt"):
        q_net.load_state_dict(torch.load("model_final.pt", weights_only=True))
        print("Wczytano model_final.pt, kontyunacja")
        eps = EPS_END
    else:
        eps = EPS_START

    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_SIZE)

    state, _ = env.reset()

    # statystyki do logow
    ep_reward   = 0.0
    ep_survived = 0
    ep_start    = 0
    ep_count    = 0
    recent_scores = deque(maxlen=100)
    best_avg = -1.0
    if os.path.exists("model_best.meta"):
        try:
            best_avg = float(open("model_best.meta").read().strip())
            print(f"dotychczasowy rekord model_best.pt: best_avg={best_avg:.2f}")
        except Exception:
            pass

    for step in range(1, TOTAL_STEPS + 1):

        # epsilon-greedy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = q_net(torch.tensor(state, dtype=torch.float32))
            action = int(q_vals.argmax())

        # krok w środowisku
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if not done:
            reward += 0.1

        buffer.push(state, action, reward, next_state, float(done))

        state       = next_state
        ep_reward  += reward
        ep_survived = info["survived"]
        ep_start    = info["start"]

        if done:
            ep_count += 1
            if ep_start == 0:
                recent_scores.append(ep_survived)
            state, _ = env.reset()
            ep_reward = 0.0

            if len(buffer) < BATCH_SIZE:  
                continue                   

            s, a, r, s2, d = buffer.sample(BATCH_SIZE)
            q_pred = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_actions = q_net(s2).argmax(1)
                q_next = target_net(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                q_target = r + GAMMA * q_next * (1.0 - d)

            loss = nn.functional.smooth_l1_loss(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)  # twardy limit normy gradientu
            optimizer.step()
            for tp, qp in zip(target_net.parameters(), q_net.parameters()):
                tp.data.copy_(TAU * qp.data + (1 - TAU) * tp.data)

        
        eps = max(EPS_END, eps * EPS_DECAY)

        # logi
        if step % LOG_EVERY == 0:
            avg = np.mean(recent_scores) if recent_scores else 0.0
            print(f"krok {step:>7} | eps {eps:.4f} | epizody {ep_count:>5} "
                  f"| avg odbić, starty od 0 (100 ep): {avg:.1f}")

            # zapisuj NAJLEPSZY model osobno 
            if len(recent_scores) >= 50 and avg > best_avg:
                best_avg = avg
                torch.save(q_net.state_dict(), "model_best.pt")
                with open("model_best.meta", "w") as f:   # zapamiętaj rekord na przyszłe biegi
                    f.write(f"{avg:.4f}")
                print(f"  → nowy rekord avg {avg:.2f} — zapisano model_best.pt")

        # zapis modelu co interwal
        if step % SAVE_EVERY == 0:
            path = f"model_{step}.pt"
            torch.save(q_net.state_dict(), path)
            print(f"  → zapisano {path}")

    torch.save(q_net.state_dict(), "model_final.pt")
    print("Model zapisany jako model_final.pt")
    env.close()


if __name__ == "__main__":
    train()
