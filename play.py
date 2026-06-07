import sys
import pygame
import torch
from env import SpikesEnv
from train import QNetwork

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "model_final.pt"

env = SpikesEnv(render_mode="human", frame_skip=9)
obs_dim   = env.observation_space.shape[0]
n_actions = env.action_space.n

q_net = QNetwork(obs_dim, n_actions)
q_net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
q_net.eval()

pygame.mixer.init()
pygame.mixer.music.load("04. Crystals.mp3")
pygame.mixer.music.play(-1)

print(f"Załadowano: {MODEL_PATH}")
print("Zamknij okno żeby wyjść.\n")

episode = 0
while True:
    obs, _ = env.reset()
    episode += 1
    total_reward = 0.0

    while True:
        if env.closed:
            raise SystemExit

        with torch.no_grad():
            q_vals = q_net(torch.tensor(obs, dtype=torch.float32))
        action = int(q_vals.argmax())

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Gra {episode:>4} | score: {info['score']:>4} | reward: {total_reward:.1f}")
            pygame.mixer.music.play(-1)
            break

env.close()
