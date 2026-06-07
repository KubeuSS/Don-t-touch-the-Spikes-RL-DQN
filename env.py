import random

import numpy as np

from game import (
    Bird, Wall, game_tick,
    WIDTH, HEIGHT, NUM_SLOTS,
    SKY, WALL_COLOR, WHITE, BLACK, GRAY,
    render_outlined, draw_text, get_title,
)


class _Box:
    def __init__(self, shape):
        self.shape = shape


class _Discrete:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randrange(self.n)


class SpikesEnv:
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, frame_skip=1, curriculum=False):
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.curriculum = curriculum
        self.start_scores = [0, 0, 0, 0, 300, 600, 900]
        self._start_score = 0

        self.observation_space = _Box((11,))
        self.action_space = _Discrete(2)

        self._screen = None
        self._clock = None
        self._chess_font = None
        self._spike_font = None
        self._font_big = None
        self._font_sm = None
        self.closed = False


    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.bird = Bird()
        self.left = Wall("left")
        self.right = Wall("right")


        start = random.choice(self.start_scores) if self.curriculum else 0
        self._start_score = start
        self.score = start
        self.left.randomize(start)
        self.right.randomize(start)

        return self._obs(), {}

    def step(self, action):
        if action == 1:
            self.bird.flap()
        target = self.right if self.bird.vx > 0 else self.left
        centers = self._gap_centers(target)

        total_reward = 0.0
        terminated = False
        for _ in range(self.frame_skip):
            self.score, reward, terminated = game_tick(self.bird, self.left, self.right, self.score)
            total_reward += reward
            self.render()
            if terminated:
                break

        if not terminated:
            nearest = min(centers, key=lambda c: abs(self.bird.y - c))
            distance = abs(self.bird.y - nearest) / HEIGHT
            total_reward += (1.0 - distance) * 0.1

        return self._obs(), total_reward, terminated, False, {
            "score":    self.score // 100,
            "survived": (self.score - self._start_score) // 100,
            "start":    self._start_score // 100,
        }

    def _gap_centers(self, wall) -> list[float]:
        slot_h = HEIGHT / NUM_SLOTS
        free = sorted(i for i in range(NUM_SLOTS) if i not in set(wall.spike_slots))
        runs = []
        start = prev = free[0]
        for s in free[1:]:
            if s == prev + 1:
                prev = s
            else:
                runs.append((start, prev))
                start = prev = s
        runs.append((start, prev))
        return [((a + b) / 2 + 0.5) * slot_h for a, b in runs]


    def _obs(self) -> np.ndarray:
        target = self.right if self.bird.vx > 0 else self.left
        spikes = set(target.spike_slots)
        obs = np.array([
            self.bird.y / HEIGHT,
            self.bird.vy / 35.0,
            *[1.0 if i in spikes else 0.0 for i in range(NUM_SLOTS)],
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)


    def render(self):
        if self.render_mode != "human":
            return

        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Don't Touch the Chess")
            self._clock = pygame.time.Clock()
            self._chess_font = pygame.font.SysFont("segoeuisymbol", 32)
            self._spike_font = pygame.font.SysFont("segoeuisymbol", 52)
            self._font_big = pygame.font.SysFont("Arial", 48, bold=True)
            self._font_sm  = pygame.font.SysFont("Arial", 20)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._screen.fill(SKY)
        self.left.draw(self._screen, self._spike_font)
        self.right.draw(self._screen, self._spike_font)
        self.bird.draw(self._screen, self._chess_font)
        draw_text(self._screen, self._font_big, f"elo: {self.score}", WHITE, WIDTH // 2, 38)
        draw_text(self._screen, self._font_sm, get_title(self.score), WHITE, WIDTH // 2, 78)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
        self.closed = True
