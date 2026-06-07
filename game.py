import pygame
import random
import sys

WIDTH, HEIGHT = 400, 600
FPS = 60
GRAVITY = 0.45
FLAP_STRENGTH = -9
BIRD_SPEED = 4
WALL_WIDTH = 45
SPIKE_LEN = 28
SPIKE_BASE = 18
NUM_SLOTS = 9

TITLES = [
    (0,    "Noob..."),
    (800,  "Amator"),
    (1200, "Nowicjusz"),
    (1500, "Średniak"),
    (2200, "CM"),
    (2300, "FM"),
    (2400, "IM"),
    (2500, "GM")
]

def get_title(elo: int) -> str:
    title = TITLES[0][1]
    for threshold, name in TITLES:
        if elo >= threshold:
            title = name
    return title

SKY = (100, 180, 255)
WALL_COLOR = (60, 60, 80)
SPIKE_COLOR = (220, 50, 50)
BIRD_COLOR = (255, 220, 30)
EYE_COLOR = (30, 30, 30)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
BLACK = (0, 0, 0)


class Bird:
    R = 14

    def __init__(self):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.vy = 0.0
        self.vx = BIRD_SPEED

    def flap(self):
        self.vy = FLAP_STRENGTH

    def update(self):
        self.vy += GRAVITY
        self.y += self.vy
        self.x += self.vx

    def draw(self, surf, font):
        img = font.render("♔", True, WHITE)
        img.set_colorkey((0, 0, 0))
        surf.blit(img, (int(self.x) - img.get_width() // 2,
                        int(self.y) - img.get_height() // 2))

    @property
    def rect(self):
        return pygame.Rect(self.x - self.R, self.y - self.R,
                           self.R * 2, self.R * 2)


class Wall:
    def __init__(self, side: str):
        self.side = side  # 'left' | 'right'
        self.x = 0 if side == 'left' else WIDTH - WALL_WIDTH
        self.slot_h = HEIGHT / NUM_SLOTS
        self.spike_slots: list[int] = []
        self.randomize()

    def randomize(self, score: int = 0):
        free = max(1, 8 - score // 300)
        empty = set(random.sample(range(NUM_SLOTS), free))
        self.spike_slots = [i for i in range(NUM_SLOTS) if i not in empty]

    def draw(self, surf, font):
        pygame.draw.rect(surf, WALL_COLOR, (self.x, 0, WALL_WIDTH, HEIGHT))
        for slot in self.spike_slots:
            cy = int((slot + 0.5) * self.slot_h)
            img = font.render("♛", True, BLACK, WALL_COLOR)
            img.set_colorkey(WALL_COLOR)
            iw, ih = img.get_width(), img.get_height()
            blit_x = (self.x + WALL_WIDTH - iw // 2) if self.side == 'left' \
                     else (self.x - iw // 2)
            surf.blit(img, (blit_x, cy - ih // 2))

    def spike_rects(self) -> list[pygame.Rect]:
        rects = []
        for slot in self.spike_slots:
            cy = int((slot + 0.5) * self.slot_h)
            y = cy - SPIKE_BASE // 2
            if self.side == 'left':
                rects.append(pygame.Rect(self.x + WALL_WIDTH, y, SPIKE_LEN, SPIKE_BASE))
            else:
                rects.append(pygame.Rect(self.x - SPIKE_LEN, y, SPIKE_LEN, SPIKE_BASE))
        return rects

    @property
    def inner_x(self):
        return self.x + WALL_WIDTH if self.side == 'left' else self.x


def render_outlined(font, text, fg, outline, width=2):
    CHROMA = (255, 0, 255)
    base = font.render(text, True, fg, CHROMA)
    base.set_colorkey(CHROMA)
    ol   = font.render(text, True, outline, CHROMA)
    ol.set_colorkey(CHROMA)
    w, h = base.get_width() + width*2, base.get_height() + width*2
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for dx in range(-width, width+1):
        for dy in range(-width, width+1):
            if dx or dy:
                surf.blit(ol, (width+dx, width+dy))
    surf.blit(base, (width, width))
    return surf

def game_tick(bird, left: Wall, right: Wall, score: int):
    """Jeden krok fizyki — bez renderowania.
    score w tej samej skali co ludzka gra (ELO, przyrost +100 za odbicie).
    Zwraca (score, reward, terminated)."""
    reward = 0.0
    terminated = False

    bird.update()

    if bird.x - bird.R <= left.inner_x:
        bird.x = left.inner_x + bird.R
        bird.vx = abs(bird.vx)
        score += 100
        right.randomize(score)
        reward += 1.0

    if bird.x + bird.R >= right.inner_x:
        bird.x = right.inner_x - bird.R
        bird.vx = -abs(bird.vx)
        score += 100
        left.randomize(score)
        reward += 1.0

    br = bird.rect
    for rect in left.spike_rects() + right.spike_rects():
        if br.colliderect(rect):
            terminated = True
            reward = -1.0
            break

    if bird.y - bird.R < 0 or bird.y + bird.R > HEIGHT:
        terminated = True
        reward = -1.0

    return score, reward, terminated


def draw_text(surf, font, text, color, cx, cy):
    img = font.render(text, True, color)
    surf.blit(img, (cx - img.get_width() // 2, cy - img.get_height() // 2))


def main():
    pygame.init()
    pygame.mixer.music.load("04. Crystals.mp3")
    pygame.mixer.music.play(-1)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Don't Touch the Chess")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("Arial", 48, bold=True)
    font_med = pygame.font.SysFont("Arial", 28, bold=True)
    font_sm  = pygame.font.SysFont("Arial", 20)
    # Segoe UI Symbol ma szachowe glify na Windowsie
    chess_font = pygame.font.SysFont("segoeuisymbol", 32)
    spike_font = pygame.font.SysFont("segoeuisymbol", 52)

    def new_game():
        bird = Bird()
        left = Wall('left')   # randomize(0) wywoływane w __init__
        right = Wall('right')
        return bird, left, right, 0

    bird, left, right, score = new_game()
    state = 'wait'  # wait | play | dead

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            tap = (event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_UP)) \
               or event.type == pygame.MOUSEBUTTONDOWN

            if tap:
                if state == 'wait':
                    state = 'play'
                    bird.flap()
                elif state == 'play':
                    bird.flap()
                elif state == 'dead':
                    bird, left, right, score = new_game()
                    state = 'wait'
                    pygame.mixer.music.play(-1)

        if state == 'play':
            bird.update()

            # odbicie od lewej ściany
            if bird.x - bird.R <= left.inner_x:
                bird.x = left.inner_x + bird.R
                bird.vx = abs(bird.vx)
                score += 100
                right.randomize(score)

            # odbicie od prawej ściany
            if bird.x + bird.R >= right.inner_x:
                bird.x = right.inner_x - bird.R
                bird.vx = -abs(bird.vx)
                score += 100
                left.randomize(score)

            br = bird.rect
            for rect in left.spike_rects() + right.spike_rects():
                if br.colliderect(rect):
                    state = 'dead'
                    pygame.mixer.music.stop()

            if bird.y - bird.R < 0 or bird.y + bird.R > HEIGHT:
                state = 'dead'
                pygame.mixer.music.stop()

        # --- rysowanie ---
        screen.fill(SKY)
        left.draw(screen, spike_font)
        right.draw(screen, spike_font)
        bird.draw(screen, chess_font)

        draw_text(screen, font_big, f"elo: {score}", WHITE, WIDTH // 2, 38)
        draw_text(screen, font_sm, get_title(score), WHITE, WIDTH // 2, 78)

        if state == 'wait':
            draw_text(screen, font_med, "Tap / Space to start", BLACK, WIDTH // 2, HEIGHT // 2 + 60)

        if state == 'dead':
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            screen.blit(overlay, (0, 0))
            draw_text(screen, font_big, "GAME OVER", WHITE, WIDTH // 2, HEIGHT // 2 - 50)
            draw_text(screen, font_med, f"Wynik: {score}", WHITE, WIDTH // 2, HEIGHT // 2 + 10)
            draw_text(screen, font_sm, "Kliknij aby zagrać ponownie", GRAY, WIDTH // 2, HEIGHT // 2 + 55)

        pygame.display.flip()


if __name__ == "__main__":
    main()
