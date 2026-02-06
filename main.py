import pygame
import sys
import math

pygame.init()

WIDTH, HEIGHT = 900, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Test")

GRID = (120, 120, 120)
BLACK = (0, 0, 0)

SEEN_EMPTY_FILL = (40, 40, 100)   # blue-ish (empty)
SEEN_WALL_FILL = (255, 255, 255)  # white (wall)
PLAYER_COLOR = (240, 240, 0)

tilesX = 9
tilesY = 9

tileWidth = WIDTH // tilesX
tileHeight = HEIGHT // tilesY

# Cell types
EMPTY = 0
WALL = 1

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def lerp(a, b, t: float):
    return a + (b - a) * t

def lerp_color(c0, c1, t: float):
    t = clamp01(t)
    return (
        int(lerp(c0[0], c1[0], t)),
        int(lerp(c0[1], c1[1], t)),
        int(lerp(c0[2], c1[2], t)),
    )

class Cell:
    def __init__(self, gx, gy, cell_type=EMPTY):
        self.gx = gx
        self.gy = gy
        self.type = cell_type  # objective reality

    @property
    def rect(self):
        return pygame.Rect(self.gx * tileWidth, self.gy * tileHeight, tileWidth, tileHeight)

    def draw(self, ever_seen: bool, belief_wall: float):
        r = self.rect

        # Always draw grid outline
        pygame.draw.rect(screen, GRID, r, 1)

        # If agent has never seen this cell, don't fill it (unknown to the agent)
        if not ever_seen:
            return

        # Subjective reality rendering: blend between empty-blue and wall-white
        fill = lerp_color(SEEN_EMPTY_FILL, SEEN_WALL_FILL, belief_wall)
        pygame.draw.rect(screen, fill, r)
        pygame.draw.rect(screen, GRID, r, 1)

cellList = []

objective = []   # objective[gx][gy] -> Cell (objective reality)
subjective = []  # subjective[gx][gy] in [0,1], probability of WALL (subjective reality)

ever_seen = set()  # cells the agent has ever observed via rays (memory of "seen")

def initCells():
    global objective, subjective, ever_seen
    cellList.clear()
    ever_seen.clear()

    objective = [[None for _ in range(tilesY)] for _ in range(tilesX)]
    subjective = [[0.5 for _ in range(tilesY)] for _ in range(tilesX)]  # unknown everywhere initially

    for gx in range(tilesX):
        for gy in range(tilesY):
            # Objective world: border walls
            is_border = (gx == 0 or gy == 0 or gx == tilesX - 1 or gy == tilesY - 1)
            ctype = WALL if is_border else EMPTY
            cell = Cell(gx, gy, ctype)
            objective[gx][gy] = cell
            cellList.append(cell)

def world_to_cell(px, py):
    cx = int(px // tileWidth)
    cy = int(py // tileHeight)
    return cx, cy

def in_bounds(cx, cy):
    return 0 <= cx < tilesX and 0 <= cy < tilesY

def update_subjective_cell(cx, cy, target, alpha):
    """Move subjective[cx][cy] toward target (0=empty, 1=wall)."""
    subjective[cx][cy] = clamp01(subjective[cx][cy] + alpha * (target - subjective[cx][cy]))

def cast_and_update(player_pos, player_angle_rad):
    """
    Fan of rays forward.

    For each ray:
      - cells before the first wall hit => evidence EMPTY (target 0)
      - first wall cell hit => evidence WALL (target 1)
      - beyond wall => no evidence

    Returns:
      current_visible: set of cells intersected this frame (optional UI)
    """
    current_visible = set()

    FOV_DEG = 60
    RAYS = 121
    STEP = 4  # pixels per step

    # learning rate for subjective updates (bigger => faster convergence)
    ALPHA = 0.35

    half_fov = math.radians(FOV_DEG) / 2
    start_angle = player_angle_rad - half_fov
    end_angle = player_angle_rad + half_fov

    max_dist = int(math.hypot(WIDTH, HEIGHT)) + 10
    steps = max_dist // STEP

    # Aggregate evidence per frame so many rays don't over-update the same cell
    evidence = {}

    for i in range(RAYS):
        t = i / (RAYS - 1) if RAYS > 1 else 0.5
        ang = start_angle + (end_angle - start_angle) * t
        dx = math.cos(ang)
        dy = math.sin(ang)

        x, y = player_pos
        ray_cells = []
        hit_wall_cell = None

        for _ in range(steps):
            x += dx * STEP
            y += dy * STEP

            cx, cy = world_to_cell(x, y)
            if not in_bounds(cx, cy):
                break

            # avoid repeating the same cell as we step within it
            if ray_cells and ray_cells[-1] == (cx, cy):
                continue

            current_visible.add((cx, cy))
            ray_cells.append((cx, cy))

            cell = objective[cx][cy]
            if cell.type == WALL:
                hit_wall_cell = (cx, cy)
                break

        # Convert the ray into logical facts:
        # - all cells before hit are empty
        # - hit cell is wall
        if hit_wall_cell is not None:
            for (cx, cy) in ray_cells:
                ever_seen.add((cx, cy))
                if (cx, cy) == hit_wall_cell:
                    evidence.setdefault((cx, cy), []).append(1.0)
                else:
                    evidence.setdefault((cx, cy), []).append(0.0)
        else:
            # If no wall was hit: all traversed cells are empty evidence
            for (cx, cy) in ray_cells:
                ever_seen.add((cx, cy))
                evidence.setdefault((cx, cy), []).append(0.0)

    # Apply aggregated subjective updates once per frame
    for (cx, cy), targets in evidence.items():
        target = sum(targets) / len(targets)
        update_subjective_cell(cx, cy, target, ALPHA)

    return current_visible

def drawPlayer(pos, angle_rad):
    px, py = int(pos[0]), int(pos[1])
    pygame.draw.circle(screen, PLAYER_COLOR, (px, py), 10)

    length = 40
    fx = px + int(math.cos(angle_rad) * length)
    fy = py + int(math.sin(angle_rad) * length)
    pygame.draw.line(screen, PLAYER_COLOR, (px, py), (fx, fy), 3)

def renderSubjective():
    for cell in cellList:
        p_wall = subjective[cell.gx][cell.gy]
        cell.draw(
            ever_seen=((cell.gx, cell.gy) in ever_seen),
            belief_wall=p_wall
        )
    drawPlayer(player_pos, player_angle)

# --- Setup ---
initCells()

player_pos = (WIDTH / 2, HEIGHT / 2)
player_angle = 0.0

running = True
clock = pygame.time.Clock()
ROT_SPEED = math.radians(120)

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]:
        player_angle -= ROT_SPEED * dt
    if keys[pygame.K_e]:
        player_angle += ROT_SPEED * dt

    # Raycast and update subjective reality (subjective map)
    current_visible = cast_and_update(player_pos, player_angle)

    screen.fill(BLACK)
    renderSubjective()
    pygame.display.flip()

pygame.quit()
sys.exit()
