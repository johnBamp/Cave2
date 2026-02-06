import pygame
import sys
import math

pygame.init()

WIDTH, HEIGHT = 900, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Test")

WHITE = (255, 255, 255)
GRID = (120, 120, 120)
BLACK = (0, 0, 0)

SEEN_EMPTY_FILL = (40, 40, 100)   # blue for visible empty
SEEN_WALL_FILL = (255, 255, 255)  # white for visible wall
PLAYER_COLOR = (240, 240, 0)

tilesX = 9
tilesY = 9

tileWidth = WIDTH // tilesX
tileHeight = HEIGHT // tilesY


# Cell types
EMPTY = 0
WALL = 1


class Cell:
    def __init__(self, gx, gy, cell_type=EMPTY):
        self.gx = gx
        self.gy = gy
        self.type = cell_type

    @property
    def rect(self):
        return pygame.Rect(self.gx * tileWidth, self.gy * tileHeight, tileWidth, tileHeight)

    def draw(self, seen=False):
        r = self.rect

        # Always draw grid outline for EVERY cell (even invisible wall/object cells)
        pygame.draw.rect(screen, GRID, r, 1)

        # Only draw the "content" (fill) if seen
        if not seen:
            return

        if self.type == WALL:
            pygame.draw.rect(screen, SEEN_WALL_FILL, r)   # white fill for seen wall
        else:
            pygame.draw.rect(screen, SEEN_EMPTY_FILL, r)  # blue fill for seen empty

        pygame.draw.rect(screen, GRID, r, 1)



cellList = []
grid = []  # grid[gx][gy] -> Cell


def initCells():
    global grid
    cellList.clear()
    grid = [[None for _ in range(tilesY)] for _ in range(tilesX)]

    for gx in range(tilesX):
        for gy in range(tilesY):
            # Make border cells walls
            is_border = (gx == 0 or gy == 0 or gx == tilesX - 1 or gy == tilesY - 1)
            ctype = WALL if is_border else EMPTY
            cell = Cell(gx, gy, ctype)
            grid[gx][gy] = cell
            cellList.append(cell)


def world_to_cell(px, py):
    cx = int(px // tileWidth)
    cy = int(py // tileHeight)
    return cx, cy


def in_bounds(cx, cy):
    return 0 <= cx < tilesX and 0 <= cy < tilesY


def cast_visibility(player_pos, player_angle_rad):
    """
    Fan of rays forward.
    Marks empty cells as seen (blue),
    and the first wall cell hit as seen (white).
    Rays stop at wall cells.
    """
    seen_cells = set()

    FOV_DEG = 60
    RAYS = 121
    STEP = 4  # pixels per step

    half_fov = math.radians(FOV_DEG) / 2
    start_angle = player_angle_rad - half_fov
    end_angle = player_angle_rad + half_fov

    max_dist = int(math.hypot(WIDTH, HEIGHT)) + 10
    steps = max_dist // STEP

    for i in range(RAYS):
        t = i / (RAYS - 1) if RAYS > 1 else 0.5
        ang = start_angle + (end_angle - start_angle) * t
        dx = math.cos(ang)
        dy = math.sin(ang)

        x, y = player_pos

        for _ in range(steps):
            x += dx * STEP
            y += dy * STEP

            cx, cy = world_to_cell(x, y)
            if not in_bounds(cx, cy):
                break

            cell = grid[cx][cy]
            seen_cells.add((cx, cy))

            # Stop ray at wall
            if cell.type == WALL:
                break

    return seen_cells


def drawPlayer(pos, angle_rad):
    px, py = int(pos[0]), int(pos[1])
    pygame.draw.circle(screen, PLAYER_COLOR, (px, py), 10)

    length = 40
    fx = px + int(math.cos(angle_rad) * length)
    fy = py + int(math.sin(angle_rad) * length)
    pygame.draw.line(screen, PLAYER_COLOR, (px, py), (fx, fy), 3)


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

    seen_cells = cast_visibility(player_pos, player_angle)

    screen.fill(BLACK)

    for cell in cellList:
        cell.draw(seen=(cell.gx, cell.gy) in seen_cells)

    drawPlayer(player_pos, player_angle)

    pygame.display.flip()

pygame.quit()
sys.exit()
