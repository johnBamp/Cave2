# Centralized tunables for the cave demo.

# Screen layout
MAP_WIDTH = 700
MAP_HEIGHT = 500
SIDEBAR_WIDTH = 200

# Cave generation
CELL_SIZE = 10
FILL_PROB = 0.42
SMOOTH_PASSES = 4

# Player
PLAYER_RADIUS = 8  # smaller footprint to reduce snagging on walls
PLAYER_MOVE_SPEED = 180.0
PLAYER_TURN_SPEED = 160.0

# Vision
MAX_VIEW_DIST = 420.0
FOV_DEG = 90.0
RAY_COUNT = 75
RAY_STEP_FRACTION = 0.25  # fraction of a tile to step each ray march

# Energy
ENERGY_MAX = 100.0
ENERGY_IDLE_DRAIN_PER_S = 0.5
ENERGY_MOVE_DRAIN_PER_PX = 0.02
FRUIT_ENERGY_GAIN = 35.0

# Bushes / fruit
BUSH_COUNT = 10
FRUIT_REGEN_SECONDS = 8.0

# Control
# "manual" -> keyboard; "nn" -> neural agent drives movement
CONTROL_MODE = "nn"
# Where to save/load NN weights
MODEL_PATH = "nn_weights.json"
# Manual play is for debugging only; no supervised recording
TRAIN_ON_MANUAL = False
# NN learning rates / RL params
TRAIN_EPOCHS = 1  # used by Q-updates internally
TRAIN_LR = 0.005
GAMMA = 0.95
EPSILON_START = 0.40
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9999  # slower decay to keep exploring
REPLAY_SIZE = 5000
BATCH_SIZE = 32
ROLLOUT_STEPS = 256
PPO_EPOCHS = 4
PPO_CLIP = 0.2
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 0.5
LAMBDA = 0.95

# Fast simulation
FAST_STEPS = 6  # physics/learning steps per render when fast mode enabled

# Reward shaping
FRUIT_REWARD_BONUS = 8.0
STALL_STEPS = 50          # steps with tiny movement before penalty
STALL_PENALTY = 0.8
STALL_RESPAWN_STEPS = 300
NEW_CELL_REWARD = 0.25
FRUIT_APPROACH_GAIN = 0.2
TURN_BONUS = 0.25
TURN_BONUS_THRESHOLD_DEG = 6.0
BACKFORTH_PENALTY = 0.08
STRAIGHT_PENALTY = 0.15
STRAIGHT_STEPS = 25
FRUIT_SEEN_BONUS = 1.0

# Corridor tolerance
CLOSE_WALL_DIST = 12.0
CLOSE_WALL_PENALTY = 0.03

# Discrete agent step sizes (agent mode only)
AGENT_TURN_DEG = 18.0
AGENT_STEP_PX = 8.0

# Intrinsic novelty rewards
NEW_OBS_REWARD = 0.12
NEW_POSE_REWARD = 0.08

# Colors
BACKGROUND_COLOR = (10, 10, 15)
WORLD_BG_COLOR = (12, 12, 18)
SIDEBAR_BG_COLOR = (24, 26, 38)
WALL_COLOR = (25, 25, 30)
PLAYER_COLOR = (220, 220, 70)
PLAYER_NOSE_COLOR = (255, 120, 50)
RAY_COLOR = (80, 200, 255)
FOV_CLEAR_COLOR = (235, 235, 240)
FOV_WALL_COLOR = (230, 80, 80)
COMPASS_RING_COLOR = (50, 80, 120)
TEXT_COLOR = (180, 190, 210)
TEXT_DIM_COLOR = (140, 160, 190)
BUSH_COLOR = (60, 130, 70)
FRUIT_COLOR = (215, 50, 50)
ENERGY_BAR_BG = (40, 45, 55)
ENERGY_BAR_FILL = (90, 200, 120)
FOV_BUSH_COLOR = (70, 170, 90)
FOV_FRUIT_COLOR = (255, 90, 90)

# Sidebar display
FOV_BAR_TOP = 24
FOV_BAR_HEIGHT = 18
FOV_BAR_MARGIN = 14
