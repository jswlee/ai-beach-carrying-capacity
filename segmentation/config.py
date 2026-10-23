"""
Shared configuration for waterline YOLO26 segmentation pipeline.
All paths are relative to the repo root (where the scripts are run from).
"""

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
SOURCE_DATASET  = "train_data/waterline-segmentation.v11i.png-mask-semantic"
OUTPUT_DATASET  = "train_data/waterline-segmentation.v11i.png-mask-semantic-preprocessed"

# ---------------------------------------------------------------------------
# Class definitions  (from _classes.csv)
# 0 = background                        — outside polygon, forced by preprocessing
# 1 = base-background-layer-segmentation — sky/horizon band inside polygon
# 2 = beach
# 3 = water
# ---------------------------------------------------------------------------
NUM_CLASSES  = 4
CLASS_NAMES  = ["background", "base-background-layer-segmentation", "beach", "water"]

# ---------------------------------------------------------------------------
# ROI Polygon for preprocessing (from mask.py)
# ---------------------------------------------------------------------------
POLYGON_PTS_LIST = [
    [2, 693], [1655, 380], [1861, 485], [1850, 617], [1819, 614],
    [1790, 628], [1794, 644], [1782, 655], [1757, 673], [1753, 681],
    [1783, 693], [1642, 749], [1621, 723], [1444, 777], [1206, 823],
    [1169, 820], [904, 868], [804, 907], [745, 937], [676, 943],
    [621, 960], [595, 954], [580, 967], [551, 968], [472, 972],
    [249, 1012], [214, 1021], [159, 1023], [49, 1050], [4, 1073]
]

# Post-move polygon (camera shifted).  mask.py selects per-frame:
# filenames that sort >= CAMERA_MOVE_CUTOFF use this polygon.
POLYGON_PTS_AFTER_LIST = [
    [1, 1060], [62, 1043], [171, 1019], [203, 1019], [236, 1017],
    [300, 1002], [330, 999], [398, 987], [421, 987], [430, 978],
    [501, 972], [509, 975], [521, 966], [589, 966], [592, 955],
    [621, 955], [630, 958], [654, 949], [677, 946], [707, 937],
    [780, 934], [792, 940], [825, 961], [866, 952], [819, 908],
    [836, 890], [866, 881], [880, 881], [913, 863], [951, 852],
    [975, 846], [1019, 837], [1048, 828], [1095, 825], [1125, 816],
    [1172, 810], [1210, 816], [1237, 810], [1260, 810], [1284, 805],
    [1305, 805], [1349, 799], [1387, 787], [1413, 787], [1443, 799],
    [1496, 784], [1461, 775], [1478, 763], [1499, 757], [1508, 754],
    [1540, 746], [1558, 746], [1575, 743], [1590, 734], [1620, 728],
    [1637, 725], [1681, 734], [1814, 687], [1779, 672], [1785, 663],
    [1802, 654], [1823, 648], [1820, 634], [1817, 622], [1835, 613],
    [1876, 607], [1875, 465], [1875, 440], [1875, 431], [1875, 431],
    [1875, 406], [1, 668], [1, 969], [1, 1016]
]
CAMERA_MOVE_CUTOFF = "youtube_snapshot_Kaanapali_20260127_222935.jpg"

# Filenames carry UTC timestamps; the camera is in Hawaii (HST = UTC-10, no DST).
LOCAL_UTC_OFFSET_H = -10
# Daytime window in local wall-clock hours for --time day
DAY_START_H = 5.0
DAY_END_H   = 20.0

SPLITS = ["train", "valid", "test"]


# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL_SIZE   = "m"
DEFAULT_EPOCHS       = 150
DEFAULT_IMGSZ        = 640
DEFAULT_BATCH        = 16
DEFAULT_PATIENCE     = 20
DEFAULT_DEVICE       = "0"
DEFAULT_PROJECT      = "segment/runs/segment/yolo-waterline-seg"
DEFAULT_RUN_NAME     = "train"
DEFAULT_LR0          = 6e-4
DEFAULT_LRF          = 0.01
DEFAULT_WEIGHT_DECAY = 0.0005
DEFAULT_WARMUP_EPOCHS = 3

# ---------------------------------------------------------------------------
# Augmentation defaults
# ---------------------------------------------------------------------------
DEFAULT_MOSAIC      = 1.0
DEFAULT_MIXUP       = 0.1
DEFAULT_COPY_PASTE  = 0.1
DEFAULT_DEGREES     = 5.0
DEFAULT_TRANSLATE   = 0.1
DEFAULT_SCALE       = 0.5
DEFAULT_SHEAR       = 2.0
DEFAULT_PERSPECTIVE = 0.0005
DEFAULT_FLIPUD      = 0.0
DEFAULT_FLIPLR      = 0.5
DEFAULT_HSV_H       = 0.015
DEFAULT_HSV_S       = 0.7
DEFAULT_HSV_V       = 0.4
DEFAULT_ERASING     = 0.4

# ---------------------------------------------------------------------------
# Tiling  (configurable tile size and selection)
# ---------------------------------------------------------------------------
TILE_SIZE      = 640   # output tile size after resize (YOLO expects square)
TILE_HEIGHT    = 720   # native tile height before resize (for bottom tiles)
TILE_MODE      = "bottom"  # "all" for full grid, "bottom" for 3 bottom tiles only
TILE_COLS_FULL = 3      # columns when mode="all"
TILE_ROWS_FULL = 2      # rows when mode="all"

# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
DEFAULT_PREVIEW_DIR   = "preview_preprocessed"
DEFAULT_PREVIEW_SPLIT = "train"
DEFAULT_PREVIEW_N     = 500
