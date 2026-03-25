"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          HOCKEY PRESSURE PIPELINE  —  CONSOLIDATED SINGLE FILE              ║
║                                                                              ║
║  Based on:                                                                   ║
║    • Andrienko et al. (2017) "Visual Analysis of Pressure in Football"       ║
║    • Radke et al. "Analyzing Passing Metrics in Ice Hockey using PPT Data"   ║
║    • Radke et al. "Identifying Completed Pass Types & Passing Lane Models"   ║
║                                                                              ║
║  USAGE:                                                                      ║
║    1. Edit the three file paths in Section 1 (CONFIGURATION) below          ║
║    2. Run:  python FINAL_hockey_pressure_pipeline.py                         ║
║    3. All outputs land in ./figures/ and ./output_data/                      ║
║                                                                              ║
║  SECTIONS:                                                                   ║
║    0  — Imports                                                              ║
║    1  — Configuration  ← EDIT THESE PATHS                                   ║
║    2  — Clock & data helpers                                                 ║
║    3  — Data loader (GameData class)                                         ║
║    4  — Pressure model (Andrienko formula, hockey-scaled)                    ║
║    5  — Pressure aggregation & normalisation                                 ║
║    6  — Pressing sequences (Andrienko stats A–D)                             ║
║    7  — Faceoff analysis (formation, K-Means, win-probability)               ║
║    8  — Forecheck linkage                                                    ║
║    9  — All plots (11 static + 3 GIFs)                                       ║
║   10  — CSV export helpers (single game + multi-game)                        ║
║   11  — run_full_analysis()  — chains everything                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os, re, warnings, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.animation import FuncAnimation, PillowWriter
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION  ← EDIT THESE PATHS FOR EACH NEW GAME
# ═══════════════════════════════════════════════════════════════════════════════

# ── File paths ─────────────────────────────────────────────────────────────────
TRACKING_FILES = [
    "/mnt/project/20251011_Team_A__Team_D_Tracking_P1.csv",
    "/mnt/project/20251011_Team_A__Team_D_Tracking_P2.csv",
    "/mnt/project/20251011_Team_A__Team_D_Tracking_P3.csv",
]
EVENTS_FILE = "/mnt/project/20251011_Team_A__Team_D_Events.csv"
SHIFTS_FILE  = "/mnt/project/20251011_Team_A__Team_D_Shifts.csv"

# ── Multi-game folder (for export_all_games) ──────────────────────────────────
# Set this to your Google Drive folder containing all game sub-folders
BDC_DATA_FOLDER = "/content/drive/MyDrive/BDC Data"
OUTPUT_FOLDER   = "/content/drive/MyDrive/BDC Outputs"

# ── Rink geometry ──────────────────────────────────────────────────────────────
RINK_LENGTH_FT  = 200.0   # -100 to +100 on X axis
RINK_WIDTH_FT   =  85.0   # -42.5 to +42.5 on Y axis
HOME_NET_X      =  89.0
HOME_NET_Y      =   0.0
AWAY_NET_X      = -89.0
AWAY_NET_Y      =   0.0
BLUE_LINE_HOME  =  25.0
BLUE_LINE_AWAY  = -25.0

FACEOFF_DOTS = {
    "centre":         (  0.0,   0.0),
    "home_left":      ( 69.0, -22.0),
    "home_right":     ( 69.0,  22.0),
    "away_left":      (-69.0, -22.0),
    "away_right":     (-69.0,  22.0),
    "home_neutral_l": ( 20.0, -22.0),
    "home_neutral_r": ( 20.0,  22.0),
    "away_neutral_l": (-20.0, -22.0),
    "away_neutral_r": (-20.0,  22.0),
}

# ── Pressure model (Andrienko et al. 2017, hockey-scaled) ─────────────────────
D_FRONT_FT  = 11.0   # max pressure distance directly ahead (~stick length + reach)
D_BACK_FT   =  3.7   # max pressure distance directly behind (3:1 ratio)
Q_EXPONENT  =  1.75  # distance-decay speed

THREAT_DIRECTION_MODE = "full"   # "full" or "puck"
PASSING_LANE_WEIGHT   = 0.3

# ── Faceoff / forecheck ────────────────────────────────────────────────────────
FORECHECK_WINDOW_SEC         = 5
FACEOFF_FORMATION_N_CLUSTERS = 5

# ── Aggregation ────────────────────────────────────────────────────────────────
PRESSURE_SMOOTH_SEC = 5
PRESSURE_THRESHOLD  = 15.0
PRESSURE_SAMPLE_EVERY = 1   # sample every N seconds (1 = every second)

# ── Output directories ─────────────────────────────────────────────────────────
FIGURES_DIR = "figures"
DATA_DIR    = "output_data"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLOCK & DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clock_to_sec(clock_str) -> int:
    """MM:SS countdown → elapsed seconds within period. '20:00'→0, '0:00'→1200."""
    try:
        parts = str(clock_str).strip().split(":")
        remaining = int(parts[0]) * 60 + int(parts[1])
        return 1200 - remaining
    except Exception:
        return 0

def clock_to_game_sec(period: int, clock_str) -> int:
    """Absolute game seconds across all periods."""
    return (period - 1) * 1200 + clock_to_sec(clock_str)

def sec_to_clock(elapsed_sec: int) -> str:
    """Elapsed seconds → MM:SS countdown string."""
    remaining = max(0, 1200 - elapsed_sec)
    return f"{remaining // 60}:{remaining % 60:02d}"

def detect_home_team(tracking_df: pd.DataFrame) -> str:
    """
    Infer which team is home by seeing which attacks the +X net on average.
    Home team attacks toward +X in periods 1 & 3.
    """
    p1 = tracking_df[tracking_df["Period"] == 1]
    puck = p1[p1["Player or Puck"] == "Puck"]
    teams = [t for t in tracking_df["Team"].dropna().unique() if t != ""]
    if len(teams) < 2:
        return teams[0] if teams else "Team A"
    # Whichever team's players have higher mean X in period 1 is attacking +X
    means = {}
    for t in teams:
        tp = p1[(p1["Team"] == t) & (p1["Player or Puck"] == "Player")]
        if len(tp):
            means[t] = tp["Rink Location X (Feet)"].mean()
    if means:
        return max(means, key=means.get)
    return teams[0]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADER (GameData class)
# ═══════════════════════════════════════════════════════════════════════════════

class GameData:
    """
    Loads tracking, events, and shifts CSVs for a single game.
    Game-agnostic: home/away sides are auto-detected.

    Attributes
    ----------
    tracking : pd.DataFrame   — combined, cleaned tracking data
    events   : pd.DataFrame   — cleaned events with elapsed seconds
    shifts   : pd.DataFrame   — cleaned shifts with elapsed seconds
    home_team, away_team : str
    """

    def __init__(self, tracking_files, events_file, shifts_file):
        print("Loading tracking data...")
        chunks = []
        for f in tracking_files:
            c = pd.read_csv(f, low_memory=False)
            chunks.append(c)
        raw = pd.concat(chunks, ignore_index=True)
        self._clean_tracking(raw)

        print("Loading events...")
        self._clean_events(pd.read_csv(events_file, low_memory=False))

        print("Loading shifts...")
        self._clean_shifts(pd.read_csv(shifts_file, low_memory=False))

        print(f"  Home: {self.home_team}  |  Away: {self.away_team}")
        print(f"  Tracking rows: {len(self.tracking):,}")
        print(f"  Events: {len(self.events)}  |  Shifts: {len(self.shifts)}")

    # ── Tracking ──────────────────────────────────────────────────────────────

    def _clean_tracking(self, df):
        df = df.rename(columns={
            "Rink Location X (Feet)": "x",
            "Rink Location Y (Feet)": "y",
            "Rink Location Z (Feet)": "z",
            "Player or Puck":         "entity",
            "Player Jersey Number":   "jersey",
            "Player Id":              "player_id",
            "Image Id":               "frame_id",
            "Goal Score":             "score",
        })
        df["elapsed_sec"] = df.apply(
            lambda r: clock_to_game_sec(int(r["Period"]), r["Game Clock"]), axis=1
        )
        df["is_player"] = df["entity"] == "Player"
        df["is_puck"]   = df["entity"] == "Puck"

        self.home_team = detect_home_team(df)
        all_teams = [t for t in df["Team"].dropna().unique() if t != ""]
        self.away_team = next((t for t in all_teams if t != self.home_team), all_teams[-1])

        # Normalise: home always attacks +X in odd periods
        # Period 1: home → +X, away → -X
        # Period 2: flip, Period 3: same as P1
        def assign_side(row):
            period = int(row["Period"])
            team   = row["Team"]
            if team == self.home_team:
                return "home" if period % 2 == 1 else "away"
            else:
                return "away" if period % 2 == 1 else "home"

        df["team_side"] = df.apply(assign_side, axis=1)

        # Flip X so home always attacks +X
        mask_flip = (df["entity"] != "Puck") & (df["team_side"] == "away") & \
                    (df["Period"] == 2)
        df.loc[mask_flip, "x"] = -df.loc[mask_flip, "x"]

        # Identify goalies: jersey number "G" or player further from centre ice
        df["is_goalie"] = df["jersey"].astype(str).str.upper() == "G"

        self.tracking = df.sort_values(["elapsed_sec", "frame_id"]).reset_index(drop=True)

    # ── Events ────────────────────────────────────────────────────────────────

    def _clean_events(self, df):
        df["elapsed_sec"] = df.apply(
            lambda r: clock_to_game_sec(int(r["Period"]), r["Clock"]), axis=1
        )
        df["team_side"] = df["Team"].apply(
            lambda t: "home" if t == self.home_team else "away"
        )
        self.events = df.sort_values("elapsed_sec").reset_index(drop=True)

    # ── Shifts ────────────────────────────────────────────────────────────────

    def _clean_shifts(self, df):
        df["start_sec"] = df.apply(
            lambda r: clock_to_game_sec(int(r["period"]), r["start_clock"]), axis=1
        )
        df["end_sec"] = df.apply(
            lambda r: clock_to_game_sec(int(r["period"]), r["end_clock"]), axis=1
        )
        df["team_side"] = df["Team"].apply(
            lambda t: "home" if t == self.home_team else "away"
        )
        # Guess goalies from shift length (goalies have very long shifts)
        df["is_goalie"] = False
        for side in ["home", "away"]:
            sub = df[df["team_side"] == side]
            if sub.empty: continue
            by_player = sub.groupby("Player_Id")["shift_length"].apply(
                lambda x: x.astype(str).apply(
                    lambda s: int(s.split(":")[0])*60 + int(s.split(":")[1])
                    if ":" in str(s) else 0
                ).sum()
            )
            if len(by_player):
                goalie_id = by_player.idxmax()
                df.loc[df["Player_Id"] == goalie_id, "is_goalie"] = True
        self.shifts = df.reset_index(drop=True)

    # ── Frame helpers ─────────────────────────────────────────────────────────

    def get_frame(self, frame_id) -> pd.DataFrame:
        return self.tracking[self.tracking["frame_id"] == frame_id].copy()

    def get_players(self, frame: pd.DataFrame, side=None) -> pd.DataFrame:
        pl = frame[frame["is_player"] & frame["x"].notna()]
        if side:
            pl = pl[pl["team_side"] == side]
        return pl

    def get_puck(self, frame: pd.DataFrame):
        p = frame[frame["is_puck"] & frame["x"].notna()]
        return p.iloc[0] if len(p) else None

    def get_faceoffs(self) -> pd.DataFrame:
        return self.events[self.events["Event"] == "Faceoff Win"].copy()

    def get_goals(self) -> pd.DataFrame:
        return self.events[self.events["Event"] == "Goal"].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRESSURE MODEL (Andrienko formula, hockey-scaled)
# ═══════════════════════════════════════════════════════════════════════════════

def pressure_oval_boundary(theta_rad: float) -> float:
    """
    Andrienko eq. 1: oval boundary length at angle theta from threat direction.
    theta=0 → front (max pressure), theta=pi → behind (min pressure).
    """
    z = (1.0 - np.cos(theta_rad)) / 2.0
    return D_BACK_FT + (D_FRONT_FT - D_BACK_FT) * (z**3 + 0.3*z) / 1.3


def single_pressure(presser_x, presser_y, target_x, target_y,
                    threat_vx, threat_vy) -> float:
    """
    Andrienko eq. 2: pressure exerted by one presser on one target.
    threat_v = unit vector from target toward the threat (goal + teammates).
    Returns 0–100.
    """
    dx = presser_x - target_x
    dy = presser_y - target_y
    d  = np.sqrt(dx**2 + dy**2)

    tnorm = np.sqrt(threat_vx**2 + threat_vy**2)
    if d < 1e-6 or tnorm < 1e-6:
        return 0.0

    # Angle between presser direction and threat direction (from target's POV)
    threat_unit_x = threat_vx / tnorm
    threat_unit_y = threat_vy / tnorm
    presser_unit_x = dx / d
    presser_unit_y = dy / d
    cos_theta = np.clip(
        presser_unit_x * threat_unit_x + presser_unit_y * threat_unit_y,
        -1.0, 1.0
    )
    theta = np.arccos(cos_theta)
    L = pressure_oval_boundary(theta)

    if d >= L:
        return 0.0
    return (1.0 - d / L) ** Q_EXPONENT * 100.0


def compute_threat_vector(target_x, target_y, target_side,
                          teammates, net_x, net_y):
    """
    Threat direction for a target:
      Primary  — vector from target toward the net they defend (goal direction)
                 combined with directions to attacking teammates closer to their net.
      Fallback — just toward the net.
    Returns (vx, vy) unit vector.
    """
    # Direction toward the net the DEFENDERS protect = net attacked by offense
    # If target is 'home' side, they protect home_net (+89, 0)
    # Their threat = away team attacking toward home net, so vector toward home net
    to_net_x = net_x - target_x
    to_net_y = net_y - target_y

    if THREAT_DIRECTION_MODE == "full" and len(teammates) > 0:
        # Teammates closer to the defended net than the target
        if target_side == "home":
            closer = teammates[teammates["x"] > target_x]
        else:
            closer = teammates[teammates["x"] < target_x]

        if len(closer):
            # Sum unit vectors toward each such teammate + toward net
            vx, vy = to_net_x, to_net_y
            for _, tm in closer.iterrows():
                dx = tm["x"] - target_x
                dy = tm["y"] - target_y
                d  = np.sqrt(dx**2 + dy**2)
                if d > 1e-6:
                    vx += dx / d
                    vy += dy / d
            return vx, vy

    return to_net_x, to_net_y


def compute_frame_pressure_fast(frame_df: pd.DataFrame,
                                  home_net, away_net) -> dict:
    """
    Compute pressure for every player in a frame.
    Returns dict: player_id → total pressure received (0–100 scale, capped).

    home_net = (HOME_NET_X, HOME_NET_Y)  — the net home team defends
    away_net = (AWAY_NET_X, AWAY_NET_Y)  — the net away team defends
    """
    players = frame_df[frame_df["is_player"] & frame_df["x"].notna()].copy()
    if players.empty:
        return {}

    pressure_map = {}

    for _, target in players.iterrows():
        tid  = target["player_id"]
        tx   = target["x"]; ty = target["y"]
        side = target["team_side"]

        # Pressers = opponents
        opp_side = "away" if side == "home" else "home"
        pressers = players[players["team_side"] == opp_side]
        teammates = players[(players["team_side"] == side) & (players["player_id"] != tid)]

        # Which net does this target defend?
        net_x, net_y = (HOME_NET_X, HOME_NET_Y) if side == "home" else (AWAY_NET_X, AWAY_NET_Y)

        tvx, tvy = compute_threat_vector(tx, ty, side, teammates, net_x, net_y)

        total_pr = 0.0
        for _, pr in pressers.iterrows():
            total_pr += single_pressure(pr["x"], pr["y"], tx, ty, tvx, tvy)

        pressure_map[tid] = min(total_pr, 100.0)

    return pressure_map


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRESSURE AGGREGATION & NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_pressure_timeline(game: GameData, sample_every: int = 1) -> pd.DataFrame:
    """
    Compute per-second team pressure throughout the game.
    Samples one frame per second (the first frame found at that elapsed_sec).

    Returns DataFrame with columns:
      elapsed_sec, period, clock, home_raw, away_raw,
      home_skaters, away_skaters, home_per_pair, away_per_pair,
      is_clean (5v5), play_segment
    """
    print("Building pressure timeline...")
    records = []

    # Get all unique seconds
    all_secs = sorted(game.tracking["elapsed_sec"].unique())
    all_secs = [s for s in all_secs if s % sample_every == 0]

    home_net = (HOME_NET_X, HOME_NET_Y)
    away_net = (AWAY_NET_X, AWAY_NET_Y)

    for i, sec in enumerate(all_secs):
        if i % 200 == 0:
            print(f"  {i}/{len(all_secs)} seconds processed...", end="\r")

        frame_rows = game.tracking[game.tracking["elapsed_sec"] == sec]
        if frame_rows.empty:
            continue

        # Take the first frame at this second
        frame_id = frame_rows["frame_id"].iloc[0]
        frame    = game.get_frame(frame_id)

        period = int(frame_rows["Period"].iloc[0])
        clock  = frame_rows["Game Clock"].iloc[0]

        pm = compute_frame_pressure_fast(frame, home_net, away_net)

        players = frame[frame["is_player"] & frame["x"].notna()]
        home_pl = players[(players["team_side"] == "home") & ~players["is_goalie"]]
        away_pl = players[(players["team_side"] == "away") & ~players["is_goalie"]]

        home_skaters = len(home_pl)
        away_skaters = len(away_pl)

        # Raw pressure = mean pressure on each team's skaters
        home_ids = set(home_pl["player_id"].tolist())
        away_ids = set(away_pl["player_id"].tolist())

        home_pressures = [pm.get(pid, 0) for pid in home_ids]
        away_pressures = [pm.get(pid, 0) for pid in away_ids]

        home_raw = np.mean(home_pressures) if home_pressures else 0.0
        away_raw = np.mean(away_pressures) if away_pressures else 0.0

        # Per-pair normalisation: divide by number of active matchup pairs
        pairs = min(home_skaters, away_skaters) if min(home_skaters, away_skaters) > 0 else 1
        home_per_pair = home_raw / pairs
        away_per_pair = away_raw / pairs

        is_clean = (home_skaters == 5 and away_skaters == 5)

        records.append({
            "elapsed_sec":  sec,
            "period":       period,
            "clock":        clock,
            "home_raw":     home_raw,
            "away_raw":     away_raw,
            "home_skaters": home_skaters,
            "away_skaters": away_skaters,
            "home_per_pair": home_per_pair,
            "away_per_pair": away_per_pair,
            "is_clean":     is_clean,
        })

    print(f"\n  Done. {len(records)} seconds computed.")
    pt = pd.DataFrame(records)

    # Mark play segments (stoppages break the sequence)
    # A stoppage is a gap of >2 seconds in elapsed_sec
    pt = pt.sort_values("elapsed_sec").reset_index(drop=True)
    seg = 0
    segs = [0]
    for i in range(1, len(pt)):
        if pt.loc[i, "elapsed_sec"] - pt.loc[i-1, "elapsed_sec"] > 2:
            seg += 1
        segs.append(seg)
    pt["play_segment"] = segs

    # Smooth
    pt["home_smooth"] = pt["home_per_pair"].rolling(PRESSURE_SMOOTH_SEC, center=True, min_periods=1).mean()
    pt["away_smooth"] = pt["away_per_pair"].rolling(PRESSURE_SMOOTH_SEC, center=True, min_periods=1).mean()
    pt["net_smooth"]  = pt["home_smooth"] - pt["away_smooth"]

    return pt


def build_player_pressure_leaderboard(game: GameData, pt: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player pressure statistics across the game.
    Returns DataFrame with player_id, team_side, jersey, mean_pressure, etc.
    """
    print("Building player leaderboard...")
    all_secs = sorted(pt["elapsed_sec"].unique())
    records = []

    for sec in all_secs[::5]:  # sample every 5s for speed
        frame_rows = game.tracking[game.tracking["elapsed_sec"] == sec]
        if frame_rows.empty: continue
        frame_id = frame_rows["frame_id"].iloc[0]
        frame    = game.get_frame(frame_id)

        pm = compute_frame_pressure_fast(frame, (HOME_NET_X, HOME_NET_Y), (AWAY_NET_X, AWAY_NET_Y))
        players = frame[frame["is_player"] & frame["x"].notna() & ~frame["is_goalie"]]

        for _, row in players.iterrows():
            records.append({
                "elapsed_sec": sec,
                "player_id":   row["player_id"],
                "jersey":      row["jersey"],
                "team_side":   row["team_side"],
                "team":        row["Team"],
                "pressure":    pm.get(row["player_id"], 0),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    board = df.groupby(["player_id", "jersey", "team_side", "team"]).agg(
        mean_pressure = ("pressure", "mean"),
        max_pressure  = ("pressure", "max"),
        frames_above_threshold = ("pressure", lambda x: (x >= PRESSURE_THRESHOLD).sum()),
        total_frames  = ("pressure", "count"),
    ).reset_index()
    board["pct_under_pressure"] = board["frames_above_threshold"] / board["total_frames"] * 100
    board = board.sort_values("mean_pressure", ascending=False).reset_index(drop=True)
    return board


def build_spatial_heatmap(game: GameData, pt: pd.DataFrame,
                           grid_x=40, grid_y=20) -> dict:
    """
    Build 2D pressure heatmaps for home and away teams.
    Returns dict with 'home', 'away', 'net' arrays shaped (grid_y, grid_x).
    """
    print("Building spatial heatmap...")
    home_acc = np.zeros((grid_y, grid_x))
    away_acc = np.zeros((grid_y, grid_x))
    counts   = np.zeros((grid_y, grid_x))

    x_edges = np.linspace(-100, 100, grid_x + 1)
    y_edges = np.linspace(-42.5, 42.5, grid_y + 1)

    all_secs = sorted(pt["elapsed_sec"].unique())

    for sec in all_secs[::3]:
        frame_rows = game.tracking[game.tracking["elapsed_sec"] == sec]
        if frame_rows.empty: continue
        frame_id = frame_rows["frame_id"].iloc[0]
        frame    = game.get_frame(frame_id)

        pm = compute_frame_pressure_fast(frame, (HOME_NET_X, HOME_NET_Y), (AWAY_NET_X, AWAY_NET_Y))
        players = frame[frame["is_player"] & frame["x"].notna() & ~frame["is_goalie"]]

        for _, row in players.iterrows():
            xi = np.searchsorted(x_edges, row["x"], side="right") - 1
            yi = np.searchsorted(y_edges, row["y"], side="right") - 1
            xi = np.clip(xi, 0, grid_x - 1)
            yi = np.clip(yi, 0, grid_y - 1)

            pr = pm.get(row["player_id"], 0)
            if row["team_side"] == "home":
                home_acc[yi, xi] += pr
            else:
                away_acc[yi, xi] += pr
            counts[yi, xi] += 1

    safe = np.where(counts > 0, counts, 1)
    home_norm = gaussian_filter(home_acc / safe, sigma=1.5)
    away_norm = gaussian_filter(away_acc / safe, sigma=1.5)

    return {
        "home": home_norm,
        "away": away_norm,
        "net":  home_norm - away_norm,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PRESSING SEQUENCES & ANDRIENKO STATS A–D
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pressing_sequences(pt: pd.DataFrame,
                                min_duration: int = 5,
                                percentile: float = 65) -> pd.DataFrame:
    """
    Detect sustained pressing windows (Andrienko pressing sequences).
    Returns DataFrame: team, start, end, duration, mean_pressure, peak_pressure, segment
    """
    clean  = pt[pt["is_clean"]].copy()
    thresh = clean["home_per_pair"].quantile(percentile / 100)

    def find_seqs(col, df, threshold, min_dur):
        seqs = []
        above = df[df[col] >= threshold].copy()
        if above.empty: return seqs

        run_start = None; run_end = None; run_seg = None
        prev_sec  = None

        for _, row in above.iterrows():
            if run_start is None:
                run_start = row["elapsed_sec"]
                run_seg   = row["play_segment"]
                run_end   = row["elapsed_sec"]
            elif (row["elapsed_sec"] - run_end <= 2 and
                  row["play_segment"] == run_seg):
                run_end = row["elapsed_sec"]
            else:
                dur = run_end - run_start
                if dur >= min_dur:
                    window = df[(df["elapsed_sec"] >= run_start) &
                                (df["elapsed_sec"] <= run_end)]
                    seqs.append({
                        "start": run_start, "end": run_end,
                        "duration": dur,
                        "mean_pressure": window[col].mean(),
                        "peak_pressure": window[col].max(),
                        "segment": run_seg,
                    })
                run_start = row["elapsed_sec"]
                run_seg   = row["play_segment"]
                run_end   = row["elapsed_sec"]
            prev_sec = row["elapsed_sec"]

        if run_start and (run_end - run_start) >= min_dur:
            window = df[(df["elapsed_sec"] >= run_start) &
                        (df["elapsed_sec"] <= run_end)]
            seqs.append({
                "start": run_start, "end": run_end,
                "duration": run_end - run_start,
                "mean_pressure": window[col].mean(),
                "peak_pressure": window[col].max(),
                "segment": run_seg,
            })
        return seqs

    home_seqs = [{"team":"home", **s} for s in find_seqs("home_per_pair", clean, thresh, min_duration)]
    away_seqs = [{"team":"away", **s} for s in find_seqs("away_per_pair", clean, thresh, min_duration)]
    return pd.DataFrame(home_seqs + away_seqs)


def compute_andrienko_stats(pt: pd.DataFrame, game: GameData) -> dict:
    """
    Compute Andrienko et al. (2017) pressing statistics A–D.

    A — pressing sequences: duration, mean/peak pressure
    B — counter-pressing (recovery pressure within 8s of turnover)
    C — pre-turnover pressure (pressure on carrier 10s before turnover)
    D — spatial pressing tendency (where does each team press most?)

    Returns dict of DataFrames.
    """
    stats = {}

    # A — pressing sequences
    stats["A_sequences"] = compute_pressing_sequences(pt)

    # B — counter-pressing
    turnovers = game.events[game.events["Event"].isin(["Turnover", "Takeaway"])].copy()
    counter_records = []
    for _, tv in turnovers.iterrows():
        tv_sec  = tv["elapsed_sec"]
        tv_side = tv["team_side"]    # team that LOST the puck
        # Recovering team = the one that lost it; they now press to win it back
        window  = pt[(pt["elapsed_sec"] >= tv_sec) &
                     (pt["elapsed_sec"] <= tv_sec + 8) &
                     pt["is_clean"]]
        if window.empty: continue
        col = "home_per_pair" if tv_side == "home" else "away_per_pair"
        counter_records.append({
            "elapsed_sec":   tv_sec,
            "recovering_side": tv_side,
            "mean_counter_pressure": window[col].mean(),
            "peak_counter_pressure": window[col].max(),
            "duration_above_thresh": (window[col] >= PRESSURE_THRESHOLD).sum(),
        })
    stats["B_counter"] = pd.DataFrame(counter_records)

    # C — pre-turnover pressure
    pre_records = []
    for _, tv in turnovers.iterrows():
        tv_sec    = tv["elapsed_sec"]
        carrier_side = "away" if tv["team_side"] == "home" else "home"  # carrier = opposite
        window    = pt[(pt["elapsed_sec"] >= tv_sec - 10) &
                       (pt["elapsed_sec"] <= tv_sec) &
                       pt["is_clean"]]
        if window.empty: continue
        col = "home_per_pair" if carrier_side == "home" else "away_per_pair"
        pre_records.append({
            "elapsed_sec": tv_sec,
            "carrier_side": carrier_side,
            "mean_pre_pressure": window[col].mean(),
            "peak_pre_pressure": window[col].max(),
        })
    stats["C_pre_turnover"] = pd.DataFrame(pre_records)

    # D — spatial: median pressure by zone (DZ / NZ / OZ)
    # Classify each row's dominant player location into a zone
    zone_records = []
    for _, row in pt[pt["is_clean"]].iterrows():
        sec = row["elapsed_sec"]
        frame_rows = game.tracking[game.tracking["elapsed_sec"] == sec]
        if frame_rows.empty: continue
        home_pl = frame_rows[(frame_rows["team_side"] == "home") &
                             (frame_rows["is_player"]) & ~frame_rows["is_goalie"]]
        away_pl = frame_rows[(frame_rows["team_side"] == "away") &
                             (frame_rows["is_player"]) & ~frame_rows["is_goalie"]]
        if home_pl.empty or away_pl.empty: continue

        hx = home_pl["x"].mean(); ax = away_pl["x"].mean()
        zone_records.append({
            "elapsed_sec": sec,
            "home_mean_x": hx,
            "away_mean_x": ax,
            "home_zone": "OZ" if hx > BLUE_LINE_HOME else ("DZ" if hx < BLUE_LINE_AWAY else "NZ"),
            "away_zone": "OZ" if ax < BLUE_LINE_AWAY else ("DZ" if ax > BLUE_LINE_HOME else "NZ"),
            "home_per_pair": row["home_per_pair"],
            "away_per_pair": row["away_per_pair"],
        })
    stats["D_spatial"] = pd.DataFrame(zone_records)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FACEOFF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_faceoff_formations(game: GameData) -> pd.DataFrame:
    """
    For each faceoff event, extract player positions relative to the dot.
    Returns DataFrame: one row per faceoff with flattened position features.
    """
    print("Extracting faceoff formations...")
    faceoffs = game.get_faceoffs()
    records  = []

    for _, fo in faceoffs.iterrows():
        sec  = fo["elapsed_sec"]
        dot_x = fo.get("X_Coordinate", 0) or 0
        dot_y = fo.get("Y_Coordinate", 0) or 0

        frame_rows = game.tracking[game.tracking["elapsed_sec"] == sec]
        if frame_rows.empty: continue
        frame_id = frame_rows["frame_id"].iloc[0]
        frame    = game.get_frame(frame_id)

        home_pl = game.get_players(frame, "home")
        away_pl = game.get_players(frame, "away")
        if home_pl.empty or away_pl.empty: continue

        def relative_positions(players, dot_x, dot_y, n=6):
            """Return list of (dx, dy) sorted by distance to dot, padded/truncated to n."""
            positions = []
            for _, p in players.iterrows():
                dx = p["x"] - dot_x
                dy = p["y"] - dot_y
                positions.append((dx, dy))
            positions.sort(key=lambda t: t[0]**2 + t[1]**2)
            while len(positions) < n:
                positions.append((0.0, 0.0))
            return positions[:n]

        h_pos = relative_positions(home_pl, dot_x, dot_y)
        a_pos = relative_positions(away_pl, dot_x, dot_y)

        # Determine zone
        if dot_x > BLUE_LINE_HOME:
            zone = "home_OZ"
        elif dot_x < BLUE_LINE_AWAY:
            zone = "away_OZ"
        elif abs(dot_x) < 5:
            zone = "centre"
        else:
            zone = "neutral"

        # Winner
        winner_side = fo.get("team_side", "unknown")

        row = {
            "elapsed_sec": sec,
            "dot_x": dot_x, "dot_y": dot_y,
            "zone": zone,
            "winner": winner_side,
            "home_skaters": fo.get("Home_Team_Skaters", 5),
            "away_skaters": fo.get("Away_Team_Skaters", 5),
        }
        for i, (dx, dy) in enumerate(h_pos):
            row[f"h{i}_dx"] = dx; row[f"h{i}_dy"] = dy
        for i, (dx, dy) in enumerate(a_pos):
            row[f"a{i}_dx"] = dx; row[f"a{i}_dy"] = dy

        records.append(row)

    return pd.DataFrame(records)


def cluster_faceoff_formations(formations_df: pd.DataFrame,
                                n_clusters: int = 5) -> pd.DataFrame:
    """
    K-Means cluster faceoff formations.
    Adds 'cluster' column to formations_df.
    """
    feat_cols = [c for c in formations_df.columns if c.endswith("_dx") or c.endswith("_dy")]
    X = formations_df[feat_cols].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    formations_df = formations_df.copy()
    formations_df["cluster"] = km.fit_predict(Xs)
    return formations_df


def faceoff_win_probability_model(formations_df: pd.DataFrame) -> dict:
    """
    Logistic regression: predict faceoff winner from cluster + zone + skater advantage.
    Returns dict with model, accuracy, feature_importance.
    """
    df = formations_df.copy()
    df["skater_advantage"] = df["home_skaters"] - df["away_skaters"]
    df["target"] = (df["winner"] == "home").astype(int)

    X_df = pd.get_dummies(df[["cluster", "zone", "skater_advantage"]], drop_first=True)
    y    = df["target"]

    if len(y.unique()) < 2 or len(y) < 10:
        return {"model": None, "accuracy": None, "X_df": X_df}

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  Faceoff win model accuracy: {acc:.1%}")

    importance = pd.Series(np.abs(clf.coef_[0]), index=X_df.columns).sort_values(ascending=False)
    return {"model": clf, "accuracy": acc, "importance": importance, "X_df": X_df, "y": y}


def compute_forecheck_linkage(game: GameData, formations_df: pd.DataFrame,
                               pt: pd.DataFrame) -> pd.DataFrame:
    """
    For each faceoff, compute offensive zone pressure in FORECHECK_WINDOW_SEC seconds after.
    Merges back into formations_df.
    """
    records = []
    for _, fo in formations_df.iterrows():
        sec      = fo["elapsed_sec"]
        home_won = (fo["winner"] == "home")
        window   = pt[(pt["elapsed_sec"] >= sec) &
                      (pt["elapsed_sec"] <= sec + FORECHECK_WINDOW_SEC)]
        if window.empty:
            records.append({"elapsed_sec": sec, "forecheck_pressure": np.nan})
            continue
        # Forecheck pressure = offensive team's pressure in their OZ window
        col = "home_per_pair" if home_won else "away_per_pair"
        records.append({
            "elapsed_sec": sec,
            "forecheck_pressure": window[col].mean(),
        })
    fc_df = pd.DataFrame(records)
    return formations_df.merge(fc_df, on="elapsed_sec", how="left")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FORECHECK ANALYSIS (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_forecheck_stats(game: GameData, pt: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forecheck pressure windows: 5 seconds after the puck enters the OZ.
    Returns DataFrame of forecheck events.
    """
    # Detect OZ entry as: puck crosses blue line
    puck = game.tracking[game.tracking["is_puck"] & game.tracking["x"].notna()].copy()
    puck = puck.sort_values("elapsed_sec")
    puck["prev_x"] = puck["x"].shift(1)
    puck["prev_sec"] = puck["elapsed_sec"].shift(1)

    entries = []
    for _, row in puck.iterrows():
        if pd.isna(row["prev_x"]): continue
        # Home OZ entry = puck crosses from <25 to >25
        if row["prev_x"] < BLUE_LINE_HOME <= row["x"]:
            entries.append({"elapsed_sec": row["elapsed_sec"], "zone": "home_OZ"})
        # Away OZ entry
        elif row["prev_x"] > BLUE_LINE_AWAY >= row["x"]:
            entries.append({"elapsed_sec": row["elapsed_sec"], "zone": "away_OZ"})

    records = []
    for e in entries:
        sec  = e["elapsed_sec"]
        zone = e["zone"]
        window = pt[(pt["elapsed_sec"] >= sec) & (pt["elapsed_sec"] <= sec + 5)]
        if window.empty: continue
        attacking_col = "home_per_pair" if zone == "home_OZ" else "away_per_pair"
        records.append({
            "elapsed_sec":      sec,
            "zone":             zone,
            "forecheck_pressure": window[attacking_col].mean(),
            "duration_in_zone": len(window),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ALL PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Rink drawing helper ────────────────────────────────────────────────────────

def draw_rink(ax, alpha=0.15, lw=1.2):
    """Draw a simplified NHL rink outline on ax."""
    # Boards
    rect = plt.Rectangle((-100, -42.5), 200, 85, fill=False,
                          edgecolor='#aaa', linewidth=lw*1.5, zorder=0)
    ax.add_patch(rect)
    # Blue lines
    for x in [BLUE_LINE_AWAY, BLUE_LINE_HOME]:
        ax.axvline(x, color='#4466cc', linewidth=lw*1.8, alpha=0.6, zorder=1)
    # Red line
    ax.axvline(0, color='#cc3333', linewidth=lw*1.5, alpha=0.5, zorder=1)
    # Goal creases
    for nx in [HOME_NET_X, AWAY_NET_X]:
        sign = 1 if nx > 0 else -1
        crease = Ellipse((nx + sign*4, 0), width=8, height=12,
                         fill=True, facecolor='#cce0ff', edgecolor='#4488cc',
                         linewidth=lw, alpha=0.4, zorder=2)
        ax.add_patch(crease)
    # Faceoff dots
    for name, (dx, dy) in FACEOFF_DOTS.items():
        ax.plot(dx, dy, 'o', color='#cc3333', markersize=3, alpha=alpha*3, zorder=2)
    ax.set_xlim(-105, 105); ax.set_ylim(-47, 47)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8faff')


# ── Plot 1: Zone pressure timeline ────────────────────────────────────────────

def plot_zone_pressure_timeline(pt: pd.DataFrame, game: GameData,
                                  save_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 5)); fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.fill_between(pt["elapsed_sec"], pt["home_smooth"], alpha=0.25, color='#1a7abf', label='Home pressure')
    ax.fill_between(pt["elapsed_sec"], pt["away_smooth"], alpha=0.25, color='#cc3333', label='Away pressure')
    ax.plot(pt["elapsed_sec"], pt["home_smooth"], color='#1a7abf', linewidth=1.2)
    ax.plot(pt["elapsed_sec"], pt["away_smooth"], color='#cc3333', linewidth=1.2)

    # Period dividers
    for p_end in [1200, 2400]:
        ax.axvline(p_end, color='#888', linewidth=1, linestyle='--', alpha=0.5)
        ax.text(p_end + 20, ax.get_ylim()[1]*0.9, f'P{p_end//1200+1}',
                color='#888', fontsize=8)

    # Goal markers
    for _, goal in game.get_goals().iterrows():
        color = '#1a7abf' if goal["team_side"] == "home" else '#cc3333'
        ax.axvline(goal["elapsed_sec"], color=color, linewidth=1.5, alpha=0.7, linestyle=':')

    ax.set_xlabel("Game time (seconds)", fontsize=10)
    ax.set_ylabel("Pressure (per-pair, smoothed)", fontsize=10)
    ax.set_title("Team Pressure Timeline", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, color='#eee', linewidth=0.5)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/zone_pressure_timeline_normalised.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 2: Player pressure leaderboard ───────────────────────────────────────

def plot_player_leaderboard(board: pd.DataFrame, save_path: str = None):
    top = board.head(20)
    fig, ax = plt.subplots(figsize=(10, 8)); fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = ['#1a7abf' if s == 'home' else '#cc3333' for s in top['team_side']]
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top['mean_pressure'], color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y_pos)
    labels = [f"{row['team']} #{row['jersey']}" for _, row in top.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean pressure received", fontsize=10)
    ax.set_title("Top 20 Players by Mean Pressure Received", fontsize=12)
    ax.grid(True, axis='x', color='#eee', linewidth=0.5)

    home_p = mpatches.Patch(color='#1a7abf', label='Home')
    away_p = mpatches.Patch(color='#cc3333', label='Away')
    ax.legend(handles=[home_p, away_p], fontsize=9)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/player_pressure_leaderboard.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 3: Full-ice pressure density ─────────────────────────────────────────

def plot_pressure_density(heatmaps: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)); fig.patch.set_facecolor('white')

    for ax, key, title, cmap in zip(
        axes,
        ['home', 'away', 'net'],
        ['Home Team Pressure', 'Away Team Pressure', 'Net Pressure (Home − Away)'],
        ['Blues', 'Reds', 'RdBu_r'],
    ):
        data = heatmaps[key]
        im = ax.imshow(data, origin='lower', aspect='auto',
                       extent=[-100, 100, -42.5, 42.5],
                       cmap=cmap, alpha=0.85)
        # Overlay rink lines
        for x in [BLUE_LINE_AWAY, BLUE_LINE_HOME]:
            ax.axvline(x, color='#4466cc', linewidth=1.2, alpha=0.5)
        ax.axvline(0, color='#cc3333', linewidth=1.0, alpha=0.4)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Rink X (ft)"); ax.set_ylabel("Rink Y (ft)")

    fig.suptitle("Full-Ice Pressure Density (normalised per visit)", fontsize=13)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/pressure_density_full_ice.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 4: Pressing sequences ────────────────────────────────────────────────

def plot_pressing_sequences(pt: pd.DataFrame, sequences_df: pd.DataFrame,
                             save_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 5)); fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(pt["elapsed_sec"], pt["home_smooth"], color='#1a7abf', linewidth=1.0, label='Home')
    ax.plot(pt["elapsed_sec"], pt["away_smooth"], color='#cc3333', linewidth=1.0, label='Away')

    for _, seq in sequences_df.iterrows():
        color = '#1a7abf' if seq['team'] == 'home' else '#cc3333'
        ax.axvspan(seq['start'], seq['end'], alpha=0.18, color=color)

    for p_end in [1200, 2400]:
        ax.axvline(p_end, color='#aaa', linewidth=1, linestyle='--', alpha=0.5)

    ax.set_xlabel("Game time (seconds)", fontsize=10)
    ax.set_ylabel("Pressure (per-pair)", fontsize=10)
    ax.set_title("Pressing Sequences (shaded = sustained pressing window)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, color='#eee', linewidth=0.5)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/pressing_sequences_normalised.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 5: Andrienko stats (4-panel) ─────────────────────────────────────────

def plot_andrienko_stats(stats: dict, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)); fig.patch.set_facecolor('white')

    # A — sequence duration distribution
    ax = axes[0, 0]
    if not stats["A_sequences"].empty:
        for side, color in [("home", '#1a7abf'), ("away", '#cc3333')]:
            s = stats["A_sequences"][stats["A_sequences"]["team"] == side]
            ax.hist(s["duration"], bins=10, color=color, alpha=0.6, label=side.capitalize())
        ax.set_title("A — Pressing sequence duration (s)", fontsize=11)
        ax.set_xlabel("Duration (s)"); ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    # B — counter-pressing
    ax = axes[0, 1]
    if not stats["B_counter"].empty:
        for side, color in [("home", '#1a7abf'), ("away", '#cc3333')]:
            s = stats["B_counter"][stats["B_counter"]["recovering_side"] == side]
            ax.scatter(s.index, s["mean_counter_pressure"], color=color, alpha=0.6,
                       s=30, label=side.capitalize())
        ax.set_title("B — Counter-pressing intensity post-turnover", fontsize=11)
        ax.set_xlabel("Turnover index"); ax.set_ylabel("Mean pressure (8s after)")
        ax.legend(fontsize=9)

    # C — pre-turnover pressure
    ax = axes[1, 0]
    if not stats["C_pre_turnover"].empty:
        for side, color in [("home", '#1a7abf'), ("away", '#cc3333')]:
            s = stats["C_pre_turnover"][stats["C_pre_turnover"]["carrier_side"] == side]
            ax.scatter(s.index, s["mean_pre_pressure"], color=color, alpha=0.6,
                       s=30, label=side.capitalize())
        ax.axhline(PRESSURE_THRESHOLD, color='#888', linewidth=1, linestyle='--', alpha=0.6)
        ax.set_title("C — Pressure on carrier before turnover (10s window)", fontsize=11)
        ax.set_xlabel("Turnover index"); ax.set_ylabel("Mean pre-turnover pressure")
        ax.legend(fontsize=9)

    # D — spatial (zone breakdown)
    ax = axes[1, 1]
    if not stats["D_spatial"].empty:
        zone_summary = stats["D_spatial"].groupby("home_zone").agg(
            home_press = ("home_per_pair", "mean"),
            away_press = ("away_per_pair", "mean"),
        ).reset_index()
        x = np.arange(len(zone_summary))
        ax.bar(x - 0.2, zone_summary["home_press"], 0.35, color='#1a7abf', alpha=0.8, label='Home')
        ax.bar(x + 0.2, zone_summary["away_press"], 0.35, color='#cc3333', alpha=0.8, label='Away')
        ax.set_xticks(x); ax.set_xticklabels(zone_summary["home_zone"])
        ax.set_title("D — Spatial pressing tendency by zone", fontsize=11)
        ax.set_ylabel("Mean pressure"); ax.legend(fontsize=9)

    for ax in axes.flat:
        ax.set_facecolor('white')
        ax.grid(True, color='#eee', linewidth=0.5)

    fig.suptitle("Andrienko Pressing Statistics A–D", fontsize=13)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/andrienko_pressing_stats_normalised.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 6: Pressure + shot heatmap ───────────────────────────────────────────

def plot_pressure_shot_heatmap(heatmaps: dict, game: GameData,
                                save_path: str = None):
    shots = game.events[game.events["Event"].isin(["Shot", "Shot on Goal", "Goal"])].copy()
    shots = shots[shots["X_Coordinate"].notna() & shots["Y_Coordinate"].notna()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10)); fig.patch.set_facecolor('white')
    titles = [
        ("Home OZ pressure", "home", None),
        ("Home OZ shots", None, "home"),
        ("Away OZ pressure", "away", None),
        ("Away OZ shots", None, "away"),
    ]

    for ax, (title, press_side, shot_side) in zip(axes.flat, titles):
        ax.set_facecolor('#f8faff')
        draw_rink(ax)

        if press_side:
            data = heatmaps[press_side]
            ax.imshow(data, origin='lower', aspect='auto',
                      extent=[-100, 100, -42.5, 42.5],
                      cmap='Blues' if press_side == 'home' else 'Reds',
                      alpha=0.6)

        if shot_side:
            s = shots[shots["team_side"] == shot_side]
            color = '#1a7abf' if shot_side == 'home' else '#cc3333'
            ax.scatter(s["X_Coordinate"], s["Y_Coordinate"],
                       color=color, alpha=0.6, s=20, zorder=4)

        ax.set_title(title, fontsize=11)

    fig.suptitle("Pressure + Shot Heatmap", fontsize=13)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/pressure_shot_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 7: Faceoff macro analysis (4-panel) ──────────────────────────────────

def plot_faceoff_pressure_macro(formations_df: pd.DataFrame,
                                 save_path: str = None):
    if formations_df.empty or "forecheck_pressure" not in formations_df.columns:
        print("  Skipping faceoff macro (no data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10)); fig.patch.set_facecolor('white')

    # Win rate by cluster
    ax = axes[0, 0]
    win_rates = formations_df.groupby("cluster").apply(
        lambda g: (g["winner"] == "home").mean()
    ).reset_index(); win_rates.columns = ["cluster", "win_rate"]
    ax.bar(win_rates["cluster"], win_rates["win_rate"], color='#1a7abf', alpha=0.8)
    ax.axhline(0.5, color='#cc3333', linewidth=1, linestyle='--')
    ax.set_title("Home win rate by formation cluster", fontsize=11)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Home win rate")

    # Forecheck pressure by winner
    ax = axes[0, 1]
    for side, color in [("home", '#1a7abf'), ("away", '#cc3333')]:
        sub = formations_df[formations_df["winner"] == side]["forecheck_pressure"].dropna()
        ax.hist(sub, bins=12, color=color, alpha=0.6, label=side.capitalize())
    ax.set_title("Forecheck pressure distribution by winner", fontsize=11)
    ax.set_xlabel("Forecheck pressure (5s post-faceoff)")
    ax.legend(fontsize=9)

    # Forecheck by zone
    ax = axes[1, 0]
    zone_means = formations_df.groupby("zone")["forecheck_pressure"].mean().reset_index()
    ax.bar(zone_means["zone"], zone_means["forecheck_pressure"], color='#4a9e6b', alpha=0.8)
    ax.set_title("Mean forecheck pressure by faceoff zone", fontsize=11)
    ax.set_ylabel("Mean pressure")

    # Win rate by zone
    ax = axes[1, 1]
    zone_win = formations_df.groupby("zone").apply(
        lambda g: (g["winner"] == "home").mean()
    ).reset_index(); zone_win.columns = ["zone", "win_rate"]
    ax.bar(zone_win["zone"], zone_win["win_rate"], color='#8844aa', alpha=0.8)
    ax.axhline(0.5, color='#cc3333', linewidth=1, linestyle='--')
    ax.set_title("Home win rate by faceoff zone", fontsize=11)
    ax.set_ylabel("Home win rate")

    for ax in axes.flat:
        ax.set_facecolor('white')
        ax.grid(True, color='#eee', linewidth=0.5)

    fig.suptitle("Post-Faceoff Pressure Macro Analysis", fontsize=13)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/faceoff_pressure_macro.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 8: Faceoff formation ice map ─────────────────────────────────────────

def plot_faceoff_formation_ice_map(formations_df: pd.DataFrame,
                                    save_path: str = None):
    if formations_df.empty or "cluster" not in formations_df.columns:
        print("  Skipping formation map (no data)")
        return

    n = formations_df["cluster"].nunique()
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5)); fig.patch.set_facecolor('white')
    if n == 1: axes = [axes]

    for i, ax in enumerate(axes):
        cluster_df = formations_df[formations_df["cluster"] == i]
        draw_rink(ax)

        for _, row in cluster_df.iterrows():
            dot_x, dot_y = row["dot_x"], row["dot_y"]
            for j in range(6):
                hx = dot_x + row.get(f"h{j}_dx", 0)
                hy = dot_y + row.get(f"h{j}_dy", 0)
                ax.plot(hx, hy, 'o', color='#1a7abf', markersize=6, alpha=0.4, zorder=3)
                ax2x = dot_x + row.get(f"a{j}_dx", 0)
                a2y  = dot_y + row.get(f"a{j}_dy", 0)
                ax.plot(ax2x, a2y, 's', color='#cc3333', markersize=6, alpha=0.4, zorder=3)

        n_fo = len(cluster_df)
        home_wins = (cluster_df["winner"] == "home").sum()
        ax.set_title(f"Cluster {i}\n{n_fo} faceoffs  |  Home win: {home_wins/max(n_fo,1):.0%}",
                     fontsize=10)

    fig.suptitle("Faceoff Formation Clusters (blue=home, red=away)", fontsize=12)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/faceoff_formation_ice_map.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 9: Faceoff win probability ───────────────────────────────────────────

def plot_faceoff_win_prob(formations_df: pd.DataFrame, model_result: dict,
                           save_path: str = None):
    if not model_result.get("model"):
        print("  Skipping win prob plot (no model)")
        return

    fig, ax = plt.subplots(figsize=(8, 5)); fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    imp = model_result["importance"].head(10)
    ax.barh(imp.index[::-1], imp.values[::-1], color='#4a9e6b', alpha=0.8)
    ax.set_title(f"Faceoff Win Probability Model\nAccuracy: {model_result['accuracy']:.1%}",
                 fontsize=12)
    ax.set_xlabel("Feature importance (|coefficient|)")
    ax.grid(True, axis='x', color='#eee', linewidth=0.5)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/faceoff_win_prob.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 10: Forecheck analysis ───────────────────────────────────────────────

def plot_forecheck_analysis(fc_df: pd.DataFrame, save_path: str = None):
    if fc_df.empty:
        print("  Skipping forecheck plot (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); fig.patch.set_facecolor('white')

    ax = axes[0]
    for zone, color in [("home_OZ", '#1a7abf'), ("away_OZ", '#cc3333')]:
        s = fc_df[fc_df["zone"] == zone]["forecheck_pressure"].dropna()
        ax.hist(s, bins=12, color=color, alpha=0.65, label=zone)
    ax.set_title("Forecheck pressure distribution by OZ", fontsize=11)
    ax.set_xlabel("Mean pressure (5s after OZ entry)"); ax.legend(fontsize=9)

    ax = axes[1]
    zone_means = fc_df.groupby("zone")["forecheck_pressure"].mean().reset_index()
    ax.bar(zone_means["zone"], zone_means["forecheck_pressure"],
           color=['#1a7abf', '#cc3333'][:len(zone_means)], alpha=0.8)
    ax.set_title("Mean forecheck pressure by zone", fontsize=11)
    ax.set_ylabel("Mean pressure")

    for ax in axes:
        ax.set_facecolor('white')
        ax.grid(True, color='#eee', linewidth=0.5)

    fig.suptitle("Forecheck Analysis", fontsize=13)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/forecheck_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 11: Pressure–events linkage ─────────────────────────────────────────

def plot_pressure_events_linkage(pt: pd.DataFrame, game: GameData,
                                  save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5)); fig.patch.set_facecolor('white')

    events_to_check = ["Shot on Goal", "Goal", "Penalty"]
    event_colors    = {'Shot on Goal': '#ff8c00', 'Goal': '#cc3333', 'Penalty': '#7b00d4'}

    # Panel 1: pressure around events
    ax = axes[0]
    ax.set_facecolor('white')
    window_size = 10
    for etype, color in event_colors.items():
        evs = game.events[game.events["Event"] == etype]
        pre_press = []
        for _, ev in evs.iterrows():
            sec = ev["elapsed_sec"]
            w   = pt[(pt["elapsed_sec"] >= sec - window_size) &
                     (pt["elapsed_sec"] <= sec)]
            if not w.empty:
                col = "home_smooth" if ev["team_side"] == "home" else "away_smooth"
                pre_press.append(w[col].mean())
        if pre_press:
            ax.bar(etype, np.mean(pre_press), color=color, alpha=0.8)
    ax.set_title("Mean pressure 10s before events", fontsize=11)
    ax.set_ylabel("Mean pressure"); ax.grid(True, axis='y', color='#eee')

    # Panel 2: event frequency by pressure tercile
    ax = axes[1]
    ax.set_facecolor('white')
    pt_copy = pt.copy()
    pt_copy["net_tercile"] = pd.qcut(pt_copy["net_smooth"], 3, labels=["Low", "Mid", "High"])
    for etype, color in event_colors.items():
        evs = game.events[game.events["Event"] == etype]
        counts = []
        for label in ["Low", "Mid", "High"]:
            seg = pt_copy[pt_copy["net_tercile"] == label]
            c = evs[evs["elapsed_sec"].isin(seg["elapsed_sec"])].shape[0]
            counts.append(c)
        ax.bar(["Low", "Mid", "High"], counts, alpha=0.7, color=color, label=etype)
    ax.set_title("Event count by net pressure tercile", fontsize=11)
    ax.set_xlabel("Net pressure tercile (home − away)"); ax.legend(fontsize=8)
    ax.grid(True, axis='y', color='#eee')

    # Panel 3: pressure over time with events
    ax = axes[2]
    ax.set_facecolor('white')
    ax.plot(pt["elapsed_sec"], pt["net_smooth"], color='#555', linewidth=1.0, label='Net pressure')
    ax.axhline(0, color='#aaa', linewidth=0.8, linestyle='--')
    for etype, color in event_colors.items():
        evs = game.events[game.events["Event"] == etype]
        for _, ev in evs.iterrows():
            ax.axvline(ev["elapsed_sec"], color=color, linewidth=1.2, alpha=0.6)
    ax.set_title("Net pressure + events over time", fontsize=11)
    ax.set_xlabel("Game time (s)"); ax.set_ylabel("Net pressure")
    ax.grid(True, color='#eee', linewidth=0.5)

    patches = [mpatches.Patch(color=c, label=e) for e, c in event_colors.items()]
    axes[2].legend(handles=patches, fontsize=8)
    fig.suptitle("Pressure–Events Linkage", fontsize=13)
    plt.tight_layout()

    path = save_path or f"{FIGURES_DIR}/pressure_events_linkage.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ── GIF 1: Faceoff net pressure field ─────────────────────────────────────────

def build_net_pressure_gif(game: GameData, pt: pd.DataFrame,
                            save_path: str = None, n_frames: int = 30):
    """Animated net pressure field averaged over post-faceoff windows."""
    print("Building net pressure field GIF...")
    faceoffs = game.get_faceoffs()
    if faceoffs.empty:
        print("  No faceoffs found — skipping GIF")
        return

    grid_x, grid_y = 40, 20
    x_edges = np.linspace(-100, 100, grid_x + 1)
    y_edges = np.linspace(-42.5, 42.5, grid_y + 1)

    fig, ax = plt.subplots(figsize=(10, 5)); fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0a1628')

    def compute_net_grid(sec):
        home_acc = np.zeros((grid_y, grid_x))
        away_acc = np.zeros((grid_y, grid_x))
        counts   = np.zeros((grid_y, grid_x))
        window_pt = pt[(pt["elapsed_sec"] >= sec) & (pt["elapsed_sec"] <= sec + 5)]
        for s in window_pt["elapsed_sec"]:
            frame_rows = game.tracking[game.tracking["elapsed_sec"] == s]
            if frame_rows.empty: continue
            frame = game.get_frame(frame_rows["frame_id"].iloc[0])
            pm = compute_frame_pressure_fast(frame, (HOME_NET_X, HOME_NET_Y), (AWAY_NET_X, AWAY_NET_Y))
            players = frame[frame["is_player"] & frame["x"].notna() & ~frame["is_goalie"]]
            for _, row in players.iterrows():
                xi = np.clip(np.searchsorted(x_edges, row["x"], side="right") - 1, 0, grid_x-1)
                yi = np.clip(np.searchsorted(y_edges, row["y"], side="right") - 1, 0, grid_y-1)
                pr = pm.get(row["player_id"], 0)
                if row["team_side"] == "home": home_acc[yi, xi] += pr
                else: away_acc[yi, xi] += pr
                counts[yi, xi] += 1
        safe = np.where(counts > 0, counts, 1)
        net  = gaussian_filter((home_acc - away_acc) / safe, sigma=1.5)
        return net

    frame_secs = faceoffs["elapsed_sec"].iloc[:n_frames].tolist()

    im = ax.imshow(np.zeros((grid_y, grid_x)), origin='lower', aspect='auto',
                   extent=[-100, 100, -42.5, 42.5],
                   cmap='RdBu_r', vmin=-5, vmax=5, alpha=0.85)
    for x in [BLUE_LINE_AWAY, BLUE_LINE_HOME]:
        ax.axvline(x, color='#4466cc', linewidth=1.5, alpha=0.7)
    ax.axvline(0, color='#cc3333', linewidth=1.2, alpha=0.5)
    title = ax.set_title("Net Pressure Field", color='white', fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.7, label='Net pressure (Home − Away)')

    def animate(i):
        sec = frame_secs[i]
        grid = compute_net_grid(sec)
        im.set_array(grid)
        title.set_text(f"Net Pressure Field — Faceoff {i+1}/{len(frame_secs)}  |  t={sec}s")
        return [im, title]

    ani = FuncAnimation(fig, animate, frames=len(frame_secs), interval=600, blit=True)
    path = save_path or f"{FIGURES_DIR}/faceoff_net_pressure_field.gif"
    ani.save(path, writer=PillowWriter(fps=1.5), dpi=100)
    plt.close()
    print(f"  Saved: {path}")


# ── GIF 2: Faceoff gradient (radial pressure) ─────────────────────────────────

def build_faceoff_gradient_gif(game: GameData, pt: pd.DataFrame,
                                 save_path: str = None, n_frames: int = 15):
    """Radial pressure gradient centred on faceoff dot."""
    print("Building faceoff gradient GIF...")
    faceoffs = game.get_faceoffs()
    if faceoffs.empty:
        print("  No faceoffs — skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 5)); fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0a1628')
    draw_rink(ax, alpha=0.08)

    scatter_h = ax.scatter([], [], c=[], cmap='Blues', vmin=0, vmax=100, s=120, zorder=4)
    scatter_a = ax.scatter([], [], c=[], cmap='Reds',  vmin=0, vmax=100, s=120, zorder=4)
    puck_dot, = ax.plot([], [], 'wo', markersize=8, zorder=5)
    title     = ax.set_title("Faceoff Pressure Gradient", color='white', fontsize=12)

    frame_list = faceoffs.iloc[:n_frames]

    def animate(i):
        fo  = frame_list.iloc[i]
        sec = fo["elapsed_sec"]
        frame_rows = game.tracking[game.tracking["elapsed_sec"] == sec]
        if frame_rows.empty: return scatter_h, scatter_a, puck_dot, title
        frame = game.get_frame(frame_rows["frame_id"].iloc[0])
        pm = compute_frame_pressure_fast(frame, (HOME_NET_X, HOME_NET_Y), (AWAY_NET_X, AWAY_NET_Y))

        hp = game.get_players(frame, "home")
        ap = game.get_players(frame, "away")
        puck = game.get_puck(frame)

        if not hp.empty:
            scatter_h.set_offsets(np.c_[hp["x"], hp["y"]])
            scatter_h.set_array(np.array([pm.get(pid, 0) for pid in hp["player_id"]]))
        if not ap.empty:
            scatter_a.set_offsets(np.c_[ap["x"], ap["y"]])
            scatter_a.set_array(np.array([pm.get(pid, 0) for pid in ap["player_id"]]))
        if puck is not None:
            puck_dot.set_data([puck["x"]], [puck["y"]])

        title.set_text(f"Faceoff {i+1}/{len(frame_list)}  |  t={sec}s")
        return scatter_h, scatter_a, puck_dot, title

    ani = FuncAnimation(fig, animate, frames=len(frame_list), interval=700, blit=True)
    path = save_path or f"{FIGURES_DIR}/faceoff_gradient.gif"
    ani.save(path, writer=PillowWriter(fps=1.4), dpi=100)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CSV EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def export_pressure_csv(game: GameData, pt: pd.DataFrame,
                         formations_df: pd.DataFrame,
                         board: pd.DataFrame,
                         output_prefix: str = "game"):
    """Save all computed data to CSVs in DATA_DIR."""
    pt.to_csv(f"{DATA_DIR}/{output_prefix}_pressure_timeline.csv", index=False)
    formations_df.to_csv(f"{DATA_DIR}/{output_prefix}_faceoff_formations.csv", index=False)
    board.to_csv(f"{DATA_DIR}/{output_prefix}_player_leaderboard.csv", index=False)
    print(f"  CSVs saved to {DATA_DIR}/")


def export_all_games(bdc_folder: str = BDC_DATA_FOLDER,
                     output_folder: str = OUTPUT_FOLDER):
    """
    Multi-game exporter.
    Auto-detects games by scanning bdc_folder for files matching the naming convention:
      YYYYMMDD_Team_X__Team_Y_Tracking_P*.csv
    Processes each game and saves one pressure timeline CSV per game + a combined CSV.
    """
    os.makedirs(output_folder, exist_ok=True)
    pattern  = os.path.join(bdc_folder, "*_Tracking_P1.csv")
    p1_files = sorted(glob.glob(pattern))
    print(f"Found {len(p1_files)} games.")

    all_timelines = []
    for p1_path in p1_files:
        prefix = p1_path.replace("_Tracking_P1.csv", "")
        p2_path = prefix + "_Tracking_P2.csv"
        p3_path = prefix + "_Tracking_P3.csv"
        ev_path = prefix + "_Events.csv"
        sh_path = prefix + "_Shifts.csv"

        if not all(os.path.exists(f) for f in [p2_path, ev_path, sh_path]):
            print(f"  Skipping {prefix} — missing files")
            continue

        game_id = os.path.basename(prefix)
        print(f"\n=== Processing {game_id} ===")
        try:
            files = [p1_path, p2_path]
            if os.path.exists(p3_path): files.append(p3_path)
            game = GameData(files, ev_path, sh_path)
            pt   = build_pressure_timeline(game, sample_every=PRESSURE_SAMPLE_EVERY)
            pt["game_id"] = game_id
            pt.to_csv(os.path.join(output_folder, f"{game_id}_pressure_timeline.csv"), index=False)
            all_timelines.append(pt)
        except Exception as e:
            print(f"  ERROR: {e}")

    if all_timelines:
        combined = pd.concat(all_timelines, ignore_index=True)
        combined.to_csv(os.path.join(output_folder, "ALL_GAMES_pressure_timeline.csv"), index=False)
        print(f"\nCombined CSV saved: {output_folder}/ALL_GAMES_pressure_timeline.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — RUN FULL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(tracking_files=None, events_file=None, shifts_file=None,
                       sample_every: int = None) -> dict:
    """
    Chains all pipeline steps and saves all outputs.
    Uses module-level TRACKING_FILES / EVENTS_FILE / SHIFTS_FILE by default.
    """
    tracking_files = tracking_files or TRACKING_FILES
    events_file    = events_file    or EVENTS_FILE
    shifts_file    = shifts_file    or SHIFTS_FILE
    sample_every   = sample_every   or PRESSURE_SAMPLE_EVERY

    print("=" * 60)
    print("HOCKEY PRESSURE PIPELINE")
    print("=" * 60)

    # Load data
    game = GameData(tracking_files, events_file, shifts_file)

    # Pressure timeline
    pt = build_pressure_timeline(game, sample_every=sample_every)

    # Player leaderboard
    board = build_player_pressure_leaderboard(game, pt)

    # Spatial heatmap
    heatmaps = build_spatial_heatmap(game, pt)

    # Pressing sequences + Andrienko stats
    stats    = compute_andrienko_stats(pt, game)
    seqs_df  = stats["A_sequences"]

    # Faceoff analysis
    formations_df = extract_faceoff_formations(game)
    if not formations_df.empty:
        formations_df = cluster_faceoff_formations(formations_df, FACEOFF_FORMATION_N_CLUSTERS)
        model_result  = faceoff_win_probability_model(formations_df)
        formations_df = compute_forecheck_linkage(game, formations_df, pt)
    else:
        model_result = {"model": None}

    # Forecheck
    fc_df = compute_forecheck_stats(game, pt)

    # Export CSVs
    export_pressure_csv(game, pt, formations_df, board)

    # ── All plots ────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_zone_pressure_timeline(pt, game)
    plot_player_leaderboard(board)
    plot_pressure_density(heatmaps)
    plot_pressing_sequences(pt, seqs_df)
    plot_andrienko_stats(stats)
    plot_pressure_shot_heatmap(heatmaps, game)
    plot_faceoff_pressure_macro(formations_df)
    plot_faceoff_formation_ice_map(formations_df)
    plot_faceoff_win_prob(formations_df, model_result)
    plot_forecheck_analysis(fc_df)
    plot_pressure_events_linkage(pt, game)

    # ── GIFs ─────────────────────────────────────────────────────────────────
    print("\nGenerating GIFs...")
    build_net_pressure_gif(game, pt)
    build_faceoff_gradient_gif(game, pt)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Figures saved to: ./{FIGURES_DIR}/")
    print(f"  Data saved to:    ./{DATA_DIR}/")
    print("=" * 60)

    return {
        "game":          game,
        "pressure_df":   pt,
        "heatmaps":      heatmaps,
        "formations_df": formations_df,
        "model_result":  model_result,
        "stats":         stats,
        "board":         board,
        "fc_df":         fc_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_full_analysis()
