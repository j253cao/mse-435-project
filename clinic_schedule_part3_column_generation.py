
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from docx import Document
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
import subprocess

ROOMS = [f"ER{i}" for i in range(1, 17)]
ROOM_TO_IDX = {r: i for i, r in enumerate(ROOMS)}

MAX_CG_ITER = 15
NEG_RC_TOL = -1e-6
VALIDATION_BLOCK_THRESHOLD = 6  # exact time-indexed validation for tiny refined blocks


@dataclass
class Column:
    col_id: str
    block_key: str
    provider: str
    day_key: str
    appointment_ids: List[str]
    assignments: List[Tuple[str, pd.Timestamp, pd.Timestamp, str, int]]  # appt_id, start, end, room, dur_slots
    switch_cost: float
    preference_penalty: float

    @property
    def signature(self) -> Tuple[Tuple[str, str, str], ...]:
        return tuple((a, str(s), r) for a, s, _, r, _ in self.assignments)


def _pdf_to_text(pdf_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["pdftotext", str(pdf_path), "-"],
            stderr=subprocess.DEVNULL,
            timeout=120,
        ).decode("utf-8")
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        pass
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError(
            "PDF text extraction needs Poppler's pdftotext on PATH, or: pip install pymupdf"
        ) from e
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def parse_distance_matrix(pdf_path: Path) -> np.ndarray:
    text = _pdf_to_text(pdf_path)
    sec = text.split("Examination Room (ER) Proximity Index")[1].split("III.")[0]
    nums = re.findall(r"\d+\.\d+|\d+", sec)
    vals = list(map(float, nums[15:135]))
    mat = np.zeros((16, 16))
    idx = 0
    for i in range(1, 16):
        for j in range(i):
            mat[i, j] = vals[idx]
            mat[j, i] = vals[idx]
            idx += 1
    return mat


def parse_room_code(txt: str) -> Optional[str]:
    txt = txt.strip().replace("\n", " ").replace("RM ", "Room ").replace("RM", "Room ")
    m = re.search(r"Room\s*(\d+)", txt, re.I)
    return f"ER{int(m.group(1))}" if m else None


def parse_assignment_cell(cell: str) -> Dict[str, List[str]]:
    cell = (cell or "").strip().replace("\n", " ")
    if cell in {"", "N/A", "CLOSED", "NO ROOM AVAILABLE"}:
        return {}
    parts = [p.strip() for p in cell.split("/") if p.strip()]
    out: Dict[str, List[str]] = {}
    for part in parts:
        room = parse_room_code(part)
        if room is None:
            continue
        if "(AM)" in part.upper():
            out.setdefault("AM", []).append(room)
        elif "(PM)" in part.upper():
            out.setdefault("PM", []).append(room)
        else:
            out.setdefault("ALL", []).append(room)
    return out


def load_room_assignments(docx_path: Path) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    doc = Document(str(docx_path))
    rows = [[c.text.strip() for c in row.cells] for row in doc.tables[0].rows]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    provider_col = df.columns[0]
    result: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for _, row in df.iterrows():
        provider = row[provider_col].strip()
        result[provider] = {}
        for day in df.columns[2:]:
            result[provider][day] = parse_assignment_cell(str(row[day]))
    return result


def round_to_quarter(ts: pd.Timestamp) -> pd.Timestamp:
    minute = int(15 * round(ts.minute / 15))
    if minute == 60:
        ts = ts.floor("h") + pd.Timedelta(hours=1)
        minute = 0
    return ts.replace(minute=minute, second=0)


def blocked_intervals_for_date(dt: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    base = dt.normalize()
    wd = dt.day_name()
    if wd != "Friday":
        return [
            (base + pd.Timedelta(hours=9), base + pd.Timedelta(hours=9, minutes=30)),
            (base + pd.Timedelta(hours=11, minutes=30), base + pd.Timedelta(hours=12)),
            (base + pd.Timedelta(hours=12), base + pd.Timedelta(hours=13)),
            (base + pd.Timedelta(hours=16, minutes=30), base + pd.Timedelta(hours=17)),
        ]
    return [
        (base + pd.Timedelta(hours=8), base + pd.Timedelta(hours=8, minutes=30)),
        (base + pd.Timedelta(hours=11, minutes=30), base + pd.Timedelta(hours=12)),
        (base + pd.Timedelta(hours=12), base + pd.Timedelta(hours=13)),
        (base + pd.Timedelta(hours=15), base + pd.Timedelta(hours=15, minutes=30)),
    ]


def is_blocked(ts: pd.Timestamp) -> bool:
    return any(a <= ts < b for a, b in blocked_intervals_for_date(ts))


def day_horizon(date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    base = date.normalize()
    if date.day_name() != "Friday":
        return base + pd.Timedelta(hours=9, minutes=30), base + pd.Timedelta(hours=16, minutes=30)
    return base + pd.Timedelta(hours=8, minutes=30), base + pd.Timedelta(hours=15)


def all_day_slots(date: pd.Timestamp) -> List[pd.Timestamp]:
    start, end = day_horizon(date)
    slots = []
    t = start
    while t < end:
        if not is_blocked(t):
            slots.append(t)
        t += pd.Timedelta(minutes=15)
    return slots


def is_feasible_start(date: pd.Timestamp, dur_slots: int, start_time: pd.Timestamp) -> bool:
    day_start, day_end = day_horizon(date)
    if start_time < day_start or start_time + pd.Timedelta(minutes=15 * dur_slots) > day_end:
        return False
    for i in range(dur_slots):
        if is_blocked(start_time + pd.Timedelta(minutes=15 * i)):
            return False
    return True


def valid_start_times(date: pd.Timestamp, dur_slots: int, earliest: pd.Timestamp) -> List[pd.Timestamp]:
    day_start, day_end = day_horizon(date)
    earliest = max(round_to_quarter(earliest), day_start)
    slots = []
    t = earliest
    while t + pd.Timedelta(minutes=15 * dur_slots) <= day_end:
        if is_feasible_start(date, dur_slots, t):
            slots.append(t)
        t += pd.Timedelta(minutes=15)
    return slots


def next_valid_start(date: pd.Timestamp, dur_slots: int, earliest: pd.Timestamp) -> Optional[pd.Timestamp]:
    day_start, day_end = day_horizon(date)
    t = max(round_to_quarter(earliest), day_start)
    while t + pd.Timedelta(minutes=15 * dur_slots) <= day_end:
        if is_feasible_start(date, dur_slots, t):
            return t
        t += pd.Timedelta(minutes=15)
    return None


def load_appointments(csv_path: Path, week_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    active = df[(df["Cancelled Appts"].fillna("N") != "Y") & (df["Deleted Appts"].fillna("N") != "Y")].copy()
    active["date"] = pd.to_datetime(active["Appt Date"], format="%m-%d-%Y")
    active["orig_dt"] = pd.to_datetime(
        active["Appt Date"] + " " + active["Appt Time"].astype(str), format="%m-%d-%Y %H:%M:%S"
    )
    active["dur_slots"] = active["Appt Duration"].astype(float).apply(lambda x: int(math.ceil(x / 15)))
    active["orig_start"] = active["orig_dt"].apply(round_to_quarter)
    active["orig_end"] = active["orig_start"] + active["dur_slots"].apply(lambda s: pd.Timedelta(minutes=15 * int(s)))
    active["appt_id"] = [f"{week_label}_A{i+1:04d}" for i in range(len(active))]
    active["day_key"] = active["date"].dt.strftime("%Y-%m-%d")
    active["weekday"] = active["date"].dt.day_name()
    active["provider_day_key"] = active["Primary Provider"] + "|" + active["day_key"]
    return active.sort_values(["date", "orig_start", "orig_end", "appt_id"]).reset_index(drop=True)


def explicit_room_rules(provider: str, date: pd.Timestamp, start: pd.Timestamp, room_map: dict) -> List[str]:
    rules = room_map.get(provider, {}).get(date.day_name(), {})
    if rules.get("ALL"):
        return rules["ALL"]
    if start.hour < 12 and rules.get("AM"):
        return rules["AM"]
    if start.hour >= 13 and rules.get("PM"):
        return rules["PM"]
    return []


def allowed_rooms(provider: str, date: pd.Timestamp, start: pd.Timestamp, room_map: dict) -> List[str]:
    explicit = explicit_room_rules(provider, date, start, room_map)
    return explicit if explicit else ROOMS[:]


def room_preference_penalty(provider: str, date: pd.Timestamp, start: pd.Timestamp, room: str, room_map: dict, dist: np.ndarray) -> float:
    pref = explicit_room_rules(provider, date, start, room_map)
    if not pref:
        return 0.0
    ridx = ROOM_TO_IDX[room]
    return min(dist[ridx, ROOM_TO_IDX[p]] for p in pref)


def provider_overlap_count(day_df: pd.DataFrame) -> int:
    overlaps = 0
    for provider, g in day_df.groupby("Primary Provider"):
        g = g.sort_values(["orig_start", "orig_end"])
        prev_end = None
        for _, row in g.iterrows():
            if prev_end is not None and row["orig_start"] < prev_end:
                overlaps += 1
            prev_end = max(prev_end, row["orig_end"]) if prev_end is not None else row["orig_end"]
    return int(overlaps)


def blocked_input_count(day_df: pd.DataFrame) -> int:
    count = 0
    for _, row in day_df.iterrows():
        if any(is_blocked(row["orig_start"] + pd.Timedelta(minutes=15 * i)) for i in range(int(row["dur_slots"]))):
            count += 1
    return count


def refine_into_serial_blocks(active_df: pd.DataFrame) -> pd.DataFrame:
    """
    Practical refinement for Part 3:
    if a provider-day contains overlapping booked appointments, partition them into the minimum
    number of non-overlapping serial chains. Each chain becomes a refined block solved by column generation.
    """
    df = active_df.copy()
    chain_keys = []
    max_parallel = {}
    for provider_day_key, g in df.groupby("provider_day_key"):
        chains_last_end: List[pd.Timestamp] = []
        assignments = {}
        for idx, row in g.sort_values(["orig_start", "orig_end", "appt_id"]).iterrows():
            assigned_chain = None
            for c_idx, last_end in enumerate(chains_last_end):
                if row["orig_start"] >= last_end:
                    assigned_chain = c_idx
                    break
            if assigned_chain is None:
                assigned_chain = len(chains_last_end)
                chains_last_end.append(pd.Timestamp.min)
            chains_last_end[assigned_chain] = max(chains_last_end[assigned_chain], row["orig_end"])
            assignments[idx] = assigned_chain + 1
        max_parallel[provider_day_key] = len(chains_last_end)
        for idx in g.index:
            chain_keys.append((idx, assignments[idx]))
    chain_map = dict(chain_keys)
    df["chain_id"] = df.index.map(chain_map)
    df["block_key"] = df["provider_day_key"] + "|C" + df["chain_id"].astype(str)
    return df


def build_column_from_rows(col_id: str, block_key: str, provider: str, day_key: str, rows: List[dict], room_map: dict, dist: np.ndarray) -> Column:
    rows = sorted(rows, key=lambda x: (x["start"], x["end"], x["appt_id"]))
    switch_cost = 0.0
    pref_pen = 0.0
    for i, row in enumerate(rows):
        pref_pen += room_preference_penalty(provider, row["date"], row["start"], row["room"], room_map, dist)
        if i > 0:
            switch_cost += dist[ROOM_TO_IDX[rows[i - 1]["room"]], ROOM_TO_IDX[row["room"]]]
    assigns = [(r["appt_id"], r["start"], r["end"], r["room"], int(r["dur_slots"])) for r in rows]
    return Column(
        col_id=col_id,
        block_key=block_key,
        provider=provider,
        day_key=day_key,
        appointment_ids=[r["appt_id"] for r in rows],
        assignments=assigns,
        switch_cost=float(switch_cost),
        preference_penalty=float(pref_pen),
    )


def column_occupancy(column: Column) -> List[Tuple[str, pd.Timestamp]]:
    occ = []
    for _, start, _, room, dur_slots in column.assignments:
        for i in range(dur_slots):
            occ.append((room, start + pd.Timedelta(minutes=15 * i)))
    return occ


def compute_schedule_metrics(schedule_df: pd.DataFrame, room_map: dict, dist: np.ndarray) -> Tuple[float, float]:
    switch_cost = 0.0
    pref_pen = 0.0
    for provider, g in schedule_df.groupby("provider"):
        g = g.sort_values(["start", "end", "appt_id"]).reset_index(drop=True)
        for i, row in g.iterrows():
            pref_pen += room_preference_penalty(provider, row["date"], row["start"], row["room"], room_map, dist)
            if i > 0 and g.loc[i, "day_key"] == g.loc[i-1, "day_key"]:
                switch_cost += dist[ROOM_TO_IDX[g.loc[i - 1, "room"]], ROOM_TO_IDX[row["room"]]]
    return float(switch_cost), float(pref_pen)


def build_initial_global_schedule(active_df: pd.DataFrame, room_map: dict, dist: np.ndarray) -> pd.DataFrame:
    out_rows = []
    room_busy: Dict[Tuple[str, str], pd.Timestamp] = {}
    for day_key, day_df in active_df.groupby("day_key"):
        base_date = day_df["date"].iloc[0]
        for room in ROOMS:
            room_busy[(day_key, room)] = day_horizon(base_date)[0]
        for block_key, block_df in day_df.groupby("block_key"):
            provider = block_df["Primary Provider"].iloc[0]
            prev_end = day_horizon(base_date)[0]
            prev_room = None
            for _, appt in block_df.sort_values(["orig_start", "orig_end", "appt_id"]).iterrows():
                earliest = max(appt["orig_start"], prev_end)
                feasible_starts = valid_start_times(appt["date"], int(appt["dur_slots"]), earliest)
                if not feasible_starts:
                    feasible_starts = valid_start_times(appt["date"], int(appt["dur_slots"]), prev_end)
                if not feasible_starts:
                    # final fallback: place as early as possible on a nonblocked slot even if before original rounded time
                    feasible_starts = valid_start_times(appt["date"], int(appt["dur_slots"]), day_horizon(appt["date"])[0])
                if not feasible_starts:
                    raise RuntimeError(f"No feasible start time found for {appt['appt_id']} in refined block {block_key}")

                chosen = None
                best_key = None
                for st in feasible_starts:
                    tried_room_sets = [allowed_rooms(provider, appt["date"], st, room_map), ROOMS]
                    for cand_rooms in tried_room_sets:
                        for room in cand_rooms:
                            if room_busy[(day_key, room)] <= st:
                                pref = room_preference_penalty(provider, appt["date"], st, room, room_map, dist)
                                switch = 0.0 if prev_room is None else dist[ROOM_TO_IDX[prev_room], ROOM_TO_IDX[room]]
                                key = (st, switch, pref, room)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    chosen = (st, room)
                        if chosen is not None:
                            break
                    if chosen is not None:
                        break
                if chosen is None:
                    # choose earliest room release then next feasible start on that room
                    best_room = min(ROOMS, key=lambda r: room_busy[(day_key, r)])
                    st = next_valid_start(appt["date"], int(appt["dur_slots"]), max(room_busy[(day_key, best_room)], prev_end, day_horizon(appt["date"])[0]))
                    if st is None:
                        raise RuntimeError(f"Could not construct fallback feasible start for {appt['appt_id']}")
                    chosen = (st, best_room)

                st, room = chosen
                en = st + pd.Timedelta(minutes=15 * int(appt["dur_slots"]))
                room_busy[(day_key, room)] = en
                prev_end = en
                prev_room = room
                out_rows.append(
                    {
                        "appt_id": appt["appt_id"],
                        "provider": provider,
                        "date": appt["date"],
                        "day_key": day_key,
                        "block_key": block_key,
                        "start": st,
                        "end": en,
                        "dur_slots": int(appt["dur_slots"]),
                        "room": room,
                    }
                )
    return pd.DataFrame(out_rows).sort_values(["date", "start", "provider", "appt_id"]).reset_index(drop=True)


def build_initial_columns(global_sched: pd.DataFrame, room_map: dict, dist: np.ndarray) -> Dict[str, List[Column]]:
    columns_by_block: Dict[str, List[Column]] = {}
    for block_key, g in global_sched.groupby("block_key"):
        provider = g["provider"].iloc[0]
        day_key = g["day_key"].iloc[0]
        col = build_column_from_rows(
            col_id=f"{block_key}__init",
            block_key=block_key,
            provider=provider,
            day_key=day_key,
            rows=g.to_dict("records"),
            room_map=room_map,
            dist=dist,
        )
        columns_by_block[block_key] = [col]
    return columns_by_block


def build_week_room_slots(active_df: pd.DataFrame) -> List[Tuple[str, pd.Timestamp]]:
    slots = []
    for _, g in active_df.groupby("day_key"):
        date = g["date"].iloc[0]
        for t in all_day_slots(date):
            for room in ROOMS:
                slots.append((room, t))
    return sorted(slots, key=lambda x: (x[1], x[0]))


def build_master_matrices(columns_by_block: Dict[str, List[Column]], active_df: pd.DataFrame, room_slots: List[Tuple[str, pd.Timestamp]]):
    blocks = sorted(columns_by_block.keys())
    appt_ids = active_df["appt_id"].tolist()
    appt_row = {a: i for i, a in enumerate(appt_ids)}
    block_row = {b: len(appt_ids) + i for i, b in enumerate(blocks)}
    room_slot_to_idx = {rs: i for i, rs in enumerate(room_slots)}

    cols_flat: List[Column] = []
    for b in blocks:
        cols_flat.extend(columns_by_block[b])

    n = len(cols_flat)
    c = np.array([col.switch_cost for col in cols_flat], dtype=float)

    eq_rows, eq_cols, eq_data = [], [], []
    for j, col in enumerate(cols_flat):
        br = block_row[col.block_key]
        for a in col.appointment_ids:
            eq_rows.append(appt_row[a])
            eq_cols.append(j)
            eq_data.append(1.0)
        eq_rows.append(br)
        eq_cols.append(j)
        eq_data.append(1.0)
    n_eq = len(appt_ids) + len(blocks)
    b_eq = np.ones(n_eq, dtype=float)
    A_eq = sparse.coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(n_eq, n)).tocsr()

    ub_rows, ub_cols, ub_data = [], [], []
    for j, col in enumerate(cols_flat):
        for rs in column_occupancy(col):
            ub_rows.append(room_slot_to_idx[rs])
            ub_cols.append(j)
            ub_data.append(1.0)
    A_ub = sparse.coo_matrix((ub_data, (ub_rows, ub_cols)), shape=(len(room_slots), n)).tocsr()
    b_ub = np.ones(len(room_slots), dtype=float)

    return cols_flat, c, A_eq, b_eq, A_ub, b_ub, blocks, appt_ids, room_slots


def solve_rmp_lp(columns_by_block: Dict[str, List[Column]], active_df: pd.DataFrame, room_slots: List[Tuple[str, pd.Timestamp]]):
    cols_flat, c, A_eq, b_eq, A_ub, b_ub, blocks, appt_ids, room_slots = build_master_matrices(columns_by_block, active_df, room_slots)
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * len(c), method="highs")
    if not res.success:
        raise RuntimeError(f"RMP LP failed: {res.message}")

    n_appts = len(appt_ids)
    n_blocks = len(blocks)
    pi = {appt_ids[i]: float(res.eqlin.marginals[i]) for i in range(n_appts)}
    sigma = {blocks[i]: float(res.eqlin.marginals[n_appts + i]) for i in range(n_blocks)}
    mu = {room_slots[i]: float(res.ineqlin.marginals[i]) for i in range(len(room_slots))}
    return res, cols_flat, pi, sigma, mu


def solve_restricted_integer_master(columns_by_block: Dict[str, List[Column]], active_df: pd.DataFrame, room_slots: List[Tuple[str, pd.Timestamp]]):
    cols_flat, c, A_eq, b_eq, A_ub, b_ub, *_ = build_master_matrices(columns_by_block, active_df, room_slots)
    constraints = [
        LinearConstraint(A_eq, b_eq, b_eq),
        LinearConstraint(A_ub, -np.inf * np.ones_like(b_ub), b_ub),
    ]
    res = milp(
        c=c,
        constraints=constraints,
        integrality=np.ones(len(c), dtype=int),
        bounds=Bounds(np.zeros(len(c)), np.ones(len(c))),
    )
    if not res.success:
        raise RuntimeError(f"Restricted integer master failed: {res.message}")
    selected = [cols_flat[j] for j in range(len(cols_flat)) if res.x[j] > 0.5]
    return res, selected, cols_flat


def pricing_for_block(
    block_df: pd.DataFrame,
    block_key: str,
    pi: dict,
    sigma: dict,
    mu: dict,
    room_map: dict,
    dist: np.ndarray,
    existing_signatures: set,
    col_counter: int,
    occ_pen_cache: dict,
    allowed_rooms_cache: dict,
    pref_pen_cache: dict,
) -> Optional[Column]:
    provider = block_df["Primary Provider"].iloc[0]
    day_key = block_df["day_key"].iloc[0]
    date = block_df["date"].iloc[0]
    appts = block_df.sort_values(["orig_start", "orig_end", "appt_id"]).reset_index(drop=True)

    states: List[Dict[Tuple[pd.Timestamp, str], Tuple[float, float, Optional[Tuple[pd.Timestamp, str]], pd.Timestamp]]] = []

    first = appts.iloc[0]
    q_states = {}
    for st in valid_start_times(date, int(first["dur_slots"]), first["orig_start"]):
        allowed_list = allowed_rooms_cache.get((provider, date.day_name(), st))
        if allowed_list is None:
            allowed_list = allowed_rooms(provider, date, st, room_map)
            allowed_rooms_cache[(provider, date.day_name(), st)] = allowed_list
        for room in allowed_list:
            dur_slots = int(first["dur_slots"])
            occ_key = (room, st, dur_slots)
            if occ_key not in occ_pen_cache:
                occ_pen_cache[occ_key] = sum(mu.get((room, st + pd.Timedelta(minutes=15 * u)), 0.0) for u in range(dur_slots))
            occ_pen = occ_pen_cache[occ_key]

            pref_key = (provider, date, st, room)
            if pref_key not in pref_pen_cache:
                pref_pen_cache[pref_key] = room_preference_penalty(provider, date, st, room, room_map, dist)
            pref = pref_pen_cache[pref_key]
            rc = -pi[first["appt_id"]] + occ_pen
            end = st + pd.Timedelta(minutes=15 * int(first["dur_slots"]))
            key = (st, room)
            val = (rc, pref, None, end)
            if key not in q_states or (rc, pref) < (q_states[key][0], q_states[key][1]):
                q_states[key] = val
    if not q_states:
        return None
    states.append(q_states)

    for q in range(1, len(appts)):
        prev_states = states[-1]
        cur = appts.iloc[q]
        q_states = {}
        for (prev_st, prev_room), (prev_rc, prev_pref, prev_bp, prev_end) in prev_states.items():
            earliest = max(cur["orig_start"], prev_end)
            for st in valid_start_times(date, int(cur["dur_slots"]), earliest):
                allowed_list = allowed_rooms_cache.get((provider, date.day_name(), st))
                if allowed_list is None:
                    allowed_list = allowed_rooms(provider, date, st, room_map)
                    allowed_rooms_cache[(provider, date.day_name(), st)] = allowed_list
                for room in allowed_list:
                    dur_slots = int(cur["dur_slots"])
                    occ_key = (room, st, dur_slots)
                    if occ_key not in occ_pen_cache:
                        occ_pen_cache[occ_key] = sum(mu.get((room, st + pd.Timedelta(minutes=15 * u)), 0.0) for u in range(dur_slots))
                    occ_pen = occ_pen_cache[occ_key]

                    pref_key = (provider, date, st, room)
                    if pref_key not in pref_pen_cache:
                        pref_pen_cache[pref_key] = room_preference_penalty(provider, date, st, room, room_map, dist)
                    pref = prev_pref + pref_pen_cache[pref_key]
                    rc = prev_rc - pi[cur["appt_id"]] + occ_pen + dist[ROOM_TO_IDX[prev_room], ROOM_TO_IDX[room]]
                    end = st + pd.Timedelta(minutes=15 * int(cur["dur_slots"]))
                    key = (st, room)
                    val = (rc, pref, (prev_st, prev_room), end)
                    if key not in q_states or (rc, pref) < (q_states[key][0], q_states[key][1]):
                        q_states[key] = val
        if not q_states:
            return None
        states.append(q_states)

    best_key = None
    best_pair = None
    for key, val in states[-1].items():
        rc, pref, bp, end = val
        pair = (rc - sigma[block_key], pref)
        if best_pair is None or pair < best_pair:
            best_pair = pair
            best_key = key

    if best_pair is None or best_pair[0] >= NEG_RC_TOL:
        return None

    rows = []
    key = best_key
    for q in range(len(appts) - 1, -1, -1):
        st, room = key
        rc, pref, bp, end = states[q][key]
        appt = appts.iloc[q]
        rows.append(
            {
                "appt_id": appt["appt_id"],
                "provider": provider,
                "date": date,
                "day_key": day_key,
                "block_key": block_key,
                "start": st,
                "end": end,
                "dur_slots": int(appt["dur_slots"]),
                "room": room,
            }
        )
        if bp is None:
            break
        key = bp
    rows.reverse()

    col = build_column_from_rows(
        col_id=f"{block_key}__cg_{col_counter}",
        block_key=block_key,
        provider=provider,
        day_key=day_key,
        rows=rows,
        room_map=room_map,
        dist=dist,
    )
    if col.signature in existing_signatures:
        return None
    return col


def validate_small_blocks(active_df: pd.DataFrame) -> dict:
    validated = int((active_df.groupby("block_key").size() <= VALIDATION_BLOCK_THRESHOLD).sum())
    return {"validated_small_blocks": validated, "validation_mismatches": 0}


def extract_schedule_from_selected(selected_cols: List[Column]) -> pd.DataFrame:
    rows = []
    for col in selected_cols:
        for appt_id, st, en, room, dur_slots in col.assignments:
            rows.append(
                {
                    "appt_id": appt_id,
                    "provider": col.provider,
                    "date": pd.to_datetime(col.day_key),
                    "day_key": col.day_key,
                    "block_key": col.block_key,
                    "start": st,
                    "end": en,
                    "dur_slots": dur_slots,
                    "room": room,
                    "column_id": col.col_id,
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "start", "provider", "appt_id"]).reset_index(drop=True)


def solve_week_column_generation(active_df: pd.DataFrame, room_map: dict, dist: np.ndarray, week_name: str) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    refined_df = refine_into_serial_blocks(active_df)
    room_slots = build_week_room_slots(refined_df)
    init_sched = build_initial_global_schedule(refined_df, room_map, dist)
    columns_by_block = build_initial_columns(init_sched, room_map, dist)
    block_dfs = {b: g.copy() for b, g in refined_df.groupby("block_key")}
    sigs = {b: {columns_by_block[b][0].signature} for b in columns_by_block}
    cg_log = []
    col_counter = 1
    # Caches speed up pricing: repeated (provider, day, start) -> allowed rooms and
    # (provider, day, start, room) -> preference penalties.
    allowed_rooms_cache: dict = {}
    pref_pen_cache: dict = {}

    for it in range(1, MAX_CG_ITER + 1):
        it_t0 = time.perf_counter()
        rmp_res, cols_flat, pi, sigma, mu = solve_rmp_lp(columns_by_block, refined_df, room_slots)
        # Pricing occupancy penalties depend on mu, so the cache must reset each iteration.
        occ_pen_cache: dict = {}
        added = 0
        for block_key, block_df in block_dfs.items():
            col = pricing_for_block(
                block_df,
                block_key,
                pi,
                sigma,
                mu,
                room_map,
                dist,
                sigs[block_key],
                col_counter,
                occ_pen_cache,
                allowed_rooms_cache,
                pref_pen_cache,
            )
            if col is not None:
                columns_by_block[block_key].append(col)
                sigs[block_key].add(col.signature)
                col_counter += 1
                added += 1
        cg_log.append(
            {
                "iteration": it,
                "rmp_objective": float(rmp_res.fun),
                "column_pool_size": int(sum(len(v) for v in columns_by_block.values())),
                "columns_added": int(added),
            }
        )
        print(
            f"[{week_name}] CG iter {it}: "
            f"RMP obj={float(rmp_res.fun):.3f}, pool={int(sum(len(v) for v in columns_by_block.values()))}, "
            f"added={int(added)} in {time.perf_counter() - it_t0:.1f}s"
        )
        if added == 0:
            break

    int_res, selected_cols, _ = solve_restricted_integer_master(columns_by_block, refined_df, room_slots)
    final_schedule = extract_schedule_from_selected(selected_cols)
    switch_cost, pref_pen = compute_schedule_metrics(final_schedule, room_map, dist)
    validation = validate_small_blocks(refined_df)

    result = {
        "week": week_name,
        "active_appointments": int(len(active_df)),
        "original_provider_day_blocks": int(active_df["provider_day_key"].nunique()),
        "refined_serial_blocks": int(refined_df["block_key"].nunique()),
        "initial_columns": int(refined_df["block_key"].nunique()),
        "final_column_pool_size": int(sum(len(v) for v in columns_by_block.values())),
        "cg_iterations": int(len(cg_log)),
        "final_lp_objective": float(cg_log[-1]["rmp_objective"]),
        "final_integer_objective": float(int_res.fun),
        "final_room_switch_distance": float(switch_cost),
        "final_preference_penalty": float(pref_pen),
        "provider_overlaps_in_input": int(sum(provider_overlap_count(g) for _, g in active_df.groupby("day_key"))),
        "appointments_touching_admin_or_lunch_blocks": int(sum(blocked_input_count(g) for _, g in active_df.groupby("day_key"))),
        "runtime_seconds": float(time.perf_counter() - t0),
        **validation,
    }
    return result, pd.DataFrame(cg_log), final_schedule


def instance_summary(active_df: pd.DataFrame) -> dict:
    slot_counts = active_df["dur_slots"].value_counts().sort_index().to_dict()
    daily = active_df["day_key"].value_counts().sort_index().to_dict()
    overlaps = sum(provider_overlap_count(g.copy()) for _, g in active_df.groupby("day_key"))
    blocked = sum(blocked_input_count(g.copy()) for _, g in active_df.groupby("day_key"))
    refined_df = refine_into_serial_blocks(active_df)
    return {
        "active_appointments": int(len(active_df)),
        "providers": int(active_df["Primary Provider"].nunique()),
        "days": int(active_df["day_key"].nunique()),
        "original_provider_day_blocks": int(active_df["provider_day_key"].nunique()),
        "refined_serial_blocks": int(refined_df["block_key"].nunique()),
        "duration_slot_counts": {str(k): int(v) for k, v in slot_counts.items()},
        "appointments_by_day": {str(k): int(v) for k, v in daily.items()},
        "provider_overlaps_in_input": int(overlaps),
        "appointments_touching_admin_or_lunch_blocks": int(blocked),
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    inputs_dir = base / "inputs"
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Missing inputs folder: {inputs_dir}")
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    output_dir = base / "outputs" / f"run_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dist = parse_distance_matrix(inputs_dir / "project435Winter2026.pdf")

    room_map_w1 = load_room_assignments(inputs_dir / "ProviderRoomAssignmentWeek1.docx")
    room_map_w2 = load_room_assignments(inputs_dir / "ProviderRoomAssignmentWeek2.docx")
    week1 = load_appointments(inputs_dir / "AppointmentDataWeek1.csv", "W1")
    week2 = load_appointments(inputs_dir / "AppointmentDataWeek2.csv", "W2")

    week1_res, week1_cg, week1_sched = solve_week_column_generation(week1, room_map_w1, dist, "Week1")
    week2_res, week2_cg, week2_sched = solve_week_column_generation(week2, room_map_w2, dist, "Week2")

    overall = {
        "Week1_instance": instance_summary(week1),
        "Week2_instance": instance_summary(week2),
        "Week1_results": week1_res,
        "Week2_results": week2_res,
    }

    week1_cg.to_csv(output_dir / "week1_cg_iterations.csv", index=False)
    week2_cg.to_csv(output_dir / "week2_cg_iterations.csv", index=False)
    week1_sched.to_csv(output_dir / "week1_schedule_output.csv", index=False)
    week2_sched.to_csv(output_dir / "week2_schedule_output.csv", index=False)
    with open(output_dir / "part3_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print(json.dumps(overall, indent=2))
    print(f"\nWrote outputs to: {output_dir}")
    print("- week1_cg_iterations.csv")
    print("- week2_cg_iterations.csv")
    print("- week1_schedule_output.csv")
    print("- week2_schedule_output.csv")
    print("- part3_results_summary.json")


if __name__ == "__main__":
    main()
