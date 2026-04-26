"""
UFC New Fight Scraper
Scrapes fights from ufcstats.com AFTER a given cutoff date.
Produces a CSV matching the exact 144-column format of the Kaggle dataset.

Run:
    python scrape_new_fights.py

Output:
    data/new_fights.csv       ← new fights only
    data/combined_data.csv    ← merged with existing Kaggle data
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
import os
import json
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BASE       = "http://www.ufcstats.com"
HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; UFCResearch/1.0)"}
CUTOFF     = "2021-03-20"   # Only scrape fights after this date
KAGGLE_CSV = r"data\ufc-master.csv\data.csv"
OUTPUT_NEW = r"data\new_fights.csv"
OUTPUT_ALL = r"data\combined_data.csv"

print_lock = threading.Lock()

def safe_print(*args):
    with print_lock:
        print(*args)

# ── HTTP helpers ───────────────────────────────────────────────────────────────

def get_soup(url, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception:
            time.sleep(2)
    return None

# ── Parsers ────────────────────────────────────────────────────────────────────

def to_float(v):
    if v is None or str(v).strip() in ("--", "", "N/A"):
        return None
    try:
        return float(str(v).replace("%", "").strip())
    except:
        return None

def to_cm(v):
    if not v or str(v).strip() in ("--", ""):
        return None
    m = re.search(r"(\d+)'\s*(\d+)", str(v))
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 1)
    return None

def to_lbs(v):
    if not v or str(v).strip() in ("--", ""):
        return None
    m = re.search(r"(\d+\.?\d*)", str(v))
    return float(m.group(1)) if m else None

def calc_age(dob_str, fight_date):
    if not dob_str or str(dob_str).strip() in ("--", ""):
        return None
    try:
        dob = datetime.strptime(dob_str.strip(), "%b %d, %Y")
        fd  = fight_date if isinstance(fight_date, date) else datetime.strptime(str(fight_date), "%Y-%m-%d").date()
        return fd.year - dob.year - ((fd.month, fd.day) < (dob.month, dob.day))
    except:
        return None

def parse_time_to_seconds(t):
    try:
        parts = str(t).strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return 0

def parse_pct(v):
    f = to_float(v)
    if f is None:
        return None
    return round(f / 100, 4) if f > 1 else round(f, 4)

# ── Step 1: Get all event URLs ────────────────────────────────────────────────

def get_event_urls_after(cutoff_str):
    """Get all event URLs with dates after cutoff."""
    cutoff = datetime.strptime(cutoff_str, "%Y-%m-%d").date()
    soup   = get_soup(f"{BASE}/statistics/events/completed?page=all")
    if not soup:
        return []

    events = []
    for link in soup.select('a[href*="event-details"]'):
        url    = link.get("href", "")
        parent = link.find_parent("tr")
        if not parent:
            continue

        # Date is in span[0] inside td[0]
        span = parent.select_one("span")
        if not span:
            continue

        date_text = span.get_text(strip=True)
        try:
            event_date = datetime.strptime(date_text, "%B %d, %Y").date()
        except:
            continue

        if event_date > cutoff:
            events.append((event_date, url))

    events.sort(key=lambda x: x[0])
    print(f"  Found {len(events)} events after {cutoff_str}")
    return events

# ── Step 2: Get fight URLs from event page ────────────────────────────────────

def get_fight_urls_from_event(event_url):
    soup = get_soup(event_url)
    if not soup:
        return []
    urls = []
    for row in soup.select("table.b-fight-details__table tbody tr"):
        link = row.get("data-link") or ""
        if "/fight-details/" in link:
            urls.append(link)
    return urls

# ── Step 3: Get fighter details ───────────────────────────────────────────────

_fighter_cache = {}
_cache_lock    = threading.Lock()

def get_fighter_details(fighter_url):
    with _cache_lock:
        if fighter_url in _fighter_cache:
            return _fighter_cache[fighter_url]

    soup = get_soup(fighter_url)
    if not soup:
        return {}

    details = {}

    # Name
    name_el = soup.select_one("span.b-content__title-highlight")
    details["name"] = name_el.text.strip() if name_el else ""

    # Record
    record_el = soup.select_one("span.b-content__title-record")
    if record_el:
        m = re.search(r"(\d+)-(\d+)-?(\d+)?", record_el.text)
        if m:
            details["wins"]   = int(m.group(1))
            details["losses"] = int(m.group(2))
            details["draws"]  = int(m.group(3)) if m.group(3) else 0

    # Stat box
    for li in soup.select("li.b-list__box-list-item"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            k, _, v = text.partition(":")
            details[k.strip().lower()] = v.strip()

    details["height_cms"] = to_cm(details.get("height"))
    details["reach_cms"]  = to_cm(details.get("reach"))
    details["weight_lbs"] = to_lbs(details.get("weight"))
    details["stance"]     = details.get("stance", "")
    details["dob"]        = details.get("dob", "")

    # Win/loss breakdown from fight history
    rows    = soup.select("table.b-fight-details__table tbody tr")
    results = []
    ko_w = sub_w = dec_w = 0
    win_streak = cur_lose = longest_win = 0
    title_bouts = 0
    total_time  = 0
    total_rounds = 0

    for row in rows:
        cols = row.select("td")
        if len(cols) < 2:
            continue
        outcome = cols[0].get_text(strip=True).upper()
        if outcome not in ("WIN", "LOSS", "DRAW", "NC", "W", "L", "D"):
            continue
        outcome = "W" if outcome == "WIN" else ("L" if outcome == "LOSS" else outcome[0])

        method = ""
        for idx in [1, 7, 8]:
            if len(cols) > idx:
                text = cols[idx].get_text(strip=True).upper()
                if any(k in text for k in ["KO", "TKO", "SUB", "DEC"]):
                    method = text
                    break

        results.append(outcome)
        if outcome == "W":
            if "KO" in method or "TKO" in method:
                ko_w += 1
            elif "SUB" in method:
                sub_w += 1
            elif "DEC" in method:
                if "UNANIMOUS" in method:
                    details.setdefault("dec_unanimous", 0)
                    details["dec_unanimous"] = details.get("dec_unanimous", 0) + 1
                elif "SPLIT" in method:
                    details.setdefault("dec_split", 0)
                    details["dec_split"] = details.get("dec_split", 0) + 1
                elif "MAJORITY" in method:
                    details.setdefault("dec_majority", 0)
                    details["dec_majority"] = details.get("dec_majority", 0) + 1
                dec_w += 1

    # Win streak
    for r in results:
        if r == "W":
            win_streak += 1
        else:
            break
    # Lose streak
    for r in results:
        if r == "L":
            cur_lose += 1
        else:
            break
    # Longest win streak
    cur = 0
    for r in results:
        if r == "W":
            cur += 1
            longest_win = max(longest_win, cur)
        else:
            cur = 0

    details["win_by_ko"]         = ko_w
    details["win_by_sub"]        = sub_w
    details["win_by_dec"]        = dec_w
    details["current_win_streak"] = win_streak
    details["current_lose_streak"] = cur_lose
    details["longest_win_streak"]  = longest_win
    details["total_title_bouts"]   = title_bouts

    with _cache_lock:
        _fighter_cache[fighter_url] = details

    return details

# ── Step 4: Parse a single fight page ────────────────────────────────────────

def parse_fight(fight_url, event_date, event_name=""):
    soup = get_soup(fight_url)
    if not soup:
        return None

    try:
        # ── Fighter names & links ──────────────────────────────────────────────
        fighter_links = soup.select("div.b-fight-details__person")
        if len(fighter_links) < 2:
            return None

        fighters = []
        for fl in fighter_links[:2]:
            name_el = fl.select_one("h3 a") or fl.select_one("a")
            status_el = fl.select_one("i.b-fight-details__person-status")
            fighters.append({
                "name":   name_el.text.strip() if name_el else "",
                "url":    name_el.get("href", "") if name_el else "",
                "status": status_el.text.strip().upper() if status_el else "",
            })

        red  = fighters[0]
        blue = fighters[1]

        # Determine winner
        if red["status"] == "W":
            winner = "Red"
        elif blue["status"] == "W":
            winner = "Blue"
        else:
            return None  # Skip draws/NCs

        # ── Fight metadata ─────────────────────────────────────────────────────
        details_box = {}
        for li in soup.select("div.b-fight-details__fight li"):
            text = li.get_text(" ", strip=True)
            if ":" in text:
                k, _, v = text.partition(":")
                details_box[k.strip().lower()] = v.strip()

        method      = details_box.get("method", "")
        weight_class = details_box.get("weight class", "")
        title_bout  = 1 if "title" in details_box.get("type", "").lower() else 0
        referee     = details_box.get("referee", "")
        rounds      = to_float(details_box.get("round", "0")) or 0
        time_str    = details_box.get("time", "0:00")
        time_secs   = parse_time_to_seconds(time_str)

        # ── Per-round stats tables ─────────────────────────────────────────────
        tables = soup.select("table.b-fight-details__table")

        def parse_stats_table(table):
            """Parse totals table → returns (red_stats, blue_stats) dicts."""
            rows = table.select("tbody tr")
            if not rows:
                return {}, {}
            cols = rows[0].select("td")
            if len(cols) < 2:
                return {}, {}

            def cell_pair(idx):
                if idx >= len(cols):
                    return None, None
                text = cols[idx].get_text(" ", strip=True)
                parts = text.split(" of ") if " of " in text else text.split()
                if len(parts) == 2:
                    return to_float(parts[0]), to_float(parts[1])
                return to_float(text), None

            red_s, blue_s = {}, {}

            # KD
            if len(cols) > 1:
                vals = cols[1].get_text(" ", strip=True).split()
                red_s["KD"]  = to_float(vals[0]) if len(vals) > 0 else 0
                blue_s["KD"] = to_float(vals[1]) if len(vals) > 1 else 0

            # SIG STR
            if len(cols) > 2:
                text = cols[2].get_text(" ", strip=True)
                pairs = re.findall(r"(\d+)\s+of\s+(\d+)", text)
                if len(pairs) >= 2:
                    red_s["SIG_STR_landed"]  = to_float(pairs[0][0])
                    red_s["SIG_STR_att"]     = to_float(pairs[0][1])
                    blue_s["SIG_STR_landed"] = to_float(pairs[1][0])
                    blue_s["SIG_STR_att"]    = to_float(pairs[1][1])

            # SIG STR pct
            if len(cols) > 3:
                vals = cols[3].get_text(" ", strip=True).split()
                red_s["SIG_STR_pct"]  = parse_pct(vals[0]) if vals else None
                blue_s["SIG_STR_pct"] = parse_pct(vals[1]) if len(vals) > 1 else None

            # TOTAL STR
            if len(cols) > 4:
                text = cols[4].get_text(" ", strip=True)
                pairs = re.findall(r"(\d+)\s+of\s+(\d+)", text)
                if len(pairs) >= 2:
                    red_s["TOTAL_STR_landed"]  = to_float(pairs[0][0])
                    red_s["TOTAL_STR_att"]     = to_float(pairs[0][1])
                    blue_s["TOTAL_STR_landed"] = to_float(pairs[1][0])
                    blue_s["TOTAL_STR_att"]    = to_float(pairs[1][1])

            # TD
            if len(cols) > 5:
                text = cols[5].get_text(" ", strip=True)
                pairs = re.findall(r"(\d+)\s+of\s+(\d+)", text)
                if len(pairs) >= 2:
                    red_s["TD_landed"]  = to_float(pairs[0][0])
                    red_s["TD_att"]     = to_float(pairs[0][1])
                    blue_s["TD_landed"] = to_float(pairs[1][0])
                    blue_s["TD_att"]    = to_float(pairs[1][1])

            # TD pct
            if len(cols) > 6:
                vals = cols[6].get_text(" ", strip=True).split()
                red_s["TD_pct"]  = parse_pct(vals[0]) if vals else None
                blue_s["TD_pct"] = parse_pct(vals[1]) if len(vals) > 1 else None

            # SUB ATT
            if len(cols) > 7:
                vals = cols[7].get_text(" ", strip=True).split()
                red_s["SUB_ATT"]  = to_float(vals[0]) if vals else 0
                blue_s["SUB_ATT"] = to_float(vals[1]) if len(vals) > 1 else 0

            # REV
            if len(cols) > 8:
                vals = cols[8].get_text(" ", strip=True).split()
                red_s["REV"]  = to_float(vals[0]) if vals else 0
                blue_s["REV"] = to_float(vals[1]) if len(vals) > 1 else 0

            # CTRL
            if len(cols) > 9:
                vals = cols[9].get_text(" ", strip=True).split()
                red_s["CTRL"]  = parse_time_to_seconds(vals[0]) if vals else 0
                blue_s["CTRL"] = parse_time_to_seconds(vals[1]) if len(vals) > 1 else 0

            return red_s, blue_s

        def parse_strikes_table(table):
            """Parse significant strikes breakdown table."""
            rows = table.select("tbody tr")
            if not rows:
                return {}, {}
            cols = rows[0].select("td")

            red_s, blue_s = {}, {}

            strike_map = [
                (1, "HEAD"),
                (2, "BODY"),
                (3, "LEG"),
                (4, "DISTANCE"),
                (5, "CLINCH"),
                (6, "GROUND"),
            ]
            for idx, name in strike_map:
                if idx < len(cols):
                    text = cols[idx].get_text(" ", strip=True)
                    pairs = re.findall(r"(\d+)\s+of\s+(\d+)", text)
                    if len(pairs) >= 2:
                        red_s[f"{name}_landed"]  = to_float(pairs[0][0])
                        red_s[f"{name}_att"]     = to_float(pairs[0][1])
                        blue_s[f"{name}_landed"] = to_float(pairs[1][0])
                        blue_s[f"{name}_att"]    = to_float(pairs[1][1])

            return red_s, blue_s

        red_totals, blue_totals = {}, {}
        red_strikes, blue_strikes = {}, {}

        if len(tables) >= 1:
            red_totals, blue_totals = parse_stats_table(tables[0])
        if len(tables) >= 3:
            red_strikes, blue_strikes = parse_strikes_table(tables[2])

        # ── Get fighter profiles ───────────────────────────────────────────────
        red_profile  = get_fighter_details(red["url"])  if red["url"]  else {}
        blue_profile = get_fighter_details(blue["url"]) if blue["url"] else {}

        fight_date_obj = event_date if isinstance(event_date, date) else datetime.strptime(str(event_date), "%Y-%m-%d").date()

        # ── Build row matching Kaggle format ───────────────────────────────────
        def f(d, k, default=np.nan):
            v = d.get(k)
            return v if v is not None else default

        # For avg columns: this fight's stats / rounds (per-round average)
        r_rounds = max(rounds, 1)

        row = {
            "R_fighter":   red["name"],
            "B_fighter":   blue["name"],
            "Referee":     referee,
            "date":        str(event_date),
            "location":    event_name,
            "Winner":      winner,
            "title_bout":  title_bout,
            "weight_class": weight_class,

            # ── Blue fighter career stats up to this fight ─────────────────
            "B_avg_KD":               f(blue_totals, "KD", 0) / r_rounds,
            "B_avg_opp_KD":           f(red_totals,  "KD", 0) / r_rounds,
            "B_avg_SIG_STR_pct":      f(blue_totals, "SIG_STR_pct"),
            "B_avg_opp_SIG_STR_pct":  f(red_totals,  "SIG_STR_pct"),
            "B_avg_TD_pct":           f(blue_totals, "TD_pct"),
            "B_avg_opp_TD_pct":       f(red_totals,  "TD_pct"),
            "B_avg_SUB_ATT":          f(blue_totals, "SUB_ATT", 0) / r_rounds,
            "B_avg_opp_SUB_ATT":      f(red_totals,  "SUB_ATT", 0) / r_rounds,
            "B_avg_REV":              f(blue_totals, "REV", 0) / r_rounds,
            "B_avg_opp_REV":          f(red_totals,  "REV", 0) / r_rounds,
            "B_avg_SIG_STR_att":      f(blue_totals, "SIG_STR_att", 0) / r_rounds,
            "B_avg_SIG_STR_landed":   f(blue_totals, "SIG_STR_landed", 0) / r_rounds,
            "B_avg_opp_SIG_STR_att":  f(red_totals,  "SIG_STR_att", 0) / r_rounds,
            "B_avg_opp_SIG_STR_landed": f(red_totals, "SIG_STR_landed", 0) / r_rounds,
            "B_avg_TOTAL_STR_att":    f(blue_totals, "TOTAL_STR_att", 0) / r_rounds,
            "B_avg_TOTAL_STR_landed": f(blue_totals, "TOTAL_STR_landed", 0) / r_rounds,
            "B_avg_opp_TOTAL_STR_att":    f(red_totals, "TOTAL_STR_att", 0) / r_rounds,
            "B_avg_opp_TOTAL_STR_landed": f(red_totals, "TOTAL_STR_landed", 0) / r_rounds,
            "B_avg_TD_att":           f(blue_totals, "TD_att", 0) / r_rounds,
            "B_avg_TD_landed":        f(blue_totals, "TD_landed", 0) / r_rounds,
            "B_avg_opp_TD_att":       f(red_totals,  "TD_att", 0) / r_rounds,
            "B_avg_opp_TD_landed":    f(red_totals,  "TD_landed", 0) / r_rounds,
            "B_avg_HEAD_att":         f(blue_strikes, "HEAD_att", 0) / r_rounds,
            "B_avg_HEAD_landed":      f(blue_strikes, "HEAD_landed", 0) / r_rounds,
            "B_avg_opp_HEAD_att":     f(red_strikes,  "HEAD_att", 0) / r_rounds,
            "B_avg_opp_HEAD_landed":  f(red_strikes,  "HEAD_landed", 0) / r_rounds,
            "B_avg_BODY_att":         f(blue_strikes, "BODY_att", 0) / r_rounds,
            "B_avg_BODY_landed":      f(blue_strikes, "BODY_landed", 0) / r_rounds,
            "B_avg_opp_BODY_att":     f(red_strikes,  "BODY_att", 0) / r_rounds,
            "B_avg_opp_BODY_landed":  f(red_strikes,  "BODY_landed", 0) / r_rounds,
            "B_avg_LEG_att":          f(blue_strikes, "LEG_att", 0) / r_rounds,
            "B_avg_LEG_landed":       f(blue_strikes, "LEG_landed", 0) / r_rounds,
            "B_avg_opp_LEG_att":      f(red_strikes,  "LEG_att", 0) / r_rounds,
            "B_avg_opp_LEG_landed":   f(red_strikes,  "LEG_landed", 0) / r_rounds,
            "B_avg_DISTANCE_att":     f(blue_strikes, "DISTANCE_att", 0) / r_rounds,
            "B_avg_DISTANCE_landed":  f(blue_strikes, "DISTANCE_landed", 0) / r_rounds,
            "B_avg_opp_DISTANCE_att":    f(red_strikes, "DISTANCE_att", 0) / r_rounds,
            "B_avg_opp_DISTANCE_landed": f(red_strikes, "DISTANCE_landed", 0) / r_rounds,
            "B_avg_CLINCH_att":       f(blue_strikes, "CLINCH_att", 0) / r_rounds,
            "B_avg_CLINCH_landed":    f(blue_strikes, "CLINCH_landed", 0) / r_rounds,
            "B_avg_opp_CLINCH_att":   f(red_strikes,  "CLINCH_att", 0) / r_rounds,
            "B_avg_opp_CLINCH_landed":f(red_strikes,  "CLINCH_landed", 0) / r_rounds,
            "B_avg_GROUND_att":       f(blue_strikes, "GROUND_att", 0) / r_rounds,
            "B_avg_GROUND_landed":    f(blue_strikes, "GROUND_landed", 0) / r_rounds,
            "B_avg_opp_GROUND_att":   f(red_strikes,  "GROUND_att", 0) / r_rounds,
            "B_avg_opp_GROUND_landed":f(red_strikes,  "GROUND_landed", 0) / r_rounds,
            "B_avg_CTRL_time(seconds)":     f(blue_totals, "CTRL", 0) / r_rounds,
            "B_avg_opp_CTRL_time(seconds)": f(red_totals,  "CTRL", 0) / r_rounds,
            "B_total_time_fought(seconds)": time_secs,
            "B_total_rounds_fought":  rounds,
            "B_total_title_bouts":    f(blue_profile, "total_title_bouts", 0),
            "B_current_win_streak":   f(blue_profile, "current_win_streak", 0),
            "B_current_lose_streak":  f(blue_profile, "current_lose_streak", 0),
            "B_longest_win_streak":   f(blue_profile, "longest_win_streak", 0),
            "B_wins":                 f(blue_profile, "wins", 0),
            "B_losses":               f(blue_profile, "losses", 0),
            "B_draw":                 f(blue_profile, "draws", 0),
            "B_win_by_Decision_Majority":  f(blue_profile, "dec_majority", 0),
            "B_win_by_Decision_Split":     f(blue_profile, "dec_split", 0),
            "B_win_by_Decision_Unanimous": f(blue_profile, "dec_unanimous", 0),
            "B_win_by_KO/TKO":        f(blue_profile, "win_by_ko", 0),
            "B_win_by_Submission":    f(blue_profile, "win_by_sub", 0),
            "B_win_by_TKO_Doctor_Stoppage": 0,
            "B_Stance":               f(blue_profile, "stance", ""),
            "B_Height_cms":           f(blue_profile, "height_cms"),
            "B_Reach_cms":            f(blue_profile, "reach_cms"),
            "B_Weight_lbs":           f(blue_profile, "weight_lbs"),

            # ── Red fighter career stats ────────────────────────────────────
            "R_avg_KD":               f(red_totals,  "KD", 0) / r_rounds,
            "R_avg_opp_KD":           f(blue_totals, "KD", 0) / r_rounds,
            "R_avg_SIG_STR_pct":      f(red_totals,  "SIG_STR_pct"),
            "R_avg_opp_SIG_STR_pct":  f(blue_totals, "SIG_STR_pct"),
            "R_avg_TD_pct":           f(red_totals,  "TD_pct"),
            "R_avg_opp_TD_pct":       f(blue_totals, "TD_pct"),
            "R_avg_SUB_ATT":          f(red_totals,  "SUB_ATT", 0) / r_rounds,
            "R_avg_opp_SUB_ATT":      f(blue_totals, "SUB_ATT", 0) / r_rounds,
            "R_avg_REV":              f(red_totals,  "REV", 0) / r_rounds,
            "R_avg_opp_REV":          f(blue_totals, "REV", 0) / r_rounds,
            "R_avg_SIG_STR_att":      f(red_totals,  "SIG_STR_att", 0) / r_rounds,
            "R_avg_SIG_STR_landed":   f(red_totals,  "SIG_STR_landed", 0) / r_rounds,
            "R_avg_opp_SIG_STR_att":  f(blue_totals, "SIG_STR_att", 0) / r_rounds,
            "R_avg_opp_SIG_STR_landed": f(blue_totals, "SIG_STR_landed", 0) / r_rounds,
            "R_avg_TOTAL_STR_att":    f(red_totals,  "TOTAL_STR_att", 0) / r_rounds,
            "R_avg_TOTAL_STR_landed": f(red_totals,  "TOTAL_STR_landed", 0) / r_rounds,
            "R_avg_opp_TOTAL_STR_att":    f(blue_totals, "TOTAL_STR_att", 0) / r_rounds,
            "R_avg_opp_TOTAL_STR_landed": f(blue_totals, "TOTAL_STR_landed", 0) / r_rounds,
            "R_avg_TD_att":           f(red_totals,  "TD_att", 0) / r_rounds,
            "R_avg_TD_landed":        f(red_totals,  "TD_landed", 0) / r_rounds,
            "R_avg_opp_TD_att":       f(blue_totals, "TD_att", 0) / r_rounds,
            "R_avg_opp_TD_landed":    f(blue_totals, "TD_landed", 0) / r_rounds,
            "R_avg_HEAD_att":         f(red_strikes,  "HEAD_att", 0) / r_rounds,
            "R_avg_HEAD_landed":      f(red_strikes,  "HEAD_landed", 0) / r_rounds,
            "R_avg_opp_HEAD_att":     f(blue_strikes, "HEAD_att", 0) / r_rounds,
            "R_avg_opp_HEAD_landed":  f(blue_strikes, "HEAD_landed", 0) / r_rounds,
            "R_avg_BODY_att":         f(red_strikes,  "BODY_att", 0) / r_rounds,
            "R_avg_BODY_landed":      f(red_strikes,  "BODY_landed", 0) / r_rounds,
            "R_avg_opp_BODY_att":     f(blue_strikes, "BODY_att", 0) / r_rounds,
            "R_avg_opp_BODY_landed":  f(blue_strikes, "BODY_landed", 0) / r_rounds,
            "R_avg_LEG_att":          f(red_strikes,  "LEG_att", 0) / r_rounds,
            "R_avg_LEG_landed":       f(red_strikes,  "LEG_landed", 0) / r_rounds,
            "R_avg_opp_LEG_att":      f(blue_strikes, "LEG_att", 0) / r_rounds,
            "R_avg_opp_LEG_landed":   f(blue_strikes, "LEG_landed", 0) / r_rounds,
            "R_avg_DISTANCE_att":     f(red_strikes,  "DISTANCE_att", 0) / r_rounds,
            "R_avg_DISTANCE_landed":  f(red_strikes,  "DISTANCE_landed", 0) / r_rounds,
            "R_avg_opp_DISTANCE_att":    f(blue_strikes, "DISTANCE_att", 0) / r_rounds,
            "R_avg_opp_DISTANCE_landed": f(blue_strikes, "DISTANCE_landed", 0) / r_rounds,
            "R_avg_CLINCH_att":       f(red_strikes,  "CLINCH_att", 0) / r_rounds,
            "R_avg_CLINCH_landed":    f(red_strikes,  "CLINCH_landed", 0) / r_rounds,
            "R_avg_opp_CLINCH_att":   f(blue_strikes, "CLINCH_att", 0) / r_rounds,
            "R_avg_opp_CLINCH_landed":f(blue_strikes, "CLINCH_landed", 0) / r_rounds,
            "R_avg_GROUND_att":       f(red_strikes,  "GROUND_att", 0) / r_rounds,
            "R_avg_GROUND_landed":    f(red_strikes,  "GROUND_landed", 0) / r_rounds,
            "R_avg_opp_GROUND_att":   f(blue_strikes, "GROUND_att", 0) / r_rounds,
            "R_avg_opp_GROUND_landed":f(blue_strikes, "GROUND_landed", 0) / r_rounds,
            "R_avg_CTRL_time(seconds)":     f(red_totals,  "CTRL", 0) / r_rounds,
            "R_avg_opp_CTRL_time(seconds)": f(blue_totals, "CTRL", 0) / r_rounds,
            "R_total_time_fought(seconds)": time_secs,
            "R_total_rounds_fought":  rounds,
            "R_total_title_bouts":    f(red_profile, "total_title_bouts", 0),
            "R_current_win_streak":   f(red_profile, "current_win_streak", 0),
            "R_current_lose_streak":  f(red_profile, "current_lose_streak", 0),
            "R_longest_win_streak":   f(red_profile, "longest_win_streak", 0),
            "R_wins":                 f(red_profile, "wins", 0),
            "R_losses":               f(red_profile, "losses", 0),
            "R_draw":                 f(red_profile, "draws", 0),
            "R_win_by_Decision_Majority":  f(red_profile, "dec_majority", 0),
            "R_win_by_Decision_Split":     f(red_profile, "dec_split", 0),
            "R_win_by_Decision_Unanimous": f(red_profile, "dec_unanimous", 0),
            "R_win_by_KO/TKO":        f(red_profile, "win_by_ko", 0),
            "R_win_by_Submission":    f(red_profile, "win_by_sub", 0),
            "R_win_by_TKO_Doctor_Stoppage": 0,
            "R_Stance":               f(red_profile, "stance", ""),
            "R_Height_cms":           f(red_profile, "height_cms"),
            "R_Reach_cms":            f(red_profile, "reach_cms"),
            "R_Weight_lbs":           f(red_profile, "weight_lbs"),
            "B_age":                  calc_age(blue_profile.get("dob"), fight_date_obj),
            "R_age":                  calc_age(red_profile.get("dob"),  fight_date_obj),
        }

        return row

    except Exception as e:
        safe_print(f"  ✗ Error parsing fight {fight_url}: {e}")
        return None


# ── Step 5: Main orchestrator ─────────────────────────────────────────────────

def run():
    print("⚡ UFC New Fight Scraper")
    print(f"  Scraping fights after: {CUTOFF}\n")

    # Get events
    print("📋 Step 1: Getting event list...")
    events = get_event_urls_after(CUTOFF)
    if not events:
        print("  No events found!")
        return

    # Collect all fight URLs
    print(f"\n🔗 Step 2: Getting fight URLs from {len(events)} events...")
    all_fights = []
    for event_date, event_url in events:
        fight_urls = get_fight_urls_from_event(event_url)
        for fu in fight_urls:
            all_fights.append((fu, event_date, event_url))
        time.sleep(0.3)

    print(f"  Found {len(all_fights)} fights to scrape\n")

    # Scrape fights
    print(f"🥊 Step 3: Scraping fight details...")
    rows   = []
    failed = 0

    for i, (fight_url, event_date, event_name) in enumerate(all_fights):
        row = parse_fight(fight_url, event_date, event_name)
        if row:
            rows.append(row)
        else:
            failed += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(all_fights):
            pct = round((i + 1) / len(all_fights) * 100)
            safe_print(f"  {i+1}/{len(all_fights)} ({pct}%) — {len(rows)} ok, {failed} failed")

        time.sleep(0.5)  # Be polite to UFCStats

    print(f"\n  ✅ Scraped {len(rows)} new fights\n")

    if not rows:
        print("  No fights scraped — check your internet connection or UFCStats availability")
        return

    # Save new fights
    new_df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    new_df.to_csv(OUTPUT_NEW, index=False)
    print(f"💾 Step 4: Saved {len(new_df)} new fights → {OUTPUT_NEW}")

    # Merge with existing Kaggle data
    print(f"\n🔀 Step 5: Merging with existing dataset...")
    if os.path.exists(KAGGLE_CSV):
        kaggle_df = pd.read_csv(KAGGLE_CSV)
        print(f"  Kaggle dataset: {len(kaggle_df)} fights")
        print(f"  New fights:     {len(new_df)} fights")

        # Align columns
        all_cols = list(kaggle_df.columns)
        for col in all_cols:
            if col not in new_df.columns:
                new_df[col] = np.nan

        new_df = new_df[all_cols]
        combined = pd.concat([kaggle_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["R_fighter", "B_fighter", "date"])
        combined = combined.sort_values("date").reset_index(drop=True)

        combined.to_csv(OUTPUT_ALL, index=False)
        print(f"  ✅ Combined: {len(combined)} total fights → {OUTPUT_ALL}")
    else:
        print(f"  ⚠ Kaggle CSV not found at {KAGGLE_CSV} — saving new fights only")
        new_df.to_csv(OUTPUT_ALL, index=False)

    print(f"\n🎉 Done! Now retrain the model:")
    print(f"   python model.py")
    print(f"   (change data_pathπυτηον  to r'data\\combined_data.csv')")


if __name__ == "__main__":
    run()