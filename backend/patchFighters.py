"""
patch_fighters.py — patches existing fighters.joblib
Only re-scrapes fighters who are missing key fields.
Much faster than running the full scraper again!

Run:  python patch_fighters.py
"""

import requests
from bs4 import BeautifulSoup
import joblib
import time
import re
import math
from datetime import date, datetime

BASE    = "http://www.ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UFCResearch/1.0)"}


def get_soup(url, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"  ⚠ Retry {i+1}/{retries} — {e}")
            time.sleep(2)
    return None


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
    m2 = re.search(r"(\d+)", str(v))
    if m2:
        return round(int(m2.group(1)) * 2.54, 1)
    return None


def to_lbs(v):
    if not v or str(v).strip() in ("--", ""):
        return None
    m = re.search(r"(\d+\.?\d*)", str(v))
    return float(m.group(1)) if m else None


def calc_age(dob_str):
    if not dob_str or dob_str.strip() in ("--", ""):
        return None
    try:
        dob   = datetime.strptime(dob_str.strip(), "%b %d, %Y")
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except:
        return None


def get_stat_box(soup):
    result = {}
    for li in soup.select("li.b-list__box-list-item"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            key, _, val = text.partition(":")
            result[key.strip().lower()] = val.strip()
    return result


def parse_fight_history(soup):
    rows     = soup.select("table.b-fight-details__table tbody tr")
    results  = []
    ko_wins  = 0
    sub_wins = 0

    for row in rows:
        cols = row.select("td")
        if len(cols) < 8:
            continue
        outcome = cols[0].get_text(strip=True).upper()
        method  = cols[7].get_text(strip=True).upper()
        results.append(outcome)
        if outcome == "W":
            if "KO" in method or "TKO" in method:
                ko_wins += 1
            elif "SUB" in method:
                sub_wins += 1

    total = len(results)
    if total == 0:
        return {}

    win_streak = 0
    for r in results:
        if r == "W":
            win_streak += 1
        else:
            break

    return {
        "win_streak": win_streak,
        "ko_avg":     round(ko_wins  / total, 3),
        "sub_avg":    round(sub_wins / total, 3),
    }


def get_fighter_url(name):
    """Search ufcstats for a fighter by name and return their page URL."""
    # Try first letter search
    first_letter = name.strip()[0].lower()
    soup = get_soup(f"{BASE}/statistics/fighters?char={first_letter}&page=all")
    if not soup:
        return None
    for row in soup.select("table.b-statistics__table tbody tr"):
        cells = row.select("td")
        if len(cells) >= 2:
            first = cells[0].get_text(strip=True)
            last  = cells[1].get_text(strip=True)
            full  = f"{first} {last}".strip()
            if full.lower() == name.lower():
                link = row.select_one("td a")
                if link:
                    return link["href"]
    return None


def patch_fighter(existing: dict, url: str) -> dict:
    """Fetch fresh data and fill in missing fields only."""
    soup = get_soup(url)
    if not soup:
        return existing

    try:
        box     = get_stat_box(soup)
        history = parse_fight_history(soup)

        # Record
        record_el = soup.select_one("span.b-content__title-record")
        if record_el:
            m = re.search(r"(\d+)-(\d+)", record_el.text)
            if m:
                existing["wins"]   = int(m.group(1))
                existing["losses"] = int(m.group(2))

        # Fill missing fields
        if not existing.get("age"):
            existing["age"]        = calc_age(box.get("dob"))
        if not existing.get("weight_lbs"):
            existing["weight_lbs"] = to_lbs(box.get("weight"))
        if not existing.get("height_cms"):
            existing["height_cms"] = to_cm(box.get("height"))
        if not existing.get("reach_cms"):
            existing["reach_cms"]  = to_cm(box.get("reach"))
        if not existing.get("stance"):
            existing["stance"]     = box.get("stance")

        # Sig str acc
        if not existing.get("sig_str_acc"):
            v = to_float(box.get("str. acc.", box.get("sig. str. acc.")))
            existing["sig_str_acc"] = (v / 100) if v and v > 1 else v

        # TD acc
        if not existing.get("td_acc"):
            v = to_float(box.get("td acc."))
            existing["td_acc"] = (v / 100) if v and v > 1 else v

        # Calculated fields from fight history
        if history:
            if existing.get("win_streak") is None:
                existing["win_streak"] = history.get("win_streak")
            if existing.get("ko_avg") is None:
                existing["ko_avg"]     = history.get("ko_avg")
            if existing.get("sub_avg") is None:
                existing["sub_avg"]    = history.get("sub_avg")

        # Weight class from fight history table
        if not existing.get("weight_class"):
            fight_rows = soup.select("table.b-fight-details__table tbody tr")
            for row in fight_rows:
                cols = row.select("td")
                if len(cols) >= 7:
                    wc = cols[6].get_text(strip=True)
                    if wc and wc not in ("--", ""):
                        existing["weight_class"] = wc
                        break

    except Exception as e:
        print(f"  ✗ Patch error: {e}")

    return existing


def needs_patch(f: dict) -> bool:
    """Return True if fighter is missing important fields."""
    missing = [
        f.get("age")         is None,
        f.get("weight_lbs")  is None,
        f.get("weight_class") is None,
        f.get("win_streak")  is None,
        f.get("ko_avg")      is None,
        f.get("sub_avg")     is None,
    ]
    return sum(missing) >= 2  # patch if 2+ fields missing


def run():
    print("🔧 UFC Fighter Patcher\n")

    print("📦 Loading existing fighters.joblib...")
    fighters = joblib.load("fighters.joblib")
    print(f"  Loaded {len(fighters)} fighters\n")

    # Find who needs patching
    to_patch = {name: f for name, f in fighters.items() if needs_patch(f)}
    print(f"  {len(to_patch)} fighters need patching")
    print(f"  {len(fighters) - len(to_patch)} fighters already have full data\n")

    if not to_patch:
        print("✅ All fighters already have complete data!")
        return

    print("🕷️  Patching fighters...\n")
    patched = 0
    failed  = 0

    for i, (name, fighter) in enumerate(to_patch.items()):
        url = get_fighter_url(name)
        if url:
            fighters[name] = patch_fighter(fighter, url)
            patched += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            pct = round((i + 1) / len(to_patch) * 100)
            print(f"  {i+1}/{len(to_patch)} ({pct}%) — {patched} patched, {failed} not found")

        time.sleep(0.3)

    print(f"\n✅ Patched {patched} fighters ({failed} not found on ufcstats)")

    print("\n💾 Saving updated fighters.joblib...")
    joblib.dump(fighters, "fighters.joblib")
    print("✅ Done! Restart uvicorn to load updated data.")
    print("   uvicorn main:app --reload")


if __name__ == "__main__":
    run()