"""
scraper_fast.py — parallel UFC scraper, ~10x faster than sequential
Gets ALL stats: wins, losses, KO avg, sub avg, str accuracy, TD%, age, reach, etc.

Run:  python scraper_fast.py
"""

import requests
from bs4 import BeautifulSoup
import joblib
import time
import re
import os
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BASE    = "http://www.ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UFCResearch/1.0)"}

# Thread-safe print lock
print_lock = threading.Lock()

def safe_print(*args):
    with print_lock:
        print(*args)


def get_soup(url, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            time.sleep(1.5)
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
    """
    Parse fight history table to get:
    - win_streak
    - ko_avg     (KO wins / total fights)
    - sub_avg    (Sub wins / total fights)
    - dec_avg    (Decision wins / total fights)
    - finish_rate (KO + Sub) / total fights
    """
    rows     = soup.select("table.b-fight-details__table tbody tr")
    results  = []
    ko_wins  = 0
    sub_wins = 0
    dec_wins = 0

    for row in rows:
        cols = row.select("td")
        if len(cols) < 2:
            continue

        outcome = cols[0].get_text(strip=True).upper()
        if outcome not in ("WIN", "LOSS", "DRAW", "NC", "W", "L", "D"):
            continue
        outcome = "W" if outcome == "WIN" else ("L" if outcome == "LOSS" else outcome[0])

        # Try multiple column indexes to find the method
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
                ko_wins += 1
            elif "SUB" in method:
                sub_wins += 1
            elif "DEC" in method:
                dec_wins += 1

    total = len(results)
    if total == 0:
        return {}

    win_streak = 0
    for r in results:
        if r == "W":
            win_streak += 1
        else:
            break

    total_wins = results.count("W")

    return {
        "win_streak":   win_streak,
        "ko_avg":       round(ko_wins  / total, 3),
        "sub_avg":      round(sub_wins / total, 3),
        "dec_avg":      round(dec_wins / total, 3),
        "finish_rate":  round((ko_wins + sub_wins) / total, 3),
        "total_fights": total,
        "total_wins":   total_wins,
        "total_losses": results.count("L"),
    }


def get_weight_class(soup):
    """Get weight class by scanning fight history for known weight class names."""
    WEIGHT_CLASSES = [
        "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
        "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
        "Heavyweight", "Super Heavyweight", "Open Weight", "Catch Weight",
        "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
        "Women's Featherweight",
    ]
    fight_rows = soup.select("table.b-fight-details__table tbody tr")
    for row in fight_rows:
        for col in row.select("td"):
            text = col.get_text(strip=True)
            for wc in WEIGHT_CLASSES:
                if wc.lower() in text.lower():
                    return wc
    return None


def parse_fighter(url):
    """Scrape a single fighter page and return full stats dict."""
    soup = get_soup(url)
    if not soup:
        return None

    try:
        # Name
        name_el = soup.select_one("span.b-content__title-highlight")
        if not name_el:
            return None
        name = name_el.text.strip()

        # Record
        record_el = soup.select_one("span.b-content__title-record")
        wins, losses = 0, 0
        if record_el:
            m = re.search(r"(\d+)-(\d+)", record_el.text)
            if m:
                wins   = int(m.group(1))
                losses = int(m.group(2))

        # Stat box (height, reach, weight, stance, dob)
        box = get_stat_box(soup)

        height_cms = to_cm(box.get("height"))
        reach_cms  = to_cm(box.get("reach"))
        weight_lbs = to_lbs(box.get("weight"))
        stance     = box.get("stance") or None
        age        = calc_age(box.get("dob"))

        # Strike & TD accuracy from stat box
        str_acc = to_float(box.get("str. acc.", box.get("sig. str. acc.")))
        td_acc  = to_float(box.get("td acc."))
        str_def = to_float(box.get("str. def.", box.get("sig. str. def.")))
        td_def  = to_float(box.get("td def."))

        # Normalize percentages to 0-1
        if str_acc and str_acc > 1: str_acc = round(str_acc / 100, 4)
        if td_acc  and td_acc  > 1: td_acc  = round(td_acc  / 100, 4)
        if str_def and str_def > 1: str_def = round(str_def / 100, 4)
        if td_def  and td_def  > 1: td_def  = round(td_def  / 100, 4)

        # Strikes landed/absorbed per min
        slpm   = to_float(box.get("slpm",  box.get("sig. str. landed", None)))
        sapm   = to_float(box.get("sapm",  box.get("sig. str. absorbed", None)))
        td_avg = to_float(box.get("td avg."))

        # Fight history
        history      = parse_fight_history(soup)
        weight_class = get_weight_class(soup)

        return {
            "name":         name,
            "wins":         wins,
            "losses":       losses,
            "height_cms":   height_cms,
            "reach_cms":    reach_cms,
            "weight_lbs":   weight_lbs,
            "stance":       stance,
            "age":          age,
            "sig_str_acc":  str_acc,
            "sig_str_def":  str_def,
            "td_acc":       td_acc,
            "td_def":       td_def,
            "slpm":         slpm,     # strikes landed per min
            "sapm":         sapm,     # strikes absorbed per min
            "td_avg":       td_avg,   # takedowns per 15 min
            "win_streak":   history.get("win_streak"),
            "ko_avg":       history.get("ko_avg"),
            "sub_avg":      history.get("sub_avg"),
            "dec_avg":      history.get("dec_avg"),
            "finish_rate":  history.get("finish_rate"),
            "total_fights": history.get("total_fights"),
            "weight_class": weight_class,
        }

    except Exception as e:
        return None


def get_all_fighter_urls():
    """Collect all fighter URLs from a-z listing pages."""
    urls = []
    for char in "abcdefghijklmnopqrstuvwxyz":
        safe_print(f"  Scanning '{char}'...")
        soup = get_soup(f"{BASE}/statistics/fighters?char={char}&page=all")
        if not soup:
            continue
        for row in soup.select("table.b-statistics__table tbody tr"):
            link = row.select_one("td a")
            if link and link.get("href"):
                urls.append(link["href"])
        time.sleep(0.3)
    return list(set(urls))


def scrape_parallel(urls, max_workers=10):
    """
    Scrape all fighter URLs in parallel using a thread pool.
    max_workers=10 means 10 fighters scraped simultaneously.
    """
    results  = {}
    failed   = 0
    total    = len(urls)
    done     = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(parse_fighter, url): url for url in urls}

        for future in as_completed(future_to_url):
            done += 1
            try:
                fighter = future.result()
                if fighter and fighter.get("name"):
                    results[fighter["name"]] = fighter
                else:
                    failed += 1
            except Exception:
                failed += 1

            if done % 100 == 0 or done == total:
                pct = round(done / total * 100)
                safe_print(f"  {done}/{total} ({pct}%) — {len(results)} ok, {failed} failed")

    return results


def run():
    print("⚡ UFC Fast Parallel Scraper\n")

    print("📋 Step 1: Collecting fighter URLs (a-z)...")
    urls = get_all_fighter_urls()
    print(f"  Found {len(urls)} fighter pages\n")

    print(f"👤 Step 2: Scraping all fighters in parallel (10 threads)...")
    start = time.time()
    scraped = scrape_parallel(urls, max_workers=10)
    elapsed = round(time.time() - start)
    print(f"\n  ✅ Scraped {len(scraped)} fighters in {elapsed}s\n")

    print("💾 Step 3: Saving fighters.joblib...")
    joblib.dump(scraped, "fighters.joblib")
    print(f"  ✅ Saved {len(scraped)} fighters\n")

    # Quick check
    conor = scraped.get("Conor McGregor", {})
    print(f"  🔍 Conor McGregor check:")
    print(f"     Record:     {conor.get('wins')}-{conor.get('losses')}")
    print(f"     Win streak: {conor.get('win_streak')}")
    print(f"     KO avg:     {conor.get('ko_avg')}")
    print(f"     Str. acc:   {conor.get('sig_str_acc')}")

    print("\n🎉 Done! Restart uvicorn to load updated data:")
    print("   uvicorn main:app --reload")


if __name__ == "__main__":
    run()