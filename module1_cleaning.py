"""Module 1 - Data cleaning. Input: raw HHS UAC CSV. Output: outputs/clean.csv"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
def find_csv():
    """Use the CSV that sits in this folder (or in a 'data' subfolder)."""
    for folder in (BASE, BASE / "data"):
        hits = sorted(folder.glob("*.csv")) if folder.exists() else []
        if hits: return hits[0]
    raise FileNotFoundError(f"Put the HHS CSV file in this folder: {BASE}")

RAW = find_csv()
OUT = BASE / "outputs"; OUT.mkdir(exist_ok=True)

COLS = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "intake",
    "Children in CBP custody": "cbp_custody",
    "Children transferred out of CBP custody": "transfers",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "discharges",
}

def clean(path=RAW):
    df = pd.read_csv(path).rename(columns=COLS)
    log = {"raw_rows": len(df)}
    df = df.dropna(how="all"); log["after_dropping_blank_rows"] = len(df)
    df["hhs_care"] = pd.to_numeric(df["hhs_care"].astype(str).str.replace(",", ""), errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df = df.dropna(subset=["date"])
    log["duplicate_dates_removed"] = int(df["date"].duplicated().sum())
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    num = [c for c in COLS.values() if c != "date"]
    log["missing_values"] = int(df[num].isna().sum().sum())
    df[num] = df[num].astype(int)
    log["negative_values"] = int((df[num] < 0).sum().sum())
    log["clean_rows"] = len(df)
    log["date_min"] = str(df["date"].min().date()); log["date_max"] = str(df["date"].max().date())
    log["median_gap_days"] = float(df["date"].diff().dt.days.median())
    log["gaps_over_3_days"] = int((df["date"].diff().dt.days > 3).sum())
    return df, log

if __name__ == "__main__":
    df, log = clean()
    df.to_csv(OUT / "clean.csv", index=False)
    pd.Series(log).to_json(OUT / "cleaning_log.json", indent=2)
    for k, v in log.items(): print(f"{k:28s} {v}")
    print("\nSaved outputs/clean.csv")
