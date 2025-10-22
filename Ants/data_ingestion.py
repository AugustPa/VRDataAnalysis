# data_ingestion.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Iterable, Tuple
import os, json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# ----------------------------
# Config (simple & optional)
# ----------------------------
@dataclass(frozen=True)
class Config:
    parquet_root: Path = Path(os.getenv("GAS_PARQUET_ROOT", "/Users/apaula/ownCloud/stage_parquet_gray_angle_sweep"))

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ----------------------------
# Small utilities
# ----------------------------
def to_stable_trial_id(df: pd.DataFrame) -> pd.Series:
    base_cols = ['SourceFile', 'CurrentStep', 'CurrentTrial']
    if 'stepIndex' in df.columns:
        base_cols.append('stepIndex')
    key = df[base_cols].astype('string').fillna('<NA>')
    return pd.util.hash_pandas_object(key, index=False).astype('uint64')

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    cast_map = {
        'CurrentTrial': 'Int64',
        'CurrentStep': 'Int64',
        'VR': 'string',
        'stepName': 'string',
        'GameObjectPosX': 'float64',
        'GameObjectPosY': 'float64',
        'GameObjectPosZ': 'float64',
        'elapsed_time': 'float64',
        'Current Time': 'float64',  # or parse to datetime if that's correct in your data
    }
    for c, dt in cast_map.items():
        if c in df.columns:
            df[c] = df[c].astype(dt, errors='ignore')
    return df

# ----------------------------
# IO helpers
# ----------------------------
def write_parquet(df: pd.DataFrame, dest: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, dest, compression="zstd", use_dictionary=True, write_statistics=True)

def find_metadata(subdir: Path) -> Optional[Dict]:
    files = list(subdir.glob('*_FlyMetaData.json'))
    if not files:
        return None
    return json.loads(files[0].read_text())

def list_vr_csvs(subdir: Path) -> list[Path]:
    return sorted([p for p in subdir.glob('*.csv') if '_VR' in p.name])

# ----------------------------
# Ingestion (directory → parquet dataset)
# ----------------------------
def ingest_run_directory(
    run_root: Path,
    trim_seconds: float,
    parquet_root: Path,
    load_csv,                 # your existing loader(csv_path) -> DataFrame
    process_dataframe,        # your existing processor(df_loaded, trim_seconds) -> DataFrame
    *,
    quiet: bool = False
) -> None:
    subdirs = sorted([p for p in run_root.iterdir() if p.is_dir()])
    if not subdirs:
        if not quiet: print(f"No subdirectories in {run_root}")
        return

    ensure_dir(parquet_root)

    for subdir in subdirs:
        subfolder_name = subdir.name
        if not quiet: print(f"[ingest] {subfolder_name}")

        meta = find_metadata(subdir)
        experimenter_name = meta.get("ExperimenterName", "") if meta else ""
        comments = meta.get("Comments", "") if meta else ""
        vr_fly_map = {
            d.get("VR"): str(d.get("FlyID"))
            for d in (meta or {}).get("Flies", [])
            if d.get("VR") and d.get("FlyID")
        }

        csv_paths = list_vr_csvs(subdir)
        if not csv_paths:
            if not quiet: print(f"  (no CSVs) {subfolder_name}")
            continue

        for csv_path in csv_paths:
            df_loaded = load_csv(csv_path)
            df = process_dataframe(df_loaded, trim_seconds)

            if df.empty:
                if not quiet: print(f"  (empty) {csv_path.name}")
                continue

            # attach metadata
            df["ExperimenterName"] = experimenter_name
            df["Comments"] = comments
            if "VR" in df.columns:
                df["FlyID"] = df["VR"].map(vr_fly_map).astype("string")

            # stable ID, schema, partitions
            df["UniqueTrialID"] = to_stable_trial_id(df)
            df = enforce_schema(df)

            df["run"] = subfolder_name
            df["vr"] = df["VR"].astype("string").str.lower() if "VR" in df.columns else pd.NA
            df["flyid_part"] = df["FlyID"].fillna("unknown")

            vr_part = df["vr"].dropna().iloc[0] if df["vr"].notna().any() else "unknown"
            flyid_part = df["flyid_part"].dropna().iloc[0]

            dest_dir = ensure_dir(parquet_root / f"run={subfolder_name}" / f"vr={vr_part}" / f"flyid={flyid_part}")
            safe_name = csv_path.stem.replace(" ", "_")
            dest_file = dest_dir / f"{safe_name}.parquet"

            out_df = df.drop(columns=["flyid_part"], errors="ignore")
            write_parquet(out_df, dest_file)
            if not quiet: print(f"  wrote {dest_file}")

# ----------------------------
# Dataset loading (Parquet → pandas)
# ----------------------------
def _cast_table_strings_to_large(table: pa.Table, only_cols=None) -> pa.Table:
    fields = []
    for f in table.schema:
        if (only_cols is None or f.name in only_cols) and pa.types.is_string(f.type):
            fields.append(pa.field(f.name, pa.large_string()))
        else:
            fields.append(f)
    return table.cast(pa.schema(fields))

def load_dataset_subset_as_pandas(
    root: Path,
    wanted_cols: Optional[Iterable[str]] = None,
    cat_cols: Optional[Iterable[str]] = None,
    cat_threshold: int = 20_000
) -> pd.DataFrame:
    dataset = ds.dataset(str(root), format="parquet", partitioning="hive")
    cols = dataset.schema.names if wanted_cols is None else [c for c in wanted_cols if c in dataset.schema.names]
    table = dataset.to_table(columns=cols)
    table = _cast_table_strings_to_large(table, only_cols=cols)  # <- fixed
    df = table.to_pandas(types_mapper=pd.ArrowDtype)

    if "Current Time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Current Time"]):
        df["Current Time"] = pd.to_datetime(df["Current Time"], errors="coerce", utc=True)

    if cat_cols is None:
        cat_cols = ["VR", "vr", "FlyID", "stepName", "run"]
    for c in cat_cols:
        if c in df.columns:
            s = df[c].astype("string[python]")
            df[c] = s.astype("category") if s.nunique(dropna=True) <= cat_threshold else s
    return df

# ----------------------------
# Trial metrics & classification
# ----------------------------
def add_trial_id_and_displacement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    df = df.copy()

    if "UniqueTrialID" not in df.columns:
        df["UniqueTrialID"] = to_stable_trial_id(df)

    sort_cols = [c for c in ['UniqueTrialID', 'Current Time'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, kind='mergesort')

    for axis, col in zip(['x','y','z'], ['GameObjectPosX','GameObjectPosY','GameObjectPosZ']):
        df[f'delta_{axis}'] = df.groupby('UniqueTrialID')[col].diff() if col in df.columns else np.nan

    df['step_distance'] = np.sqrt(df['delta_x'].fillna(0)**2 + df['delta_y'].fillna(0)**2 + df['delta_z'].fillna(0)**2)

    if 'elapsed_time' in df.columns:
        df = df.sort_values(['UniqueTrialID', 'elapsed_time'], kind='mergesort')

    agg = (
        df.groupby('UniqueTrialID')
          .agg(first_x=('GameObjectPosX','first'),
               last_x =('GameObjectPosX','last'),
               first_z=('GameObjectPosZ','first'),
               last_z =('GameObjectPosZ','last'))
          .reset_index()
    )
    agg['TotalDisplacement'] = np.hypot(agg['last_x']-agg['first_x'], agg['last_z']-agg['first_z'])
    agg['TravelDirectionDeg'] = (np.degrees(np.arctan2(agg['last_x']-agg['first_x'],
                                                       agg['last_z']-agg['first_z'])) + 360) % 360

    path = (df.groupby('UniqueTrialID', observed=True)['step_distance']
              .sum().reset_index()
              .rename(columns={'step_distance':'TotalPathLength'}))

    return (df
        .merge(agg[['UniqueTrialID','TotalDisplacement','TravelDirectionDeg']], on='UniqueTrialID', how='left')
        .merge(path, on='UniqueTrialID', how='left')
    )

def classify_by_displacement(df: pd.DataFrame, min_disp: float = 0.0, max_disp: float = 50.0):
    if df.empty: return df, df, df, [], [], []
    td = df.groupby('UniqueTrialID', observed=True)['TotalDisplacement'].first().reset_index()
    stationary = td.loc[td['TotalDisplacement'] < min_disp, 'UniqueTrialID'].unique()
    normal     = td.loc[(td['TotalDisplacement'] >= min_disp) & (td['TotalDisplacement'] <= max_disp), 'UniqueTrialID'].unique()
    excessive  = td.loc[td['TotalDisplacement'] > max_disp, 'UniqueTrialID'].unique()
    return (df[df['UniqueTrialID'].isin(stationary)].reset_index(drop=True),
            df[df['UniqueTrialID'].isin(normal)].reset_index(drop=True),
            df[df['UniqueTrialID'].isin(excessive)].reset_index(drop=True),
            stationary, normal, excessive)

def classify_by_path_length(df: pd.DataFrame, min_length: float = 0.0, max_length: float = 50.0):
    if df.empty: return df, df, df, [], [], []
    pl = df.groupby('UniqueTrialID', observed=True)['TotalPathLength'].first().reset_index()
    stationary = pl.loc[pl['TotalPathLength'] < min_length, 'UniqueTrialID'].unique()
    normal     = pl.loc[(pl['TotalPathLength'] >= min_length) & (pl['TotalPathLength'] <= max_length), 'UniqueTrialID'].unique()
    excessive  = pl.loc[pl['TotalPathLength'] > max_length, 'UniqueTrialID'].unique()
    return (df[df['UniqueTrialID'].isin(stationary)].reset_index(drop=True),
            df[df['UniqueTrialID'].isin(normal)].reset_index(drop=True),
            df[df['UniqueTrialID'].isin(excessive)].reset_index(drop=True),
            stationary, normal, excessive)

# ----------------------------
# Convenience constants/exports
# ----------------------------
CORE_COLS = [
    "SourceFile", "CurrentStep", "CurrentTrial", "stepIndex",
    "VR", "FlyID", "stepName",
    "GameObjectPosX", "GameObjectPosY", "GameObjectPosZ",
    "elapsed_time", "Current Time",
    "UniqueTrialID", "run", "vr",
]
__all__ = [
    "Config", "ensure_dir",
    "to_stable_trial_id", "enforce_schema",
    "write_parquet", "find_metadata", "list_vr_csvs",
    "ingest_run_directory",
    "load_dataset_subset_as_pandas", "CORE_COLS",
    "add_trial_id_and_displacement",
    "classify_by_displacement", "classify_by_path_length",
]
