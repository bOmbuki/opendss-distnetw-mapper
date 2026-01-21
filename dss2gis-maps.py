"""
build_feeders_and_city_maps.py

Creates BOTH per-feeder AND city-wide artifacts from OpenDSS feeder folders:

Per-feeder (for each ./Cities/feeder_*):
  - Shapefiles:
      feeder_X_lines.shp
      feeder_X_buses.shp (includes voltage attrs if solve succeeds)
      feeder_X_substations.shp (subset of buses)
  - HTML maps:
      feeder_X.html            (lines + substations)
      feeder_X_heat.html       (voltage heat + colorbar + legend)
      feeder_X_bus.html        (bus voltage dots + colorbar + legend)
  - PNG:
      feeder_X_voltage.png     (basemap + lines + bus voltages + substations + legend)

City-wide combined:
  - Shapefiles:
      combined_lines.shp
      combined_buses.shp
      combined_substations.shp
  - HTML maps:
      combined_feeders.html    (all feeders colored, layer control)
      combined_heat.html       (combined voltage heat + colorbar)
  - PNG:
      city_voltage.png         (basemap + ALL feeder lines + ALL bus voltages + substations)
      city_topology.png        (basemap + ALL feeder lines + substations; cleaner “Figure-ready”)

Requirements:
  pip install pandas geopandas shapely folium pyproj opendssdirect matplotlib contextily

Notes:
  - BusCoords.csv accepted formats:
      (1) columns: bus,lon,lat
      (2) columns: bus,x,y with CONFIG["COORDS_CRS"] set (e.g., EPSG:26912)
      (3) headerless: bus,lon,lat
  - Substations are detected by bus names containing: bus0, source, sub, substation (edit in is_substation_name)
"""

from __future__ import annotations

import re
import hashlib
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd

# ------------------------- CONFIG -------------------------
CONFIG = {
    "BASE_DIR": r"./Cities",                 # Folder containing feeder_* subfolders
    "COORDS_CRS": None,                      # Set if BusCoords.csv uses x/y (e.g., "EPSG:26912")
    "TARGET_CRS": "EPSG:4326",               # WGS84 for output/maps
    "LINES_FILE": "lines.dss",
    "BUS_CSV_FILE": "BusCoords.csv",
    "MASTER_FILE": "master.dss",

    # Optional overrides; if None, auto center/zoom
    "MAP_CENTER": None,                      # (lat, lon) or None
    "MAP_ZOOM": None,                        # int or None (defaults ~12)

    # Fixed voltage bounds (used when ADAPTIVE_PER_FEEDER is False)
    "V_MIN": 0.95,
    "V_MAX": 1.02,

    # Heatmap rendering
    "HEAT_RADIUS": 18,
    "HEAT_BLUR": 22,
    "HEAT_MAX_ZOOM": 18,

    # HTML styling
    "HTML_LINE_WEIGHT": 5,
    "HTML_LINE_OPACITY": 0.95,
    "HTML_BUS_RADIUS": 4.5,
    "HTML_SUB_RADIUS": 7,

    # PNG styling
    "PNG_FIGSIZE_IN": (12, 12),
    "PNG_DPI": 300,
    "PNG_LINE_WIDTH": 1.4,
    "PNG_SUB_SIZE": 90,
    "PNG_BUS_SIZE": 10,

    # Voltage scaling
    "ADAPTIVE_PER_FEEDER": True,             # per-feeder min/max color scale
    "CITY_PNG_FIXED_BOUNDS": True,           # for city_voltage.png; True uses [V_MIN,V_MAX]

    # Basemap provider (contextily)
    "BASEMAP_PROVIDER": "CartoDB.Positron",  # e.g. "OpenStreetMap.Mapnik", "CartoDB.Positron"

    # Output folder name under BASE_DIR
    "OUT_DIRNAME": "outputs",
}
# ----------------------------------------------------------

# Dependencies
try:
    import geopandas as gpd
    from shapely.geometry import LineString
except Exception as e:
    raise SystemExit("Requires: pip install geopandas shapely") from e

try:
    import folium
    from folium.plugins import HeatMap
except Exception as e:
    raise SystemExit("Requires: pip install folium") from e

try:
    import pyproj as _pyproj
except Exception:
    _pyproj = None

try:
    import opendssdirect as odd
except Exception as e:
    raise SystemExit("Requires: pip install opendssdirect") from e

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D
    import contextily as cx
except Exception as e:
    raise SystemExit("Requires: pip install matplotlib contextily") from e


# ---------------------- UTILITIES ----------------------
def ensure_outputs_dir(base: Path) -> Path:
    out = base / CONFIG["OUT_DIRNAME"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def strip_comments_and_join_continuations(text: str) -> list[str]:
    lines, buf = [], ""
    for raw in text.splitlines():
        s = re.sub(r"//.*", "", raw)
        s = re.sub(r"!.*", "", s)
        s = s.strip()
        if not s:
            continue
        if s.endswith("~"):
            buf += s[:-1] + " "
            continue
        buf += s
        lines.append(buf)
        buf = ""
    if buf:
        lines.append(buf)
    return lines


def base_bus_name(bus: Optional[str]) -> Optional[str]:
    return str(bus).split(".")[0].strip().lower() if bus else None


def dss_path(p: Path) -> str:
    """Return DSS-safe quoted path (forward slashes)."""
    return f'"{p.resolve().as_posix()}"'


def stable_color_for_name(name: str) -> str:
    # A compact, readable palette for feeder differentiation on maps
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16)
    return palette[h % len(palette)]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_v(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def sanitize_for_shapefile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shapefile field names max ~10 chars; create truncated, unique names.
    """
    keep_order = list(df.columns)
    new_cols = []
    seen = set()
    for c in keep_order:
        lc = c.lower()
        if lc == "line_name":
            nc = "LINENAME"
        elif lc == "feeder":
            nc = "FEEDER"
        elif lc == "bus":
            nc = "BUS"
        elif lc == "vpu_min":
            nc = "VPU_MIN"
        elif lc == "vpu_mean":
            nc = "VPU_MEAN"
        elif lc == "vpua":
            nc = "VPU_A"
        elif lc == "vpub":
            nc = "VPU_B"
        elif lc == "vpuc":
            nc = "VPU_C"
        else:
            nc = re.sub(r"[^A-Za-z0-9_]+", "_", c)[:10].upper()
            if not nc:
                nc = "F_"
        base = nc
        k = 1
        while nc in seen:
            suffix = f"_{k}"
            nc = (base[: (10 - len(suffix))] + suffix)
            k += 1
        new_cols.append(nc)
        seen.add(nc)

    out = df.copy()
    out.columns = new_cols
    return out


# ---------------------- PARSERS ----------------------
def parse_lines_dss(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    logical = strip_comments_and_join_continuations(text)
    out = []
    for ln in logical:
        if re.search(r"(?i)\bnew\s+line\.\S+", ln):
            mname = re.search(r"(?i)new\s+line\.([^\s]+)", ln)
            name = mname.group(1) if mname else None
            kv = {}
            for tok in ln.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k.strip().lower()] = v.strip().strip('"').strip("'").rstrip(",")
            b1 = base_bus_name(kv.get("bus1"))
            b2 = base_bus_name(kv.get("bus2"))
            if b1 and b2:
                out.append({"line_name": name, "bus1": b1, "bus2": b2, "raw": ln})
    return pd.DataFrame(out)


def read_bus_coords(csv_path: Path, coords_crs: Optional[str], target_crs: str) -> pd.DataFrame:
    """
    Accepts:
      - header: bus,lon,lat
      - header: bus,x,y (requires coords_crs)
      - headerless: bus,lon,lat
    Returns DataFrame [bus, lon, lat] in target_crs.
    """
    try:
        df = pd.read_csv(csv_path)
        lower = {c.lower(): c for c in df.columns}

        if {"bus", "lon", "lat"}.issubset(lower):
            df = df.rename(columns={lower["bus"]: "bus", lower["lon"]: "lon", lower["lat"]: "lat"})
            df["bus"] = df["bus"].astype(str).apply(base_bus_name)
            return df[["bus", "lon", "lat"]]

        if {"bus", "x", "y"}.issubset(lower):
            if coords_crs is None:
                raise ValueError("Set CONFIG['COORDS_CRS'] for x/y CSVs.")
            if _pyproj is None:
                raise ValueError("pip install pyproj for x/y reprojection.")
            src = _pyproj.CRS(coords_crs)
            dst = _pyproj.CRS(target_crs)
            transformer = _pyproj.Transformer.from_crs(src, dst, always_xy=True)
            lon, lat = transformer.transform(
                df[lower["x"]].astype(float).values,
                df[lower["y"]].astype(float).values
            )
            out = pd.DataFrame({
                "bus": df[lower["bus"]].astype(str).apply(base_bus_name),
                "lon": lon,
                "lat": lat
            })
            return out[["bus", "lon", "lat"]]

        raise ValueError("Unknown BusCoords format.")
    except Exception:
        df = pd.read_csv(csv_path, header=None, names=["bus", "lon", "lat"])
        df["bus"] = df["bus"].astype(str).apply(base_bus_name)
        return df[["bus", "lon", "lat"]]


def build_lines_gdf(lines_df: pd.DataFrame, coords_df: pd.DataFrame, target_crs: str) -> "gpd.GeoDataFrame":
    m1 = (
        lines_df
        .merge(coords_df.rename(columns={"bus": "bus1"}), on="bus1", how="left")
        .rename(columns={"lon": "lon1", "lat": "lat1"})
    )
    m2 = (
        m1
        .merge(coords_df.rename(columns={"bus": "bus2"}), on="bus2", how="left")
        .rename(columns={"lon": "lon2", "lat": "lat2"})
    )
    m2 = m2.dropna(subset=["lon1", "lat1", "lon2", "lat2"]).copy()
    if m2.empty:
        return gpd.GeoDataFrame(columns=["line_name", "bus1", "bus2", "raw", "geometry"], geometry="geometry", crs=target_crs)

    m2["geometry"] = m2.apply(
        lambda r: LineString([(float(r["lon1"]), float(r["lat1"])), (float(r["lon2"]), float(r["lat2"]))]),
        axis=1
    )
    return gpd.GeoDataFrame(m2[["line_name", "bus1", "bus2", "raw", "geometry"]], geometry="geometry", crs=target_crs)


# ---------------------- DSS / SOLVE ----------------------
def build_auto_master(feeder_dir: Path) -> Path:
    """
    If master.dss doesn't exist, create a minimal compile file that tries common redirects.
    """
    lines = ["clear", "set defaultbasefrequency=60.0"]
    for fn in ["sources.dss", "lines.dss", "loads.dss", "transformers.dss", "capacitors.dss", "regcontrols.dss"]:
        p = feeder_dir / fn
        if p.exists():
            lines.append(f"redirect {dss_path(p)}")
    lines += ["calcv", "solve mode=snap"]
    tmp = feeder_dir / "_auto_master.dss"
    tmp.write_text("\n".join(lines), encoding="utf-8")
    return tmp


def solve_feeder_and_get_bus_voltages(master_or_auto: Path, bus_coords_csv: Optional[Path]) -> pd.DataFrame:
    """
    Returns DataFrame:
      [bus, Vpu_min, Vpu_mean, VpuA, VpuB, VpuC]
    """
    feeder_dir = master_or_auto.parent

    odd.Basic.ClearAll()
    odd.Text.Command("Clear")
    odd.Text.Command(f"Set DataPath={dss_path(feeder_dir)}")
    odd.Text.Command(f"Compile {dss_path(master_or_auto)}")

    if bus_coords_csv and bus_coords_csv.exists():
        odd.Text.Command(f"BusCoords {dss_path(bus_coords_csv)}")
        odd.Text.Command("CalcVoltageBases")

    odd.Solution.Solve()

    rows = []
    for bus_name in odd.Circuit.AllBusNames():
        odd.Circuit.SetActiveBus(bus_name)
        pu = odd.Bus.puVmagAngle()
        mags = [pu[i] for i in range(0, len(pu), 2)] if pu else []
        mags = [m for m in mags if pd.notna(m)]
        if not mags:
            continue

        vpu_min = float(min(mags))
        vpu_mean = float(sum(mags) / len(mags))
        vA = mags[0] if len(mags) >= 1 else None
        vB = mags[1] if len(mags) >= 2 else None
        vC = mags[2] if len(mags) >= 3 else None

        rows.append({
            "bus": base_bus_name(bus_name),
            "Vpu_min": vpu_min,
            "Vpu_mean": vpu_mean,
            "VpuA": vA,
            "VpuB": vB,
            "VpuC": vC
        })

    return pd.DataFrame(rows)


# ---------------------- MAP HELPERS (FOLIUM) ----------------------
def make_map(center: Tuple[float, float], zoom: int = 12):
    return folium.Map(location=[center[0], center[1]], zoom_start=zoom, tiles="CartoDB Positron")


def add_lines_layer_to_map(m, gdf_lines, color: str, feeder: str):
    fg = folium.FeatureGroup(name=f"Feeder: {feeder}", show=True)
    for _, row in gdf_lines.iterrows():
        coords = [
            (row.geometry.coords[0][1], row.geometry.coords[0][0]),
            (row.geometry.coords[-1][1], row.geometry.coords[-1][0]),
        ]
        folium.PolyLine(
            coords,
            color=color,
            weight=CONFIG["HTML_LINE_WEIGHT"],
            opacity=CONFIG["HTML_LINE_OPACITY"],
            tooltip=f"{feeder} • {row.get('line_name','')}",
        ).add_to(fg)
    fg.add_to(m)


def is_substation_name(busname: str) -> bool:
    b = (busname or "").lower()
    keys = ["bus0", "source", "sub", "substation"]
    return any(k in b for k in keys)


def add_substation_markers(m, buses_df: pd.DataFrame):
    df = buses_df[buses_df["bus"].apply(is_substation_name)].copy()
    for _, r in df.iterrows():
        folium.CircleMarker(
            location=(float(r["lat"]), float(r["lon"])),
            radius=CONFIG["HTML_SUB_RADIUS"],
            color="#1f78ff",
            fill=True,
            fill_color="#1f78ff",
            fill_opacity=1.0,
            tooltip=f"Substation: {r['bus']}",
        ).add_to(m)


def add_symbol_legend_html(m):
    html = """
    <div style="
      position: absolute; bottom: 16px; left: 16px; z-index: 9999;
      background: rgba(255,255,255,0.92); padding: 8px 10px;
      border: 1px solid #888; border-radius: 8px; font: 12px/1.2 Arial, sans-serif;">
      <div style="margin-bottom:4px; font-weight:600;">Legend</div>
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
        <div style="width:18px; height:18px; border-radius:50%; background:#1f78ff;"></div>
        <span>Substation</span>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:28px; height:0; border-top: 4px solid #555;"></div>
        <span>Lines</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def add_continuous_colorbar(m, vmin: float, vmax: float, palette: str, label: str):
    # palette: 'heat' (green->yellow->orange->red where red = LOW voltage) or 'bus' (red->yellow for dots)
    if palette == "heat":
        gradient_css = "linear-gradient(to top, #d7191c 0%, #fdae61 33%, #fee08b 66%, #1a9641 100%)"
    else:
        gradient_css = "linear-gradient(to top, #d7191c 0%, #fdae61 50%, #ffffbf 100%)"

    html = f"""
    <div style="position:absolute; bottom:16px; right:16px; z-index:9999;
         background: rgba(255,255,255,0.92); padding:10px 12px; border:1px solid #888; border-radius:8px;
         box-shadow:0 1px 3px rgba(0,0,0,0.25); font-family:Arial, sans-serif; font-size:12px;">
      <div style="margin-bottom:6px; text-align:center;"><b>{label}</b></div>
      <div style="display:flex; align-items:stretch; gap:8px;">
        <div style="width:16px; height:160px; background: {gradient_css}; border:1px solid #666;"></div>
        <div style="display:flex; flex-direction:column; justify-content:space-between; height:160px;">
          <div>{vmax:.3f} pu</div>
          <div>{(vmin + (vmax-vmin)*0.75):.3f} pu</div>
          <div>{(vmin + (vmax-vmin)*0.50):.3f} pu</div>
          <div>{(vmin + (vmax-vmin)*0.25):.3f} pu</div>
          <div>{vmin:.3f} pu</div>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def add_bus_markers_colored_by_voltage(m, buses_df: pd.DataFrame, lo: float, hi: float):
    # red->yellow scale
    red = (215, 25, 28)
    yellow = (255, 255, 191)

    def rgb_to_hex(c):
        return "#{:02x}{:02x}{:02x}".format(*c)

    def lerp(c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    for _, r in buses_df.iterrows():
        v = r.get("Vpu_min", None)
        if pd.isna(v):
            continue
        t = normalize_v(float(v), lo, hi)
        col = rgb_to_hex(lerp(red, yellow, t))
        folium.CircleMarker(
            location=(float(r["lat"]), float(r["lon"])),
            radius=CONFIG["HTML_BUS_RADIUS"],
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.9,
            tooltip=f"Bus: {r['bus']} • {float(v):.3f} pu (min)",
        ).add_to(m)


# ---------------------- PNG HELPERS (MATPLOTLIB) ----------------------
def _get_ctx_provider(path: str):
    prov = cx.providers
    for part in path.split("."):
        prov = getattr(prov, part)
    return prov


def _safe_bounds(gdf):
    # returns (xmin, ymin, xmax, ymax) in EPSG:3857
    if gdf.empty:
        return None
    b = gdf.total_bounds
    return (b[0], b[1], b[2], b[3])


def plot_voltage_png(
    title: str,
    gdf_lines_wgs84: "gpd.GeoDataFrame",
    buses_df_wgs84: pd.DataFrame,
    subs_df_wgs84: pd.DataFrame,
    out_path: Path,
    vmin: Optional[float],
    vmax: Optional[float],
    show_bus_voltages: bool = True,
):
    """
    City or feeder PNG:
      - basemap
      - lines
      - buses colored by Vpu_min (optional)
      - substations
      - legend + colorbar (if voltages shown)
    """
    fig, ax = plt.subplots(figsize=CONFIG["PNG_FIGSIZE_IN"])
    ax.set_axis_off()

    # Project
    lines3857 = gdf_lines_wgs84.to_crs(3857)

    # Basemap bounds come from lines (fallback to buses)
    bounds = _safe_bounds(lines3857)

    # Lines
    lines3857.plot(ax=ax, color="dimgray", linewidth=CONFIG["PNG_LINE_WIDTH"], alpha=0.85, zorder=2)

    # Substations
    if not subs_df_wgs84.empty:
        gdf_subs = gpd.GeoDataFrame(
            subs_df_wgs84.copy(),
            geometry=gpd.points_from_xy(subs_df_wgs84["lon"], subs_df_wgs84["lat"]),
            crs=CONFIG["TARGET_CRS"],
        ).to_crs(3857)
        ax.scatter(
            gdf_subs.geometry.x, gdf_subs.geometry.y,
            s=CONFIG["PNG_SUB_SIZE"], marker="^",
            facecolor="#1f78ff", edgecolor="white", linewidth=0.8,
            zorder=4,
        )
        if bounds is None:
            bounds = _safe_bounds(gdf_subs)

    # Buses (voltages)
    sc = None
    if show_bus_voltages:
        buses_with_v = buses_df_wgs84[buses_df_wgs84["Vpu_min"].notna()].copy()
        if not buses_with_v.empty:
            gdf_buses = gpd.GeoDataFrame(
                buses_with_v,
                geometry=gpd.points_from_xy(buses_with_v["lon"], buses_with_v["lat"]),
                crs=CONFIG["TARGET_CRS"],
            ).to_crs(3857)

            v = gdf_buses["Vpu_min"].astype(float)
            if vmin is None or vmax is None:
                vmin = float(v.min())
                vmax = float(v.max())
            if vmax <= vmin:
                vmax = vmin + 1e-3

            cmap = plt.cm.plasma
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            sc = ax.scatter(
                gdf_buses.geometry.x, gdf_buses.geometry.y,
                c=v, cmap=cmap, norm=norm,
                s=CONFIG["PNG_BUS_SIZE"], linewidths=0,
                zorder=3,
            )
            if bounds is None:
                bounds = _safe_bounds(gdf_buses)

    # Set bounds with padding
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        pad_x = (xmax - xmin) * 0.05
        pad_y = (ymax - ymin) * 0.05
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    # Basemap
    try:
        provider = _get_ctx_provider(CONFIG["BASEMAP_PROVIDER"])
        cx.add_basemap(ax, source=provider, attribution_size=6, zorder=1)
    except Exception as e:
        warnings.warn(f"Basemap provider failed ({e}); continuing without tiles.")

    # Title
    ax.set_title(title, fontsize=14, pad=12)

    # Colorbar
    if show_bus_voltages and sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label("Bus Voltage (p.u.)", fontsize=11)

    # Legend
    handles = [
        Line2D([0], [0], color="dimgray", lw=CONFIG["PNG_LINE_WIDTH"] + 0.6, label="Lines"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1f78ff",
               markeredgecolor="white", markersize=10, linestyle="None", label="Substation"),
    ]
    if show_bus_voltages:
        handles.insert(
            0,
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#9b59b6",
                   markeredgecolor="gray", markersize=7, linestyle="None", label="Bus"),
        )

    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="gray", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=CONFIG["PNG_DPI"], facecolor="white")
    plt.close(fig)


# ---------------------- MAIN PIPELINE ----------------------
def build_everything():
    base = Path(CONFIG["BASE_DIR"]).expanduser().resolve()
    out_root = ensure_outputs_dir(base)

    feeders = sorted([p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("feeder_")])
    if not feeders:
        raise SystemExit(f"No feeder_* folders found in {base}")

    # Shared BusCoords (optional)
    shared_coords_path = base / CONFIG["BUS_CSV_FILE"]
    shared_coords_df = read_bus_coords(shared_coords_path, CONFIG["COORDS_CRS"], CONFIG["TARGET_CRS"]) if shared_coords_path.exists() else None

    summary_rows: List[Dict] = []

    combined_lines: List[gpd.GeoDataFrame] = []
    combined_bus_attrs: List[pd.DataFrame] = []
    combined_subs: List[pd.DataFrame] = []
    folium_layers: List[Tuple[str, gpd.GeoDataFrame, str]] = []

    for feeder_path in feeders:
        feeder = feeder_path.name
        color = stable_color_for_name(feeder)

        lines_path = feeder_path / CONFIG["LINES_FILE"]
        bus_csv_path = feeder_path / CONFIG["BUS_CSV_FILE"]
        master_path = feeder_path / CONFIG["MASTER_FILE"]

        if not lines_path.exists():
            warnings.warn(f"{feeder}: missing {CONFIG['LINES_FILE']} — skipping feeder")
            summary_rows.append({"feeder": feeder, "status": "no lines.dss", "min": None, "max": None, "mean": None})
            continue

        # coords
        if bus_csv_path.exists():
            coords_df = read_bus_coords(bus_csv_path, CONFIG["COORDS_CRS"], CONFIG["TARGET_CRS"])
            used_buscoords = bus_csv_path
        elif shared_coords_df is not None:
            coords_df = shared_coords_df
            used_buscoords = shared_coords_path
        else:
            warnings.warn(f"{feeder}: no BusCoords.csv (per-feeder or shared) — skipping feeder")
            summary_rows.append({"feeder": feeder, "status": "no BusCoords.csv", "min": None, "max": None, "mean": None})
            continue

        # build lines geometry
        lines_df = parse_lines_dss(lines_path)
        gdf_lines = build_lines_gdf(lines_df, coords_df, CONFIG["TARGET_CRS"])
        if gdf_lines.empty:
            warnings.warn(f"{feeder}: no valid line segments after join — skipping feeder")
            summary_rows.append({"feeder": feeder, "status": "no line geoms", "min": None, "max": None, "mean": None})
            continue

        # solve
        master = master_path if master_path.exists() else build_auto_master(feeder_path)
        feeder_ok = True
        try:
            bus_volt_df = solve_feeder_and_get_bus_voltages(master, used_buscoords)
        except Exception as e:
            warnings.warn(f"{feeder}: DSS solve failed: {e}")
            feeder_ok = False
            bus_volt_df = pd.DataFrame(columns=["bus", "Vpu_min", "Vpu_mean", "VpuA", "VpuB", "VpuC"])

        buses_join = coords_df.merge(bus_volt_df, on="bus", how="left")
        subs_df = buses_join[buses_join["bus"].apply(is_substation_name)].copy()

        # per-feeder outputs folder
        feeder_out = out_root / feeder
        feeder_out.mkdir(parents=True, exist_ok=True)

        # shapefiles
        gdf_lines_out = gdf_lines.assign(feeder=feeder).copy()
        sanitize_lines = sanitize_for_shapefile(gdf_lines_out.drop(columns=["geometry"]).copy())
        gpd.GeoDataFrame(
            sanitize_lines.join(gdf_lines_out[["geometry"]]),
            geometry="geometry",
            crs=CONFIG["TARGET_CRS"],
        ).to_file(feeder_out / f"{feeder}_lines.shp")

        gdf_buses = gpd.GeoDataFrame(
            buses_join.assign(feeder=feeder),
            geometry=gpd.points_from_xy(buses_join["lon"], buses_join["lat"]),
            crs=CONFIG["TARGET_CRS"],
        )
        sanitize_buses = sanitize_for_shapefile(gdf_buses.drop(columns=["geometry"]).copy())
        gpd.GeoDataFrame(
            sanitize_buses.join(gdf_buses[["geometry"]]),
            geometry="geometry",
            crs=CONFIG["TARGET_CRS"],
        ).to_file(feeder_out / f"{feeder}_buses.shp")

        if not subs_df.empty:
            gdf_subs = gpd.GeoDataFrame(
                subs_df.assign(feeder=feeder),
                geometry=gpd.points_from_xy(subs_df["lon"], subs_df["lat"]),
                crs=CONFIG["TARGET_CRS"],
            )
            sanitize_subs = sanitize_for_shapefile(gdf_subs.drop(columns=["geometry"]).copy())
            gpd.GeoDataFrame(
                sanitize_subs.join(gdf_subs[["geometry"]]),
                geometry="geometry",
                crs=CONFIG["TARGET_CRS"],
            ).to_file(feeder_out / f"{feeder}_substations.shp")

        # map center/zoom
        center = CONFIG["MAP_CENTER"] or (float(coords_df["lat"].mean()), float(coords_df["lon"].mean()))
        zoom = CONFIG["MAP_ZOOM"] or 12

        # base HTML
        m_base = make_map(center, zoom)
        add_lines_layer_to_map(m_base, gdf_lines, color, feeder)
        add_substation_markers(m_base, buses_join)
        add_symbol_legend_html(m_base)
        folium.LayerControl(collapsed=False).add_to(m_base)
        m_base.save(str(feeder_out / f"{feeder}.html"))

        # voltage bounds
        valid_v = buses_join["Vpu_min"].dropna().astype(float)
        v_min = float(valid_v.min()) if not valid_v.empty else None
        v_max = float(valid_v.max()) if not valid_v.empty else None
        v_mean = float(valid_v.mean()) if not valid_v.empty else None

        # choose color bounds
        if not valid_v.empty:
            if CONFIG["ADAPTIVE_PER_FEEDER"]:
                lo, hi = v_min, v_max
                if hi <= lo:
                    hi = lo + 1e-3
            else:
                lo, hi = CONFIG["V_MIN"], CONFIG["V_MAX"]

            # HEAT HTML (lower voltage = “hotter”)
            m_heat = make_map(center, zoom)
            add_lines_layer_to_map(m_heat, gdf_lines, color, feeder)

            df_heat = buses_join[["lat", "lon", "Vpu_min"]].dropna().copy()
            if not df_heat.empty and hi > lo:
                df_heat["norm"] = df_heat["Vpu_min"].apply(lambda v: clamp((float(v) - lo) / (hi - lo), 0, 1))
                df_heat["weight"] = 1.0 - df_heat["norm"]  # lower V -> hotter
                gradient = {0.0: "#1a9641", 0.33: "#fee08b", 0.66: "#fdae61", 1.0: "#d7191c"}
                HeatMap(
                    data=df_heat[["lat", "lon", "weight"]].values.tolist(),
                    name="Voltage Heat",
                    max_zoom=CONFIG["HEAT_MAX_ZOOM"],
                    radius=CONFIG["HEAT_RADIUS"],
                    blur=CONFIG["HEAT_BLUR"],
                    gradient=gradient,
                ).add_to(m_heat)

            add_substation_markers(m_heat, buses_join)
            add_symbol_legend_html(m_heat)
            add_continuous_colorbar(m_heat, lo, hi, palette="heat", label="Voltage (p.u.)")
            folium.LayerControl(collapsed=False).add_to(m_heat)
            m_heat.save(str(feeder_out / f"{feeder}_heat.html"))

            # BUS HTML
            m_bus = make_map(center, zoom)
            add_lines_layer_to_map(m_bus, gdf_lines, color, feeder)
            add_bus_markers_colored_by_voltage(m_bus, buses_join[buses_join["Vpu_min"].notna()].copy(), lo, hi)
            add_substation_markers(m_bus, buses_join)
            add_symbol_legend_html(m_bus)
            add_continuous_colorbar(m_bus, lo, hi, palette="bus", label="Bus Voltage (p.u.)")
            folium.LayerControl(collapsed=False).add_to(m_bus)
            m_bus.save(str(feeder_out / f"{feeder}_bus.html"))

            # FEEDER PNG
            plot_voltage_png(
                title=f"{feeder} — Bus Voltages",
                gdf_lines_wgs84=gdf_lines,
                buses_df_wgs84=buses_join,
                subs_df_wgs84=subs_df,
                out_path=feeder_out / f"{feeder}_voltage.png",
                vmin=lo,
                vmax=hi,
                show_bus_voltages=True,
            )

            status = "ok" if feeder_ok else "solve failed (maps produced if partial voltages exist)"
        else:
            status = "no voltage data"

        summary_rows.append({"feeder": feeder, "status": status, "min": v_min, "max": v_max, "mean": v_mean})

        # combined accumulators
        combined_lines.append(gdf_lines.assign(feeder=feeder))
        combined_bus_attrs.append(buses_join.assign(feeder=feeder))
        if not subs_df.empty:
            combined_subs.append(subs_df.assign(feeder=feeder))
        folium_layers.append((feeder, gdf_lines, color))

    # ---------------------- COMBINED OUTPUTS ----------------------
    if not combined_lines:
        raise SystemExit("No valid feeders were processed — nothing to combine.")

    all_lines = pd.concat(combined_lines, ignore_index=True)
    all_buses = pd.concat(combined_bus_attrs, ignore_index=True)
    all_subs = pd.concat(combined_subs, ignore_index=True) if combined_subs else pd.DataFrame(columns=all_buses.columns)

    # combined shapefiles
    sanitize = sanitize_for_shapefile(all_lines.drop(columns=["geometry"]).copy())
    gpd.GeoDataFrame(
        sanitize.join(all_lines[["geometry"]]),
        geometry="geometry",
        crs=CONFIG["TARGET_CRS"],
    ).to_file(out_root / "combined_lines.shp")

    gdf_all_buses = gpd.GeoDataFrame(
        all_buses.copy(),
        geometry=gpd.points_from_xy(all_buses["lon"], all_buses["lat"]),
        crs=CONFIG["TARGET_CRS"],
    )
    sanitize_b = sanitize_for_shapefile(gdf_all_buses.drop(columns=["geometry"]).copy())
    gpd.GeoDataFrame(
        sanitize_b.join(gdf_all_buses[["geometry"]]),
        geometry="geometry",
        crs=CONFIG["TARGET_CRS"],
    ).to_file(out_root / "combined_buses.shp")

    if not all_subs.empty:
        gdf_all_subs = gpd.GeoDataFrame(
            all_subs.copy(),
            geometry=gpd.points_from_xy(all_subs["lon"], all_subs["lat"]),
            crs=CONFIG["TARGET_CRS"],
        )
        sanitize_s = sanitize_for_shapefile(gdf_all_subs.drop(columns=["geometry"]).copy())
        gpd.GeoDataFrame(
            sanitize_s.join(gdf_all_subs[["geometry"]]),
            geometry="geometry",
            crs=CONFIG["TARGET_CRS"],
        ).to_file(out_root / "combined_substations.shp")

    # combined HTML: all feeders
    center = CONFIG["MAP_CENTER"] or (float(all_buses["lat"].mean()), float(all_buses["lon"].mean()))
    zoom = CONFIG["MAP_ZOOM"] or 11

    m_comb = make_map(center, zoom)
    for feeder, gdf_lines, color in folium_layers:
        add_lines_layer_to_map(m_comb, gdf_lines, color, feeder)
    add_substation_markers(m_comb, all_buses)
    add_symbol_legend_html(m_comb)
    folium.LayerControl(collapsed=False).add_to(m_comb)
    m_comb.save(str(out_root / "combined_feeders.html"))

    # combined heat HTML (fixed bounds for comparability)
    m_comb_heat = make_map(center, zoom)
    for feeder, gdf_lines, color in folium_layers:
        add_lines_layer_to_map(m_comb_heat, gdf_lines, color, feeder)

    lo, hi = CONFIG["V_MIN"], CONFIG["V_MAX"]
    df_heat = all_buses[["lat", "lon", "Vpu_min"]].dropna().copy()
    if not df_heat.empty and hi > lo:
        df_heat["norm"] = df_heat["Vpu_min"].apply(lambda v: clamp((float(v) - lo) / (hi - lo), 0, 1))
        df_heat["weight"] = 1.0 - df_heat["norm"]
        gradient = {0.0: "#1a9641", 0.33: "#fee08b", 0.66: "#fdae61", 1.0: "#d7191c"}
        HeatMap(
            data=df_heat[["lat", "lon", "weight"]].values.tolist(),
            name="Voltage Heat",
            max_zoom=CONFIG["HEAT_MAX_ZOOM"],
            radius=CONFIG["HEAT_RADIUS"],
            blur=CONFIG["HEAT_BLUR"],
            gradient=gradient,
        ).add_to(m_comb_heat)

    add_substation_markers(m_comb_heat, all_buses)
    add_symbol_legend_html(m_comb_heat)
    add_continuous_colorbar(m_comb_heat, lo, hi, palette="heat", label="Voltage (p.u.)")
    folium.LayerControl(collapsed=False).add_to(m_comb_heat)
    m_comb_heat.save(str(out_root / "combined_heat.html"))

    # combined PNGs
    gdf_all_lines = gpd.GeoDataFrame(all_lines, geometry="geometry", crs=CONFIG["TARGET_CRS"])

    # City “topology” PNG (cleanest for exhibits)
    plot_voltage_png(
        title="Phoenix Suburb (Gilbert AZ) Distribution Topology (Synthetic Feeders)",
        gdf_lines_wgs84=gdf_all_lines,
        buses_df_wgs84=all_buses,
        subs_df_wgs84=all_subs if not all_subs.empty else all_buses[all_buses["bus"].apply(is_substation_name)].copy(),
        out_path=out_root / "city_topology.png",
        vmin=None,
        vmax=None,
        show_bus_voltages=False,
    )

    # City voltage PNG
    if CONFIG["CITY_PNG_FIXED_BOUNDS"]:
        vmin_city, vmax_city = CONFIG["V_MIN"], CONFIG["V_MAX"]
    else:
        vv = all_buses["Vpu_min"].dropna().astype(float)
        vmin_city, vmax_city = (float(vv.min()), float(vv.max())) if not vv.empty else (CONFIG["V_MIN"], CONFIG["V_MAX"])
        if vmax_city <= vmin_city:
            vmax_city = vmin_city + 1e-3

    plot_voltage_png(
        title="City-wide Bus Voltages (Synthetic Feeders)",
        gdf_lines_wgs84=gdf_all_lines,
        buses_df_wgs84=all_buses,
        subs_df_wgs84=all_subs if not all_subs.empty else all_buses[all_buses["bus"].apply(is_substation_name)].copy(),
        out_path=out_root / "city_voltage.png",
        vmin=vmin_city,
        vmax=vmax_city,
        show_bus_voltages=True,
    )

    # summary
    print("\n===== SUMMARY =====")
    for row in summary_rows:
        print(f"{row['feeder']}: {row['status']} | min={row['min']}, max={row['max']}, mean={row['mean']}")
    print(f"\nDone. Outputs are in: {out_root}")


if __name__ == "__main__":
    build_everything()
