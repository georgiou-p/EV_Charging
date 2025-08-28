# addResiliencyTargetedWeighted.py
import json
import random
import math
import networkx as nx

from station_assignment import assign_charging_stations_to_nodes


def add_weighted_targeted_failures(
    input_file: str,
    output_file: str,
    geojson_path: str,
    failure_rate: float = 0.10,
    alpha: float = 1.6,         # centrality emphasis (the higher the sharper targeting)
    decay_factor: float = 0.6,  # soft anti-clustering across a station (1.0 = none)
    rng_seed: int = 42
):
    """
    Fail ~failure_rate of charging *points* by flipping whole connections,
    sampled stochastically with betweenness-based weights and per-station decay
    to reduce 'all-dead' stations.

    Parameters
    ----------
    input_file : cleaned station JSON (Stations -> Connections[] with Quantity)
    output_file: path to write updated JSON (sets/keeps 'Working' flags)
    geojson_path: UK regions GeoJSON (used by station_assignment to build graph)
    failure_rate: target share of total points to fail (e.g., 0.10 = 10%)
    alpha: exponent for centrality weighting (1.0 ~ proportional; 1.6 = sharper)
    decay_factor: per-station downweight for each *failed point* (0.6 strong; 0.8 mild; 1.0 off)
    rng_seed: RNG seed
    """

    random.seed(rng_seed)

    print("=" * 88)
    print("BETWEENNESS-WEIGHTED FAILURES with SOFT ANTI-CLUSTERING (connection-level)")
    print("=" * 88)

    # 1) Load JSON
    with open(input_file, "r", encoding="utf-8") as f:
        stations_data = json.load(f)

    # Ensure Working defaults to True
    for st in stations_data:
        for conn in st.get("Connections", []):
            conn["Working"] = True
            if "Quantity" not in conn or conn["Quantity"] is None:
                conn["Quantity"] = 1

    # 2) Graph + betweenness + station→node map
    graph, node_stations = assign_charging_stations_to_nodes(geojson_path, input_file)
    betw = nx.betweenness_centrality(graph, weight="weight")

    station_to_node = {}
    for n in graph.nodes:
        for st_obj in graph.nodes[n]["charging_stations"]:
            station_to_node[st_obj.get_station_id()] = n

    # 3) Flatten connections + count total points
    connections = []  # each entry: {station_id, node, centrality, quantity, conn_ref, w_base}
    total_points = 0

    for st in stations_data:
        sid = st.get("StationID")
        node = station_to_node.get(sid, None)
        cval = betw.get(node, 0.0) if node is not None else 0.0

        for conn in st.get("Connections", []):
            q = int(conn.get("Quantity", 1) or 1)
            total_points += q
            connections.append({
                "station_id": sid,
                "node": node,
                "centrality": cval,
                "quantity": q,
                "conn_ref": conn
            })

    target_fail = int(round(total_points * failure_rate))
    print(f"Total charging points: {total_points}")
    print(f"Target to fail:        {target_fail} points  ({failure_rate*100:.1f}%)")

    if not connections or target_fail <= 0:
        print("Nothing to do.")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stations_data, f, indent=4, ensure_ascii=False)
        return {
            "total_points": total_points, "failed_points": 0,
            "failure_rate": 0.0, "stations_affected": 0
        }

    # 4) Normalize centrality & build base weights
    cvals = [c["centrality"] for c in connections]
    cmin, cmax = min(cvals), max(cvals)
    span = (cmax - cmin) or 1.0

    for c in connections:
        cnorm = (c["centrality"] - cmin) / span
        c["w_base"] = (cnorm + 1e-9) ** alpha  # epsilon avoids zeroing

    # 5) Soft anti-clustering memory: per-station points already failed
    station_failed_points = {c["station_id"]: 0 for c in connections}

    def pick_weight(conn):
        # decay per *point* already failed at that station
        already = station_failed_points.get(conn["station_id"], 0)
        return conn["w_base"] * (decay_factor ** already)

    # 6) Iterative sampling until budget is met
    available = list(connections)  # connections not yet failed
    failed_points = 0
    draws = 0

    while available and failed_points < target_fail:
        weights = [max(pick_weight(c), 0.0) for c in available]
        total_w = sum(weights)

        # choose an index proportional to weight (fallback to uniform if degenerate)
        if total_w <= 0:
            idx = random.randrange(len(available))
        else:
            r = random.random() * total_w
            acc = 0.0
            idx = 0
            for i, w in enumerate(weights):
                acc += w
                if r <= acc:
                    idx = i
                    break

        chosen = available.pop(idx)

        # flip this connection OFF 
        chosen["conn_ref"]["Working"] = False
        q = chosen["quantity"]
        failed_points += q
        station_failed_points[chosen["station_id"]] += q
        draws += 1

        if draws % 100 == 0 or failed_points >= target_fail:
            print(f"  progress: failed={failed_points}/{target_fail} "
                  f"({failed_points/total_points*100:.2f}%) after {draws} picks")

    # 7) Save + summary
    actual_failed = 0
    stations_with_fail = 0
    for st in stations_data:
        station_failed = 0
        for conn in st.get("Connections", []):
            if conn.get("Working") is False:
                station_failed += int(conn.get("Quantity", 1) or 1)
        if station_failed > 0:
            stations_with_fail += 1
            actual_failed += station_failed

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stations_data, f, indent=4, ensure_ascii=False)

    print("\nSUMMARY")
    print("-------")
    print(f"Stations affected: {stations_with_fail} / {len(stations_data)}")
    print(f"Failed points:     {actual_failed} / {total_points} "
          f"({actual_failed/total_points*100:.2f}%)")
    print(f"Draws (connections flipped): {draws}")
    print(f"alpha={alpha}  decay_factor={decay_factor}")

    return {
        "total_points": total_points,
        "failed_points": actual_failed,
        "failure_rate": actual_failed / total_points if total_points else 0.0,
        "stations_affected": stations_with_fail,
        "alpha": alpha,
        "decay_factor": decay_factor
    }


def visualize_failures_map(output_file, geojson_path, save_path="TargetedWeighted_Map.png"):
    """
    Plot stations colored by share of failed points (uses your same basemap idea).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd

    with open(output_file, "r", encoding="utf-8") as f:
        stations = json.load(f)

    lats, lons, tot, fail = [], [], [], []
    for s in stations:
        lat, lon = s.get("Latitude"), s.get("Longitude")
        if not lat or not lon:
            continue
        tp = 0; fp = 0
        for c in s.get("Connections", []):
            q = int(c.get("Quantity", 1) or 1)
            tp += q
            if c.get("Working") is False:
                fp += q
        if tp == 0:
            continue
        lats.append(lat); lons.append(lon); tot.append(tp); fail.append(fp)

    if not lats:
        print("Nothing to plot.")
        return

    tot = np.array(tot, float); fail = np.array(fail, float)
    share = np.divide(fail, tot, out=np.zeros_like(fail), where=tot > 0)

    fig, ax = plt.subplots(figsize=(12, 14), dpi=150)
    try:
        uk = gpd.read_file(geojson_path)
        uk.plot(ax=ax, linewidth=0.4, edgecolor="white", facecolor="lightblue", alpha=0.6)
    except Exception as e:
        print(f"Basemap failed: {e}")

    sizes = 10 + 90 * (tot / max(1, np.percentile(tot, 95)))
    sizes = np.clip(sizes, 10, 120)

    sc = ax.scatter(lons, lats, c=share, s=sizes, cmap="Reds", vmin=0, vmax=1,
                    edgecolors="k", linewidths=0.3, alpha=0.85)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Share of failed charging points per station", rotation=270, labelpad=18)

    ax.set_title("EV Charging: Betweenness‑weighted failures (softly spread across sites)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.2)

    try:
        ax.set_xlim(-8.8, 2.6); ax.set_ylim(49.5, 61.2)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved map to {save_path}")
    plt.show()


def main():
    input_file = "./data/cleaned_charging_stations.json"
    output_file = "TargetedWeightedFailures.json"
    geojson_path = "data/UK_Mainland_GB_simplified.geojson"

    stats = add_weighted_targeted_failures(
        input_file, output_file, geojson_path,
        failure_rate=0.10,  # global target
        alpha=1.6,          # how hard to target hubs
        decay_factor=0.6,   # how much to avoid wiping a site
        rng_seed=42
    )

    print(f"\nDone. Achieved failure rate: {stats['failure_rate']*100:.2f}% (target 10%).")

    visualize_failures_map(output_file, geojson_path, save_path="TargetedWeighted_Map.png")


if __name__ == "__main__":
    main()
