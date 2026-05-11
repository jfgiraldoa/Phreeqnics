"""
Script Name: Phreeqc + Fenics Reactive Transport V3 - Adaptivity
Author: Juan Giraldo
Email: juan.giraldo@csiro.au

"""
import numpy as np
import h5py
import os

def save_exchange_outputs_h5(filename_pore, filename_ts, accum, p, ts=40, to_mmol=True, attrs=None):
    import numpy as np
    import h5py

    if len(accum) == 0:
        raise ValueError("accum is empty")

    if "Nx" not in p:
        raise KeyError("p must contain 'Nx'")
    if "Lsupx" not in p:
        raise KeyError("p must contain 'Lsupx'")

    Nx = p.get("Nx_save", p["Nx"])
    L = p["Lsupx"]
    species_all = ['Ca', 'Cl', 'Na', 'K', 'N']

    snap0 = accum[0]
    if not isinstance(snap0, dict) or len(snap0) == 0:
        raise ValueError("accum[0] must be a non-empty dict")

    species = [k for k in species_all if k in snap0]
    if len(species) == 0:
        raise ValueError("No species were found in accum[0]")

    full_profile_mode = "dist" in snap0

    data_pv = {}

    if full_profile_mode:
        dist0 = np.asarray(snap0["dist"][0], dtype=float)

        prof0 = np.asarray(snap0[species[0]], dtype=float)
        if prof0.ndim != 2:
            raise ValueError(
                f"Dataset {species[0]} in accum[0] has shape {prof0.shape}, expected 2D in full-profile mode"
            )

        nprof = prof0.shape[1]

        if len(dist0) == nprof + 1:
            dist0 = dist0[:-1]

        if len(dist0) != nprof:
            raise ValueError(
                f"Length mismatch in full-profile mode: len(dist0)={len(dist0)} != nprof={nprof}"
            )

        order0 = np.argsort(dist0)

        for k in species:
            values = np.asarray(snap0[k], dtype=float)
            if values.ndim != 2:
                raise ValueError(
                    f"Dataset {k} in accum[0] has shape {values.shape}, expected 2D in full-profile mode"
                )

            values = values[:, order0]
            values = values[:, 0] 

            if to_mmol:
                values = 1000.0 * values
            data_pv[k] = values

    else:
        for k in species:
            values = np.asarray(snap0[k], dtype=float)
            if values.ndim != 1:
                raise ValueError(
                    f"Dataset {k} in accum[0] has shape {values.shape}, expected 1D history"
                )
            if to_mmol:
                values = 1000.0 * values
            data_pv[k] = values

    n_steps = len(data_pv[species[0]])
    for k in species:
        if len(data_pv[k]) != n_steps:
            raise ValueError(f"Dataset {k} has length {len(data_pv[k])}, expected {n_steps}")

    pore_vol = (np.arange(n_steps, dtype=float) + 0.5) / Nx

    os.makedirs(os.path.dirname(filename_pore), exist_ok=True)

    with h5py.File(filename_pore, "w") as h5:
        h5.attrs["x_name"] = "Pore_vol"
        h5.attrs["x_unit"] = "-"
        h5.attrs["y_unit"] = "mmol/kgw" if to_mmol else "mol/kgw"
        h5.attrs["output_type"] = "concentration_vs_pore_volume"
        h5.attrs["Nx"] = Nx
        h5.attrs["Lsupx"] = L
        h5.attrs["n_steps"] = n_steps
        h5.attrs["source"] = "accum[0]"

        if attrs:
            for k, v in attrs.items():
                h5.attrs[k] = v

        h5.create_dataset("x", data=pore_vol, compression="gzip", shuffle=True)

        grp = h5.create_group("fields")
        for k in species:
            grp.create_dataset(k, data=data_pv[k], compression="gzip", shuffle=True)


    if full_profile_mode:
        x_dist = np.asarray(accum[0]["dist"][ts], dtype=float)

        data_ts = {}
        for k in species:
            arr = np.asarray(accum[0][k][ts], dtype=float)
            if to_mmol:
                arr = 1000.0 * arr
            data_ts[k] = arr

        nprof = len(data_ts[species[0]])

        if len(x_dist) == nprof + 1:
            x_dist = x_dist[:-1]

        if len(x_dist) != nprof:
            raise ValueError(
                f"Length mismatch in distance output: len(x_dist)={len(x_dist)} != nprof={nprof}"
            )

        order = np.argsort(x_dist)
        x_dist = x_dist[order]
        data_ts = {k: v[order] for k, v in data_ts.items()}

        source_str = "accum[0][species][ts], accum[0]['dist'][ts]"

    else:
        valid_accum = [cell_data for cell_data in accum if cell_data is not None]
        if len(valid_accum) == 0:
            raise ValueError("No valid point-history entries found in accum")

        n_cells = len(valid_accum)
        x_dist = np.linspace(0.0, L, n_cells)

        data_ts = {k: [] for k in species}

        for cell_data in valid_accum:
            for k in species:
                arr = np.asarray(cell_data[k], dtype=float)
                value = arr[ts]
                if to_mmol:
                    value *= 1000.0
                data_ts[k].append(value)

        data_ts = {k: np.asarray(v, dtype=float)[::-1] for k, v in data_ts.items()}
        x_dist = x_dist[::-1]

        source_str = "accum[:]"

    os.makedirs(os.path.dirname(filename_ts), exist_ok=True)

    with h5py.File(filename_ts, "w") as h5:
        h5.attrs["x_name"] = "Distance"
        h5.attrs["x_unit"] = "m"
        h5.attrs["y_unit"] = "mmol/kgw" if to_mmol else "mol/kgw"
        h5.attrs["output_type"] = "concentration_vs_distance"
        h5.attrs["Nx"] = Nx
        h5.attrs["Lsupx"] = L
        h5.attrs["ts"] = ts
        h5.attrs["source"] = source_str

        if attrs:
            for k, v in attrs.items():
                h5.attrs[k] = v

        h5.create_dataset("x", data=x_dist, compression="gzip", shuffle=True)
        h5.create_dataset("t", data=np.asarray(ts))

        grp = h5.create_group("fields")
        for k in species:
            grp.create_dataset(k, data=data_ts[k], compression="gzip", shuffle=True)



def save_exchange_outputs_h5_adaptive(
    filename_timesolution,
    filename_snaps,
    transport_accum,
    p,
    ts=40,
    to_mmol=True,
    attrs=None
):

    if transport_accum == 0:
        raise ValueError("transport_accum is 0")

    if "timestep" not in transport_accum or "snaps" not in transport_accum:
        raise KeyError("transport_accum must contain 'timestep' and 'snaps'")

    timestep_data = transport_accum["timestep"]
    snaps_data = transport_accum["snaps"]

    if "Lsupx" not in p:
        raise KeyError("p must contain 'Lsupx'")

    L = p["Lsupx"]
    species_all = ['Ca', 'Cl', 'Na', 'K', 'N']

    with h5py.File(filename_timesolution, "w") as h5:
        h5.attrs["output_type"] = "adaptive_transport_timesolution"
        h5.attrs["x_name"] = "Distance"
        h5.attrs["x_unit"] = "m"
        h5.attrs["y_unit"] = "mmol/kgw" if to_mmol else "mol/kgw"
        h5.attrs["Lsupx"] = L

        if attrs:
            for k, v in attrs.items():
                h5.attrs[k] = v

        for ts_key in sorted(timestep_data.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
            snap = timestep_data[ts_key]
            grp_ts = h5.create_group(f"ts_{ts_key}")

            if "dist" not in snap:
                raise KeyError(f"Missing 'dist' in timestep_data[{ts_key}]")

            x = np.asarray(snap["dist"], dtype=float)
            order = np.argsort(x)
            x = x[order]

            grp_ts.create_dataset("x", data=x, compression="gzip", shuffle=True)

            fields = grp_ts.create_group("fields")
            species = [k for k in species_all if k in snap]

            for k in species:
                vals = np.asarray(snap[k], dtype=float)
                vals = vals[:len(x)]
                vals = vals[order]
                if to_mmol:
                    vals = 1000.0 * vals
                fields.create_dataset(k, data=vals, compression="gzip", shuffle=True)


    with h5py.File(filename_snaps, "w") as h5:
        h5.attrs["output_type"] = "adaptive_transport_snaps"
        h5.attrs["x_name"] = "Distance"
        h5.attrs["x_unit"] = "m"
        h5.attrs["y_unit"] = "mmol/kgw" if to_mmol else "mol/kgw"
        h5.attrs["Lsupx"] = L
        h5.attrs["requested_ts"] = ts
    
        if attrs:
            for k, v in attrs.items():
                h5.attrs[k] = v
    
        for ts_key in sorted(snaps_data.keys(), key=lambda x: int(x)):
            grp_ts = h5.create_group(f"ts_{ts_key}")
    
            for ref_level in sorted(snaps_data[ts_key].keys()):
                snap = snaps_data[ts_key][ref_level]
                grp_ref = grp_ts.create_group(f"ref_{ref_level}")

                if "dist" not in snap:
                    raise KeyError(f"Missing 'dist' in snaps[{ts_key}][{ref_level}]")
    
                x = np.asarray(snap["dist"], dtype=float)
                order = np.argsort(x)
                x = x[order]
    
                grp_ref.create_dataset(
                    "x",
                    data=x,
                    compression="gzip",
                    shuffle=True
                )
    
                fields = grp_ref.create_group("fields")
                species = [k for k in species_all if k in snap]
    
                for k in species:
                    vals = np.asarray(snap[k], dtype=float)
    
                    if len(vals) != len(order):
                        raise ValueError(
                            f"Field length mismatch for species '{k}' in "
                            f"snaps[{ts_key}][{ref_level}]: len(vals)={len(vals)}, len(x)={len(x)}"
                        )
    
                    vals = vals[order]
                    if to_mmol:
                        vals = 1000.0 * vals
    
                    fields.create_dataset(
                        k,
                        data=vals,
                        compression="gzip",
                        shuffle=True
                    )

                if "x_cells" in snap:
                    x_cells = np.asarray(snap["x_cells"], dtype=float)
                    x_cells = np.sort(x_cells)
    
                    grp_ref.create_dataset(
                        "x_cells",
                        data=x_cells,
                        compression="gzip",
                        shuffle=True
                    )

                if "marked_x" in snap:
                    marked_x = np.asarray(snap["marked_x"], dtype=float)
                    marked_x = np.sort(marked_x)
    
                    grp_ref.create_dataset(
                        "marked_x",
                        data=marked_x,
                        compression="gzip",
                        shuffle=True
                    )

                if "g_list" in snap:
                    ggrp = grp_ref.create_group("errors")
    
                    g_list = snap["g_list"]
                    n_err = None
                    if len(g_list) > 0:
                        n_err = len(np.asarray(g_list[0], dtype=float))
    
                    for i, g in enumerate(g_list):
                        g = np.asarray(g, dtype=float)
    
                        if n_err is not None and len(g) != n_err:
                            raise ValueError(
                                f"Inconsistent error length in snaps[{ts_key}][{ref_level}] "
                                f"for g_{i}: expected {n_err}, got {len(g)}"
                            )
    
                        if "x_cells" in snap and len(g) != len(np.asarray(snap["x_cells"], dtype=float)):
                            raise ValueError(
                                f"Error length mismatch in snaps[{ts_key}][{ref_level}] "
                                f"for g_{i}: len(g)={len(g)} but len(x_cells)="
                                f"{len(np.asarray(snap['x_cells'], dtype=float))}"
                            )
    
                        ggrp.create_dataset(
                            f"g_{i}",
                            data=g,
                            compression="gzip",
                            shuffle=True
                        )
 