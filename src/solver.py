#!/usr/bin/env python3
"""
3D Heat Transfer FEM Solver - Main Solver Module (Implicit Backward Euler)

Implicit Backward Euler:
    (M + dt * kappa * K) T^{n+1} = M T^n

Solved with Preconditioned Conjugate Gradient (PCG) using a Jacobi (diagonal) preconditioner.
Matrix-free style is preserved via matvec operators for M and K (no global assembly).

Notes:
- For pure heat conduction, the system matrix is symmetric positive definite (SPD), so CG is appropriate.
- Dirichlet BCs are enforced via a lifting: solve for u where T = T_dir + u, u[dir]=0.
- JAX on GPU is asynchronous; for fair wall-clock timing, we periodically synchronize.
- For benchmarking, set benchmark_mode=True to avoid per-step host sync (min/max/history/logging).
"""

import sys
import time
import csv
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from .fem_utils import (
    generate_mesh,
    compute_element_matrices,
    apply_K_global,
    apply_M_global,
    assemble_diagonal,
    assert_affine_reuse_ok,
)

jax.config.update("jax_enable_x64", False)


# ============================================================
# Helpers
# ============================================================
def _pick_device(prefer_gpu=True):
    """Pick GPU if available, otherwise CPU. Never crash if GPU is absent."""
    if prefer_gpu:
        try:
            g = jax.devices("gpu")
            if len(g) > 0:
                return g[0]
        except RuntimeError:
            pass
    return jax.devices("cpu")[0]


@jit
def _project_free(v, dir_nodes):
    """Set Dirichlet DOFs to zero (unknown lives on free DOFs)."""
    return v.at[dir_nodes].set(0.0)


# ============================================================
# Linear Solver: Preconditioned Conjugate Gradient (Jacobi)
# ============================================================
def cg_solve(matvec, b, x0, M_inv, rtol=1e-6, atol=0.0, maxiter=200, dir_nodes=None):
    """
    Preconditioned CG for SPD system: A x = b

    Stopping criterion (PETSc-style):
        ||r|| <= atol + rtol * ||b||

    Returns:
        x, iters, final_res_norm
    """
    if dir_nodes is not None:
        b = _project_free(b, dir_nodes)
        x = _project_free(x0, dir_nodes)
    else:
        x = x0

    r = b - matvec(x)
    if dir_nodes is not None:
        r = _project_free(r, dir_nodes)

    z = M_inv * r
    p = z
    rz_old = jnp.dot(r, z)

    res_norm0 = jnp.linalg.norm(r)
    b_norm = jnp.linalg.norm(b)
    tol = atol + rtol * jnp.maximum(b_norm, res_norm0) + 1e-30
    res_norm = res_norm0
    if res_norm <= tol:
        return x, 0, res_norm

    for it in range(maxiter):
        Ap = matvec(p)
        alpha = rz_old / (jnp.dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap

        if dir_nodes is not None:
            x = _project_free(x, dir_nodes)
            r = _project_free(r, dir_nodes)

        res_norm = jnp.linalg.norm(r)
        if res_norm <= tol:
            return x, it + 1, res_norm

        z = M_inv * r
        rz_new = jnp.dot(r, z)
        beta = rz_new / (rz_old + 1e-30)
        p = z + beta * p
        rz_old = rz_new

    return x, maxiter, res_norm


# ============================================================
# Implicit Backward Euler Step (matrix-free PCG)
# ============================================================
def backward_euler_step(
    Tn,
    elem_dofs,
    Ke,
    Me,
    dt,
    kappa,
    dir_nodes,
    T_bc_vals,
    M_inv,
    u0,
    cg_rtol=1e-6,
    cg_atol=0.0,
    cg_maxiter=200,
):
    """
    Solve one implicit Backward Euler step:
        (M + dt*kappa*K) T_{n+1} = M T_n

    Dirichlet enforced by lifting:
        T_{n+1} = T_dir + u, with u[dir]=0
    """
    T_dir = jnp.zeros_like(Tn).at[dir_nodes].set(T_bc_vals)

    def A_mv(v):
        return apply_M_global(v, elem_dofs, Me) + (dt * kappa) * apply_K_global(v, elem_dofs, Ke)

    rhs = apply_M_global(Tn, elem_dofs, Me)
    rhs_eff = rhs - A_mv(T_dir)

    u, iters, res = cg_solve(
        A_mv,
        rhs_eff,
        u0,
        M_inv,
        rtol=cg_rtol,
        atol=cg_atol,
        maxiter=cg_maxiter,
        dir_nodes=dir_nodes,
    )
    Tnp1 = u + T_dir
    return Tnp1, u, iters, res


# ============================================================
# Main Simulation Function
# ============================================================
def run_simulation(
    nx=20,
    ny=20,
    nz=20,
    dt=None,
    steps=500,
    T_bottom=100.0,
    T_top=0.0,
    kappa=1.0,
    Lx=1.0,
    Ly=1.0,
    Lz=1.0,
    save_history=False,
    log_file=None,
    verbose=True,
    cg_rtol=None,
    cg_atol=None,
    cg_maxiter=200,
    mode="run",
    prefer_gpu=True,
    benchmark_mode=False,
    sync_every=50,
    implicit_dt_scale=1.0,
):
    """
    PDE (semi-discrete):
        M dT/dt + kappa K T = 0

    Backward Euler:
        (M + dt*kappa*K) T^{n+1} = M T^n

    benchmark_mode:
      - If True: disables per-step min/max history/logging/saving to avoid host sync overhead.
    sync_every:
      - Synchronize (block_until_ready) every N steps for accurate wall-clock timing on GPU.
    implicit_dt_scale:
      - When dt is None, multiply the auto (explicit-CFL-style) dt by this factor. Implicit BE
        often allows values > 1; time accuracy remains O(dt). For similar physical end time with
        larger dt, reduce steps proportionally (e.g. scale=4 and steps/4).
    """
    t_total_start = time.time()

    mode = str(mode).lower()
    if mode not in ("run", "verify"):
        raise ValueError("mode must be either 'run' or 'verify'")
    verify_mode = mode == "verify"

    # Keep solver in single precision for best GPU throughput.
    dtype = jnp.float32

    if cg_rtol is None:
        cg_rtol = 1e-10 if verify_mode else 1e-6
    if cg_atol is None:
        cg_atol = 1e-12 if verify_mode else 0.0

    # If benchmarking, force off anything that syncs to host frequently
    if benchmark_mode:
        save_history = False
        log_file = None

    history = {
        "T_min": [],
        "T_max": [],
        "timing": {},
        "stats": {
            "cg_iters_per_step": [],
            "cg_iters_total": 0,
            "cg_res_per_step": [],
        },
    }
    T_history = [] if save_history else None

    device = _pick_device(prefer_gpu=prefer_gpu)
    if verbose:
        print(f"Device: {device}")

    with jax.default_device(device):
        # ----------------- Assembly -----------------
        t_asm_start = time.time()

        coords_global, elem_dofs, Nx, Ny, Nz, Ne, Nnodes = generate_mesh(nx, ny, nz, Lx, Ly, Lz, dtype=dtype)

        hx, hy, hz = Lx / nx, Ly / ny, Lz / nz
        coords0 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [hx, 0.0, 0.0],
                [hx, hy, 0.0],
                [0.0, hy, 0.0],
                [0.0, 0.0, hz],
                [hx, 0.0, hz],
                [hx, hy, hz],
                [0.0, hy, hz],
            ],
            dtype=dtype,
        )

        Ke, Me = compute_element_matrices(coords0, check_orientation=verify_mode)
        if verify_mode:
            assert_affine_reuse_ok(coords_global, elem_dofs, tol=1e-10)

        diagM = assemble_diagonal(Me, elem_dofs, Nnodes)
        diagK = assemble_diagonal(Ke, elem_dofs, Nnodes)
        diagA = diagM + (dt * kappa) * diagK if dt is not None else None

        if dt is None:
            h_min = float(min(hx, hy, hz))
            dt_stable = (h_min**2) / (6.0 * float(kappa))
            dt = 0.5 * dt_stable
            scale = float(implicit_dt_scale)
            if scale <= 0:
                raise ValueError("implicit_dt_scale must be positive")
            if scale != 1.0:
                dt = dt * scale
            if verbose:
                print(
                    f"Default dt (explicit-CFL baseline × implicit_dt_scale={scale:g}): "
                    f"dt={dt:.6e} — implicit BE allows larger dt; smaller time error needs smaller dt or more steps."
                )
        if diagA is None:
            diagA = diagM + (dt * kappa) * diagK

        # Initial condition: hot sphere at center
        T = jnp.full(Nnodes, T_top, dtype=dtype)
        cx, cy, cz = Lx / 2.0, Ly / 2.0, Lz / 2.0
        R = min(Lx, Ly, Lz) / 6.0
        X, Y, Z = coords_global[:, 0], coords_global[:, 1], coords_global[:, 2]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        T = jnp.where(dist_sq < (R**2), T_bottom, T)

        # Sphere Dirichlet boundary condition.
        # All nodes inside radius R around (xc, yc, zc) are fixed to T_bottom.
        xc, yc, zc = Lx / 2.0, Ly / 2.0, Lz / 2.0
        sphere_bc_radius = min(Lx, Ly, Lz) / 5.0
        sphere_dist = jnp.sqrt((X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2)
        sphere_mask = sphere_dist <= sphere_bc_radius
        T = jnp.where(sphere_mask, T_bottom, T)

        dir_nodes = jnp.where(sphere_mask)[0]
        T_bc_vals = jnp.full((dir_nodes.shape[0],), T_bottom, dtype=dtype)
        diagA = diagA.at[dir_nodes].set(1.0)
        M_inv = 1.0 / diagA
        T_dir = jnp.zeros_like(T).at[dir_nodes].set(T_bc_vals)
        u_guess = _project_free(T - T_dir, dir_nodes)

        # Ensure assembly ops are complete before timing is read
        jax.block_until_ready(T)

        assembly_s = time.time() - t_asm_start
        if verbose:
            print(f"\nMesh: {nx}×{ny}×{nz} elements, {Nnodes} nodes")
            print(f"Assembly time: {assembly_s:.3f} s")

        # ----------------- Warm-up (JIT) - not timed -----------------
        # Compile kernels without advancing physical state/time.
        _, u_guess_warm, _, _ = backward_euler_step(
            T,
            elem_dofs,
            Ke,
            Me,
            dt,
            kappa,
            dir_nodes,
            T_bc_vals,
            M_inv,
            u_guess,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
        )
        jax.block_until_ready(u_guess_warm)

        # ----------------- Solve (timed) -----------------
        t_solve_start = time.time()

        # Optional logging setup (disabled in benchmark_mode)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "time", "T_min", "T_max", "cg_iters", "cg_res"])

        report_every = max(1, steps // 10)

        for step in range(steps):
            T, u_guess, cg_iters, cg_res = backward_euler_step(
                T,
                elem_dofs,
                Ke,
                Me,
                dt,
                kappa,
                dir_nodes,
                T_bc_vals,
                M_inv,
                u_guess,
                cg_rtol=cg_rtol,
                cg_atol=cg_atol,
                cg_maxiter=cg_maxiter,
            )

            # Collect CG stats (cheap; cg_iters/cg_res are already scalars)
            history["stats"]["cg_iters_total"] += int(cg_iters)
            if not benchmark_mode:
                history["stats"]["cg_iters_per_step"].append(int(cg_iters))
                history["stats"]["cg_res_per_step"].append(float(cg_res))

            # Periodic sync for accurate wall-clock timing on GPU
            if sync_every and ((step + 1) % sync_every == 0):
                jax.block_until_ready(T)

            # Avoid per-step host sync unless needed
            do_report = verbose and ((step + 1) % report_every == 0)
            do_log = log_file is not None
            do_save_hist = save_history

            if (do_report or do_log) and (not benchmark_mode):
                Tmin = float(T.min())
                Tmax = float(T.max())
                history["T_min"].append(Tmin)
                history["T_max"].append(Tmax)

                if do_log:
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([step, step * dt, Tmin, Tmax, int(cg_iters), float(cg_res)])

                if do_report:
                    print(f"  Step {step+1}/{steps}: T_min={Tmin:.2f}, T_max={Tmax:.2f}, CG iters={int(cg_iters)}")

            if do_save_hist and (not benchmark_mode):
                T_history.append(T.copy())

        # Final sync before stopping the timer
        jax.block_until_ready(T)

        solve_s = time.time() - t_solve_start
        total_s = time.time() - t_total_start

        history["timing"] = {
            "assembly_s": assembly_s,
            "solve_s": solve_s,
            "total_s": total_s,
            "mesh_size": (nx, ny, nz),
            "num_elements": Ne,
            "num_nodes": Nnodes,
            "dt": float(dt),
            "steps": int(steps),
            "sec_per_step": float(solve_s / max(steps, 1)),
            "cg_rtol": float(cg_rtol),
            "cg_atol": float(cg_atol),
            "cg_maxiter": int(cg_maxiter),
            "mode": mode,
            "dtype": "float64" if verify_mode else "float32",
            "benchmark_mode": bool(benchmark_mode),
            "sync_every": int(sync_every) if sync_every else 0,
            "implicit_dt_scale": float(implicit_dt_scale),
        }

        # Final temperature stats (do once; acceptable sync)
        Tmin_final = float(T.min())
        Tmax_final = float(T.max())

        if verbose:
            print("\n" + "="*60)
            print("[FEM IMPLICIT (Backward Euler + PCG-Jacobi, Matrix-Free)]")
            print(f"Mesh     : {nx} x {ny} x {nz}")
            print(f"Elements : {Ne:6d}")
            print(f"Nodes    : {Nnodes:6d}")
            print(f"dt       : {float(dt):.6e}")
            print(f"Assembly : {assembly_s:8.3f} s")
            print(f"Solve    : {solve_s:8.3f} s")
            print(f"Total    : {total_s:8.3f} s")
            print("--------------------------------")
            print(f"CG iters total: {history['stats']['cg_iters_total']}")
            print(f"sec/step : {solve_s/max(steps,1):.6f} s")
            print(f"avg CG/step: {history['stats']['cg_iters_total']/max(steps,1):.2f}")
            print(f"Tmin = {Tmin_final:.4f}, Tmax = {Tmax_final:.4f}")
            print("="*60 + "\n")

        T_final = T.reshape(Nx, Ny, Nz)

        if save_history and (not benchmark_mode):
            history["T_history"] = [Ti.reshape(Nx, Ny, Nz) for Ti in T_history]

        np.save("temperature.npy", np.array(T_final))
        if verbose:
            print("Saved temperature.npy")

        return T_final, history


def run_fem_explicit(nx=20, ny=20, nz=20, dt=None, steps=500, T_bottom=100.0, T_top=0.0, kappa=1.0):
    """Backward compatibility wrapper (now runs implicit BE)."""
    T, _ = run_simulation(
        nx=nx,
        ny=ny,
        nz=nz,
        dt=dt,
        steps=steps,
        T_bottom=T_bottom,
        T_top=T_top,
        kappa=kappa,
        save_history=False,
        verbose=True,
    )
    return T


def main():
    argv = [a for a in sys.argv[1:] if a != "--fast"]
    fast = "--fast" in sys.argv[1:]

    nx = ny = nz = 20
    if len(argv) >= 3:
        nx = int(argv[0])
        ny = int(argv[1])
        nz = int(argv[2])
        print(f"\nUsing mesh size from CLI: {nx} {ny} {nz}")
    else:
        print("\nUsing default mesh size: 20 20 20")
        print("Usage: python -m src.solver [nx] [ny] [nz] [--fast]")
        print("  --fast  : larger implicit dt + fewer steps (~same physical time as default 500 steps)")

    if fast:
        print("\n[--fast] implicit_dt_scale=4, steps=125 (approx. same physical time as default run)")

    common_kwargs = dict(
        nx=nx,
        ny=ny,
        nz=nz,
        dt=None,
        steps=125 if fast else 500,
        T_bottom=100.0,
        T_top=0.0,
        kappa=1.0,
        verbose=True,
        benchmark_mode=True,   # <-- set True for performance runs
        sync_every=50,         # <-- sync every 50 steps for accurate wall-clock
        cg_rtol=1e-6,
        cg_atol=0.0,
        cg_maxiter=200,
        implicit_dt_scale=4.0 if fast else 1.0,
    )

    print("\nExecution 1 (Includes JIT Compilation)")
    t_exec1_start = time.time()
    run_simulation(**common_kwargs)
    exec1_s = time.time() - t_exec1_start
    print(f"Execution 1 wall time: {exec1_s:.3f} s")

    print("\nExecution 2 (Pure GPU Execution)")
    t_exec2_start = time.time()
    run_simulation(**common_kwargs)
    exec2_s = time.time() - t_exec2_start
    print(f"Execution 2 wall time: {exec2_s:.3f} s")


if __name__ == "__main__":
    main()
    # Execution 2 is the fair comparison against a pre-compiled CPU solver
    # because it excludes one-time XLA compilation overhead.