# reproduce_arbogast.py
# ============================================================================
# Reproduction of the 1D Two-Phase Flow Benchmark (Full Domain -2 to 2)
# With corrected Coordinate System for Plotting (Depth-based)
# ============================================================================

import firedrake
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. Mesh and Geometry (Physical Domain: Elevation z)
# ----------------------------------------------------------------------------
H_total = 2.0
n_cells = 400
mesh = IntervalMesh(n_cells, -H_total, H_total)
x = SpatialCoordinate(mesh)
z = x[0]

# ----------------------------------------------------------------------------
# 2. Function Spaces
# ----------------------------------------------------------------------------
V = FunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
Pc = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace([V, P, Pc])

sol = Function(W, name="Solution")
u, p_f, p_c_tilde = split(sol)
v, q_f, q_c = TestFunctions(W)

# ----------------------------------------------------------------------------
# 3. Physical Parameters
# ----------------------------------------------------------------------------
g_val = 1.0
rho_s_val = 2.0
rho_f_val = 1.0
eta_0_val = 1.0
xi_0_val = 1.0      
n_exp = 2.0         
m_exp = 1.0         

g = Constant(-g_val) 
eta = Constant(eta_0_val)
rho_s = Constant(rho_s_val)
rho_f = Constant(rho_f_val)

# ----------------------------------------------------------------------------
# 4. Porosity Profile
# ----------------------------------------------------------------------------
phi_max = 0.004
phi_expr = conditional(z > 0, 0.0, phi_max * (z / -H_total)**2)
epsilon = 1e-10
phi = phi_expr + epsilon

# ----------------------------------------------------------------------------
# 5. Constitutive Relations
# ----------------------------------------------------------------------------
K_D = phi**n_exp
xi = xi_0_val / (phi**m_exp)
d_phi = sqrt(K_D)

# ----------------------------------------------------------------------------
# 6. Variational Formulation
# ----------------------------------------------------------------------------
# Stokes
term_stress = 2 * eta * u.dx(0) * v.dx(0) * dx
term_pressure_f = - p_f * v.dx(0) * dx
term_pressure_c = - (d_phi * p_c_tilde) * v.dx(0) * dx
term_gravity = - (phi * rho_f + (1 - phi) * rho_s) * g * v * dx
F1 = term_stress + term_pressure_f + term_pressure_c + term_gravity

# Darcy
term_div_u = u.dx(0) * q_f * dx
term_darcy_flux = K_D * p_f.dx(0) * q_f.dx(0) * dx
term_darcy_gravity = - K_D * rho_f * g * q_f.dx(0) * dx
F2 = term_div_u + term_darcy_flux + term_darcy_gravity

# Compaction
term_compact_div = - (d_phi * u.dx(0)) * q_c * dx
term_compact_relax = - (d_phi**2 / xi) * p_c_tilde * q_c * dx
is_impermeable = conditional(phi < 2*epsilon, 1.0, 0.0)
penalty = 1.0e10
term_constraint = - penalty * is_impermeable * p_c_tilde * q_c * dx
F3 = term_compact_div + term_compact_relax + term_constraint

F = F1 + F2 + F3

# ----------------------------------------------------------------------------
# 7. Boundary Conditions
# ----------------------------------------------------------------------------
bc_u_bot = DirichletBC(W.sub(0), Constant(0.0), 1)
bc_u_top = DirichletBC(W.sub(0), Constant(0.0), 2)
bc_p_top = DirichletBC(W.sub(1), Constant(0.0), 2)
bcs = [bc_u_bot, bc_u_top, bc_p_top]

# ----------------------------------------------------------------------------
# 8. Solver
# ----------------------------------------------------------------------------
solver_parameters = {
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None
}

print("[INFO] Solving Full Domain Arbogast Benchmark...")
solve(F == 0, sol, bcs=bcs, solver_parameters=solver_parameters)
print("[SUCCESS] Solution converged.")

# ----------------------------------------------------------------------------
# 9. Post-Processing & Plotting (Correction applied here)
# ----------------------------------------------------------------------------
u_sol, p_sol, pc_tilde_sol = sol.split()
Pc_space = FunctionSpace(mesh, "DG", 1)

# Recover Variables
pc_real = Function(Pc_space, name="Compaction Pressure").interpolate(d_phi * pc_tilde_sol)
u_melt = Function(Pc_space, name="Melt Velocity")
grad_p = Function(Pc_space).interpolate(p_sol.dx(0))
v_melt_expr = u_sol - (K_D / phi) * (grad_p - rho_f * g)
u_melt.interpolate(v_melt_expr)
phi_func = Function(P).interpolate(phi_expr)

# --- Numpy Sampling Helper ---
def sample_function(func, z_pts):
    P1 = FunctionSpace(func.function_space().mesh(), "CG", 1)
    f_p1 = Function(P1).interpolate(func)
    coords = f_p1.function_space().mesh().coordinates.dat.data_ro
    vals = f_p1.dat.data_ro
    idx = np.argsort(coords)
    return np.interp(z_pts, coords[idx], vals[idx])

# Sample data
z_plot = np.linspace(-H_total, H_total, 500)
val_phi = sample_function(phi_func, z_plot)
val_u   = sample_function(u_sol, z_plot)
val_um  = sample_function(u_melt, z_plot)
val_pf  = sample_function(p_sol, z_plot)
val_pc  = sample_function(pc_real, z_plot)

# === CORRECTION SECTION: Align with Benchmark Paper Coordinates ===

# 1. Flip Z-axis for plotting: Paper uses Depth (positive down)
#    Current z: [-2 (bottom) ... 2 (top)]
#    Paper z' : [-2 (top) ... 2 (bottom)] 
#    Actually, looking at paper Fig 2: Y-axis is -2 (Top) to 2 (Bottom).
plot_depth = -z_plot 

# 2. Flip Velocity Signs: 
#    In paper (Depth coords): Down is +, Up is -.
#    In simulation (Elevation coords): Down is -, Up is +.
#    Therefore: Plot_Value = -1 * Simulation_Value
val_u_plot  = -val_u
val_um_plot = -val_um

# 3. Shift Pressure: 
#    Paper pressure is centered around 0 (or removes hydrostatic mean).
#    Our P goes from 0 to ~8. Let's center it to match visual range [-4, 4].
val_pf_plot = val_pf - np.mean(val_pf)

# ----------------------------------------------------------------------------
# Plotting
fig, axes = plt.subplots(1, 5, figsize=(16, 8), sharey=True)
plt.subplots_adjust(wspace=0.4)

# 1. Porosity
axes[0].plot(val_phi, plot_depth, 'r-', linewidth=2)
axes[0].set_title("Porosity")
axes[0].set_xlabel("$\phi$")
axes[0].invert_yaxis() # -2 at top, 2 at bottom
axes[0].grid(True, linestyle=':')

# 2. Solid Velocity (Corrected Sign)
axes[1].plot(val_u_plot, plot_depth, 'r-', linewidth=2)
axes[1].set_title("Velocity (Solid)")
axes[1].set_xlabel("$u_s$")
axes[1].grid(True, linestyle=':')

# 3. Melt Velocity (Corrected Sign)
axes[2].plot(val_um_plot, plot_depth, 'r-', linewidth=2)
axes[2].set_title("Melt Velocity")
axes[2].set_xlabel("$v_m$")
axes[2].grid(True, linestyle=':')

# 4. Fluid Pressure (Centered)
axes[3].plot(val_pf_plot, plot_depth, 'r-', linewidth=2)
axes[3].set_title("Fluid Pressure")
axes[3].set_xlabel("$P_f$")
axes[3].grid(True, linestyle=':')

# 5. Compaction Pressure
axes[4].plot(val_pc, plot_depth, 'r-', linewidth=2)
axes[4].set_title("Compaction Pressure")
axes[4].set_xlabel("$P_c$")
axes[4].grid(True, linestyle=':')

plt.suptitle(f"Firedrake Reproduction of Arbogast et al. (2017) Benchmark\n(Sign conventions aligned with literature)", fontsize=14)
plt.savefig("arbogast_reproduction_comparison.png", dpi=150)
print("[INFO] Comparison plot saved to 'arbogast_reproduction_comparison.png'")