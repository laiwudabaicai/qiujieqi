# -*- coding: utf-8 -*-
import firedrake
from firedrake import *
import numpy as np

# ============================================================================
# 1. Real Physical Mesh (20 km depth)
# ============================================================================
# Domain: z from -10,000m to 10,000m (Total 20km)
# Resolution: 400 cells (50m per cell)
H = 10000.0
mesh = IntervalMesh(400, -H, H)
x = SpatialCoordinate(mesh)
z = x[0]

# Function Spaces
# V: Velocity (CG2), P: Fluid Pressure (CG1), Pc: Compaction Pressure (DG1)
V = FunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
Pc = FunctionSpace(mesh, "DG", 1)
W = MixedFunctionSpace([V, P, Pc])

# Solution Function
sol = Function(W, name="Solution")
# Split for defining the form (Fixes the "Linear Form" error)
u, p_f, p_c_tilde = split(sol)

# Test Functions
v, q_f, q_c = TestFunctions(W)

# ============================================================================
# 2. Real Physical Parameters (Ref: Table 3 in Paper) 
# ============================================================================
# Basic Constants
g_val = 9.81                  # Gravity (m/s^2)
rho_s_val = 3000.0            # Solid density (kg/m^3)
rho_f_val = 2500.0            # Fluid density (kg/m^3)
Delta_rho = rho_s_val - rho_f_val # Density contrast (~500 kg/m^3)

# Viscosities (Pa s)
eta_0_val = 1.0e18            # Shear viscosity [cite: 417]
xi_0_val  = 4.0e20            # Bulk viscosity [cite: 417]
mu_melt   = 1.0               # Melt viscosity [cite: 417]

# Permeability and Porosity
K0_val = 1.0e-7               # Reference Permeability (m^2) [cite: 417]
phi0_val = 0.05               # Reference Porosity (5%) [cite: 417]
n_exp = 3.0                   # Permeability exponent (k ~ phi^3) [cite: 201]

# Create Firedrake Constants
g = Constant(-g_val)          # Gravity acts downwards
eta = Constant(eta_0_val)
rho_s = Constant(rho_s_val)
rho_f = Constant(rho_f_val)

# ============================================================================
# 3. Parameter Fields Setup
# ============================================================================
# Porosity Profile: Quadratic in lower half (z < 0), Zero in upper half (z >= 0)
# Note: Realistically, melt is deeper. Let's say z < 0 is deep (melt), z > 0 is shallow (solid).
# phi = phi0 * (z / H)^2 for z < 0
phi = conditional(z < 0, phi0_val * (z/H)**2, 0.0)

# Avoid division by zero
epsilon = 1e-15

# --- Permeability K (m^2) ---
# k = k0 * (phi / phi0)^n
# K_D = k / mu_melt
permeability = K0_val * (phi / phi0_val)**n_exp
K_D = permeability / mu_melt

# --- Bulk Viscosity xi (Pa s) ---
# xi = xi0 * (phi0 / phi) [cite: 266]
# We add epsilon to phi to handle the limit, but the new formulation handles the singularity.
xi = xi_0_val * (phi0_val / (phi + epsilon))

# --- Scaling Factor d(phi) ---
# Definition: d(phi) = sqrt(K_D / K_D0) [cite: 106]
# K_D0 is the reference Darcy coefficient at phi0.
K_D0 = (K0_val * (1.0)**n_exp) / mu_melt
d_phi = sqrt(K_D / K_D0 + epsilon)

# ============================================================================
# 4. Variational Form (New Formulation)
# ============================================================================
# 1D derivatives: grad(u) -> u.dx(0), div(u) -> u.dx(0)

# --- F1: Stokes / Momentum ---
F1 = (
    2 * eta * u.dx(0) * v.dx(0) * dx
    - p_f * v.dx(0) * dx
    - (d_phi * p_c_tilde) * v.dx(0) * dx
    - (phi * rho_f + (1 - phi) * rho_s) * g * v * dx
)

# --- F2: Darcy / Fluid Mass ---
F2 = (
    u.dx(0) * q_f * dx
    + K_D * p_f.dx(0) * q_f.dx(0) * dx
    - K_D * rho_f * g * q_f.dx(0) * dx
)

# --- F3: Compaction Relation ---
# Eq (15): d * div(u) + d^2/xi * p_c = 0
compaction_term = (
    - (d_phi * u.dx(0)) * q_c * dx
    - (1.0/xi * d_phi**2 * p_c_tilde) * q_c * dx
)

# --- Constraint for Solid Region (z >= 0) ---
# Enforce p_c_tilde = 0 where phi is 0 (i.e., z >= 0)
is_solid = conditional(z >= 0, 1.0, 0.0)
# Penalty needs to be scaled relative to the huge viscosity (1e20)
# Let's use a value comparable to system scale or just very large
penalty = 1e25 
constraint_term = - penalty * is_solid * p_c_tilde * q_c * dx

F3 = compaction_term + constraint_term

F = F1 + F2 + F3

# ============================================================================
# 5. Boundary Conditions
# ============================================================================
# 1. Solid Velocity: Fixed at bottom (-H) and top (+H)
bc_u_bot = DirichletBC(W.sub(0), Constant(0.0), 1) # ID 1 is left/bottom (-H)
bc_u_top = DirichletBC(W.sub(0), Constant(0.0), 2) # ID 2 is right/top (+H)

# 2. Fluid Pressure: Fix at top to allow drainage or just reference
# Let's fix P at top = Hydrostatic pressure of solid (approx)
# Or just 0 for reference (simpler for convergence checking)
bc_p_top = DirichletBC(W.sub(1), Constant(0.0), 2)

bcs = [bc_u_bot, bc_u_top, bc_p_top]

# ============================================================================
# 6. Solve with MUMPS (Direct Solver)
# ============================================================================
# Direct solver is strongly recommended for 1D with such high viscosity contrasts
solver_parameters = {
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_monitor": None,
    "ksp_monitor": None
}

print("[INFO] Solving with Real Physical Parameters...")
print(f"   Viscosity: {eta_0_val:.1e} Pa s")
print(f"   Permeability: {K0_val:.1e} m^2")
print(f"   Density Contrast: {Delta_rho} kg/m^3")

solve(F == 0, sol, bcs=bcs, solver_parameters=solver_parameters)
print("[SUCCESS] Solution converged!")

# ============================================================================
# 7. Post-processing & Verification
# ============================================================================
u_sol, p_sol, pc_tilde_sol = sol.split()

# Recover real Compaction Pressure: p_c = d * p_c_tilde
pc_real = Function(Pc, name="Compaction Pressure (Pa)").interpolate(d_phi * pc_tilde_sol)

# Calculate Melt Velocity: v_melt = u_s + q/phi = u_s - K_D/phi * (grad(p_f) - rho_f*g)
# This is a bit complex to visualize in Paraview directly without projection,
# but we can save the primary variables.

# Save
outfile = File("output_real_physics_1d.pvd")
porosity_out = Function(P, name="Porosity").interpolate(phi)
outfile.write(u_sol, p_sol, pc_real, porosity_out)

print("[INFO] Results saved to output_real_physics_1d.pvd")
print("   -> Check 'Compaction Pressure'. Expected magnitude ~ 1e6 - 1e7 Pa (MPa range).")
print("   -> Check 'Velocity'. Expected magnitude ~ cm/yr (very small in m/s).")


#--------------------------------------------------------------------------------
print("\n" + "="*50)
print("   PHYSICS VERIFICATION")
print("="*50)

# 1. Verify Solid Velocity
u_vals = u_sol.dat.data_ro
u_max = np.max(np.abs(u_vals))  # Max velocity magnitude
u_cm_yr = u_max * 100 * (365 * 24 * 3600) # Convert m/s to cm/yr

print("1. Solid Velocity:")
print(f"   Max Value (Raw):  {u_max:.4e} m/s")
print(f"   Max Value (Unit): {u_cm_yr:.4f} cm/yr")

if 0.1 < u_cm_yr < 100:
    print("   [OK] Reasonable range (Mantle convection scale)")
else:
    print("   [WARNING] Value out of expected range!")

# 2. Verify Compaction Pressure
#    Only check melt region (z < 0), as solid region is constrained to 0
pc_vals = pc_real.dat.data_ro
# Create a mask for z < 0 if possible, or just check global max since solid is 0
pc_max = np.max(pc_vals)
pc_min = np.min(pc_vals)
pc_max_MPa = pc_max / 1.0e6

print("\n2. Compaction Pressure:")
print(f"   Range (Pa):  [{pc_min:.4e}, {pc_max:.4e}]")
print(f"   Max (MPa):   {pc_max_MPa:.4f} MPa")

if 1.0 < pc_max_MPa < 200.0:
    print("   [OK] Reasonable range (1-200 MPa)")
elif pc_max_MPa < 1e-5:
    print("   [WARNING] Too small (Did the solver do anything?)")
else:
    print("   [WARNING] Too large (Possible numerical instability)")

# 3. Verify Zero-Melt Constraint
#    Check the top boundary node (index -1), which is in the solid region
top_pc_value = pc_vals[-1] 
print("\n3. Zero-Melt Constraint Check (z > 0):")
print(f"   Top Surface Pc: {top_pc_value:.4e} Pa")

if abs(top_pc_value) < 1e-5:
    print("   [OK] Constraint active (Pc is zero in solid region)")
else:
    print("   [FAIL] Constraint failed (Pc is non-zero in solid region)")

print("="*50 + "\n")