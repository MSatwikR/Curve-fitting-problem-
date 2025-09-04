import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Importing all required libraries

df = pd.read_csv("C:/Users/satwi/Desktop/Imprinttec/Stahl_N_20MnB4.csv", sep=';', encoding='latin1')
# using pandas reading the CSV file with tensile test data

print("A short description of the data: ", df.describe())

def stress_vs_strain_plot(stress, strain):
    plt.figure(figsize=(12, 10))
    plt.plot(strain, stress, label='Stress-Strain Curve')
    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (MPa)')
    plt.title('Engineering Stress-Strain Curve')
    plt.legend()
    plt.grid(False)
    plt.show()
#defined a function for plotting full stress-strain curve
whole_stress_strain_curve = stress_vs_strain_plot(df['Stress_MPa'], df['Strain_%'])


'''def ludwig(strain_p, sigma_0, k, n):
    return sigma_0 + (k*(strain_p)**n)
#n = hardening_exponent
#strain_p = Plastic strain
#k = Strength_coefficient
#sigma_0 = Yield stress


def elastic(strain_e, E):
    return strain_e*E
#strain_e = Elastic strain
#E = Elasticity coefficient transfer


def swift():
    pass'''


for col in ['Strain_%', 'Stress_MPa', 'Wahre Spannung_N/mm²']:
    if col in df.columns:
        s = df[col].astype(str).str.strip()
        s = s.replace(',', '.', regex=False)     # decimal comma to dot
        df[col] = pd.to_numeric(s, errors='coerce')
#Cleaning numeric columns for swapping dot and column

# Drop rows missing essentials
required = [c for c in ['Strain_%', 'Stress_MPa'] if c in df.columns]
df = df.dropna(subset=required).copy()


eps_eng = df['Strain_%'].to_numpy() / 100.0          # engineering strain (fraction)
sig_eng = df['Stress_MPa'].to_numpy()                # engineering stress (MPa)
#creating numpy arrays


# Prefer true stress if available
if 'Wahre Spannung_N/mm²' in df.columns:
    sig_true = df['Wahre Spannung_N/mm²'].to_numpy()
else:
    sig_true = sig_eng * (1.0 + eps_eng)

# True strain from engineering
eps_true = np.log1p(eps_eng)


mask_E = (eps_eng >= 0.0001) & (eps_eng <= 0.0025)
if mask_E.sum() < 5:
    N = min(50, len(eps_eng))
    mask_E = np.zeros_like(eps_eng, dtype=bool)
    mask_E[:N] = True
coef = np.polyfit(eps_eng[mask_E], sig_eng[mask_E], 1)  # [slope, intercept]
E_hat = float(coef[0])  # slope = Young's modulus
print(f"Estimated E (MPa): {E_hat:.1f}")
#Finding E using the slope of the curve


sigma_offset = E_hat * (eps_eng - 0.002)
yield_idx = int(np.argmin(np.abs(sig_eng - sigma_offset)))
sigma_y = float(sig_eng[yield_idx])
print(f"Yield approx: idx={yield_idx}, sigma_y={sigma_y:.1f} MPa")
#using E and eps_eng trying to find yield stress


#εp ≈ εtrue − σtrue/E beyond yield
ep_all = eps_true - sig_true / E_hat
mask_plastic = np.arange(len(ep_all)) >= yield_idx
ep_fit = ep_all[mask_plastic]
sig_fit = sig_true[mask_plastic]
#to use plastic strains after the yielding

valid = ep_fit > 0
ep_fit = ep_fit[valid]
sig_fit = sig_fit[valid]

def ludwig(ep, sigma0, K, n):
    return sigma0 + K * (ep ** n)

def swift():
    pass

#Fitting into Ludwik curve
sigma0_g = max(sigma_y, np.nanpercentile(sig_fit, 5))
n_g = float(np.clip(0.2, 0.01, 1.0))
K_g = (np.nanmax(sig_fit) - sigma0_g) / max(ep_fit.max()**n_g, 1e-6)

p0_lud = [sigma0_g, K_g, n_g]
lb_lud = np.array([0.0, 0.0, 0.01], dtype=float)
ub_lud = np.array([np.inf, np.inf, 1.0], dtype=float)

# Clip p0 into (bounds) with small epsilon margin
eps_b = 1e-8
p0_lud = np.array(p0_lud, dtype=float)
p0_lud = np.maximum(p0_lud, lb_lud + eps_b)
p0_lud = np.minimum(p0_lud, ub_lud - eps_b)

# Fallback if non-finite
if not np.all(np.isfinite(p0_lud)):
    p0_lud = np.array([
        max(sigma_y, np.nanmin(sig_fit)),
        max(np.nanmedian(sig_fit), 1.0),
        0.2
    ], dtype=float)
    p0_lud = np.maximum(p0_lud, lb_lud + eps_b)
    p0_lud = np.minimum(p0_lud, ub_lud - eps_b)

pars_lud, cov_lud = curve_fit(ludwig, ep_fit, sig_fit,p0=p0_lud)


sigma0_hat, K_hat, n_hat = [float(x) for x in pars_lud]
print({'E': E_hat, 'sigma0 (MPa)': sigma0_hat, 'K': K_hat, 'n (-)': n_hat})



fig, ax = plt.subplots(figsize=(9,6))
ax.plot(df['Strain_%'], df['Stress_MPa'], 'c.', ms=1, alpha=0.3, label='Engineering (for reference)')
ax.plot(eps_true, sig_true, 'k.', ms=2, alpha=0.5, label='True stress–strain')
ax.plot(ep_fit, ludwig(ep_fit, *pars_lud), 'r-', lw=2, label=f'Ludwik fit (σ0={sigma0_hat:.1f}, K={K_hat:.1f}, n={n_hat:.3f})')
ax.set_xlabel('Strain')
ax.set_ylabel('Stress (MPa)')
ax.legend()
ax.grid(True, alpha=0.3)

# Sparse ticks: min, mids, max for x using ep_fit; y using sig_fit
def sparse_ticks(a, n_middle=10):
    amin, amax = np.nanmin(a), np.nanmax(a)
    if amin == amax:
        return [amin]
    mids = np.linspace(amin, amax, n_middle + 2)[1:-1]
    return np.concatenate(([amin], mids, [amax]))

ax.set_xticks(sparse_ticks(ep_fit))
ax.set_yticks(sparse_ticks(sig_fit))
plt.tight_layout()
plt.show()
