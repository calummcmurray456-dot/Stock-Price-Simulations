import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

S0 = 100
T = 1.0
N = 252
dt = T / N

a1, b1 = 0.05, 0.1
a2, b2 = 0.15, 0.3

t = np.linspace(0, T, N+1)

# Set 1
np.random.seed(42)
dW1 = np.sqrt(dt) * np.random.normal(0, 1, N)
W1 = np.cumsum(dW1)
W1 = np.insert(W1, 0, 0)

S1_num = np.zeros(N+1)
S1_num[0] = S0
for i in range(N):
    dS = a1 * S1_num[i] * dt + b1 * S1_num[i] * dW1[i]
    S1_num[i+1] = S1_num[i] + dS

S1_exact = S0 * np.exp((a1 - 0.5 * b1**2) * t + b1 * W1)

# Set 2
np.random.seed(100)
dW2 = np.sqrt(dt) * np.random.normal(0, 1, N)
W2 = np.cumsum(dW2)
W2 = np.insert(W2, 0, 0)

S2_num = np.zeros(N+1)
S2_num[0] = S0
for i in range(N):
    dS = a2 * S2_num[i] * dt + b2 * S2_num[i] * dW2[i]
    S2_num[i+1] = S2_num[i] + dS

S2_exact = S0 * np.exp((a2 - 0.5 * b2**2) * t + b2 * W2)

print("First set of parameters")
print(f" Final Price (Numerical): £{S1_num[-1]:.2f}")
print(f" Final Price (Exact): £{S1_exact[-1]:.2f}")
print(f" Error: {abs(S1_num[-1]-S1_exact[-1])/S1_exact[-1]*100:.3f}%")
print("Second set of parameters")
print(f" Final Price (Numerical): £{S2_num[-1]:.2f}")
print(f" Final Price (Exact): £{S2_exact[-1]:.2f}")
print(f" Error: {abs(S2_num[-1]-S2_exact[-1])/S2_exact[-1]*100:.3f}%")


fig, ax = plt.subplots(figsize=(10, 6))

# Set 1 (blue shades)
ax.plot(t, S1_exact, color='black', linestyle='-', linewidth=2, 
        label='Set 1 Exact (a=0.05, b=0.1)')
ax.plot(t, S1_num, color='yellow', linestyle='--', linewidth=1.5, 
        label='Set 1 Numerical')

# Set 2 (red/orange shades)
ax.plot(t, S2_exact, color='red', linestyle='-', linewidth=2, 
        label='Set 2 Exact (a=0.15, b=0.3)')
ax.plot(t, S2_num, color='blue', linestyle='--', linewidth=1.5, 
        label='Set 2 Numerical')

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Stock Price (£)', fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('combined_parameter_comparison.pdf', dpi=1000, bbox_inches='tight')
print("Saved: combined_parameter_comparison.pdf")
plt.show()

#Extension
print("S&P 500 Calibration Simulation")

# Download S&P 500 historical data
sp500_data = yf.download('^GSPC', start='2020-01-01', end='2025-01-01', progress=False)
prices = sp500_data['Close']

log_returns = np.log(prices / prices.shift(1)).dropna()

a_sp500 = (log_returns.mean() * 252).item()
b_sp500 = (log_returns.std() * np.sqrt(252)).item()

print(f"\nCalibrated parameters from S&P 500 (2020-2025):")
print(f"  Drift (a): {a_sp500:.4f} ({a_sp500*100:.2f}% annual return)")
print(f"  Volatility (b): {b_sp500:.4f} ({b_sp500*100:.2f}% annual volatility)")

# Simulate forward one year using calibrated parameters
S0_sp500 = prices.iloc[-1].item()  # Also use .item() here
T_sp500 = 1.0
N_sp500 = 252
dt_sp500 = T_sp500 / N_sp500
t_sp500 = np.linspace(0, T_sp500, N_sp500+1)

np.random.seed(123)
dW_sp500 = np.sqrt(dt_sp500) * np.random.normal(0, 1, N_sp500)
W_sp500 = np.cumsum(dW_sp500)
W_sp500 = np.insert(W_sp500, 0, 0)

S_sp500_num = np.zeros(N_sp500+1)
S_sp500_num[0] = S0_sp500
for i in range(N_sp500):
    dS = a_sp500 * S_sp500_num[i] * dt_sp500 + b_sp500 * S_sp500_num[i] * dW_sp500[i]
    S_sp500_num[i+1] = S_sp500_num[i] + dS

S_sp500_exact = S0_sp500 * np.exp((a_sp500 - 0.5 * b_sp500**2) * t_sp500 + b_sp500 * W_sp500)

print(f"\nOne-year forward simulation:")
print(f"  Starting price: ${S0_sp500:.2f}")
print(f"  Projected price (numerical): ${S_sp500_num[-1]:.2f}")
print(f"  Projected price (exact): ${S_sp500_exact[-1]:.2f}")
print(f"  Error: {abs(S_sp500_num[-1]-S_sp500_exact[-1])/S_sp500_exact[-1]*100:.3f}%")

# Plot S&P 500 simulation
fig_sp500, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_sp500, S_sp500_exact, color='black', linestyle='-', label='Exact Solution', linewidth=2)
ax.plot(t_sp500, S_sp500_num, color='yellow', linestyle='--', label='Euler-Maruyama', linewidth=1.5)
ax.axhline(y=S0_sp500, color='grey', linestyle=':', alpha=0.7, label='Starting Price')
ax.set_xlabel('Time (years)')
ax.set_ylabel('S&P 500 Index ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Monte Carlo
print("Monte Carlo Simulatio With Real S&P 500 Data")
print("\nDownloading actual 2025 S&P 500 data")
sp500_real = yf.download('^GSPC', start='2025-01-01', end='2026-01-01', progress=False)
real_prices = sp500_real['Close'].dropna()
real_returns = (real_prices / real_prices.iloc[0] - 1) * 100
real_dates = real_prices.index
days_from_start = [(date - real_dates[0]).days for date in real_dates]
real_months = [day / 30.44 for day in days_from_start]  # Convert to months

print(f"Real S&P 500 data points: {len(real_prices)}")
print(f"Starting price: ${real_prices.iloc[0].item():.2f}")
print(f"Ending price: ${real_prices.iloc[-1].item():.2f}")
print(f"Actual 2025 return: {real_returns.iloc[-1].item():.2f}%")

# Params calibrated from historical data
a_sp500 = 0.1639
b_sp500 = 0.1690
S0_sp500 = 100
T_sp500 = 1.0
N_sp500 = 252
dt_sp500 = T_sp500 / N_sp500
t_sp500 = np.linspace(0, T_sp500, N_sp500+1)
t_months = t_sp500 * 12

M = 1000
np.random.seed(999)

all_returns = np.zeros((M, N_sp500+1))

print(f"\nRunning {M} Monte Carlo simulations...")
for j in range(M):
    dW_mc = np.sqrt(dt_sp500) * np.random.normal(0, 1, N_sp500)
    S_mc = np.zeros(N_sp500+1)
    S_mc[0] = S0_sp500

    for i in range(N_sp500):
        S_mc[i+1] = S_mc[i] + a_sp500 * S_mc[i] * dt_sp500 + b_sp500 * S_mc[i] * dW_mc[i]

    all_returns[j, :] = (S_mc / S0_sp500 - 1) * 100

average_path = np.mean(all_returns, axis=0)
positive_count = np.sum(all_returns[:, -1] > 0)

print(f"\nMonte Carlo Statistics:")
print(f"  Paths with positive returns: {positive_count}/{M} ({positive_count/M*100:.1f}%)")
print(f"  Average final return: {average_path[-1]:.2f}%")
print(f"  Min final return: {np.min(all_returns[:, -1]).item():.2f}%")
print(f"  Max final return: {np.max(all_returns[:, -1]).item():.2f}%")
print(f"  Actual S&P 500 return: {real_returns.iloc[-1].item():.2f}%")

# Check if actual return is within simulated range
within_range = np.min(all_returns[:, -1]) <= real_returns.iloc[-1].item() <= np.max(all_returns[:, -1])
print(f"\nValidation: Actual return {'is' if within_range else 'is not'} within simulated range " if within_range else "✗")

fig_mc, ax = plt.subplots(figsize=(12, 7))

for j in range(M):
    ax.plot(t_months, all_returns[j, :], color='blue', alpha=0.1, linewidth=0.5)

ax.plot(t_months, average_path, color='red', linewidth=2.5, label='Average Path', zorder=10)

# Plot Real S&P 500 data
ax.plot(real_months, real_returns.values, color='black', linewidth=2.5, 
        label='Actual 2025 S&P 500', zorder=11, marker='', linestyle='-')

ax.set_xlabel('t (months)', fontsize=12)
ax.set_ylabel('Total Return (%)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('monte_carlo.pdf', dpi=1000, bbox_inches='tight')
print("\nSaved monte_carlo.pdf")
plt.show()


print("MC Sim Complete")
print("\nThe actual 2025 S&P 500 performance has been overlaid on the simulated distribution, validating the GBM model's ability to capture real market behaviour within its probability envelope.")


