import numpy as np

# Parameters
S0 = 100
K_low = 90
K_high = 110
coupon_increase = 1
maturity = 1
risk_free_rate = 0.02
num_paths = 10000
num_steps = 12
dt = maturity / num_steps

# Stock price dynamics (GBM)
mu = 0.05  # assumed drift
sigma = 0.2  # assumed volatility

# Seed for reproducibility
np.random.seed(42)

# Function to simulate stock price paths
def simulate_paths(S0, mu, sigma, dt, num_steps, num_paths):
    dW = np.random.normal(0, np.sqrt(dt), (num_steps, num_paths))
    S = np.zeros((num_steps + 1, num_paths))
    S[0, :] = S0
    for t in range(1, num_steps + 1):
        S[t, :] = S[t - 1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t - 1, :])
    return S

# Simulate stock price paths
S = simulate_paths(S0, mu, sigma, dt, num_steps, num_paths)

# Calculate snowball option payoff for each path
payoffs = np.zeros(num_paths)
for i in range(num_paths):
    path = S[:, i]
    coupon_value = 0
    for t in range(1, num_steps + 1):
        if K_low <= path[t] <= K_high:
            coupon_value += coupon_increase
    payoffs[i] = coupon_value

# Discount payoffs
discount_factor = np.exp(-risk_free_rate * maturity)
discounted_payoffs = payoffs * discount_factor

# Estimate option price
option_price = np.mean(discounted_payoffs)

# Calculate confidence interval
standard_error = np.std(discounted_payoffs) / np.sqrt(num_paths)
confidence_interval = (option_price - 1.96 * standard_error, option_price + 1.96 * standard_error)

print("Estimated Snowball Option Price: {:.2f}".format(option_price))
print("95% Confidence Interval: ({:.2f}, {:.2f})".format(confidence_interval[0], confidence_interval[1]))
