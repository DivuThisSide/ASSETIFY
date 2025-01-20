import numpy as np
from scipy.optimize import minimize

# Asset data
assets = {
    "Large Cap": {"return": 15.51, "volatility": 18.80},
    "Mid Cap": {"return": 26.77, "volatility": 20.53},
    "Small Cap": {"return": 26.25, "volatility": 22.12},
    "Emerging Market": {"return": 3.62, "volatility": 18.78},
    "International Equity": {"return": 12.96, "volatility": 17.85},
    "Indian Treasury": {"return": 6.07, "volatility": 2.85},
    "Corporate Bonds": {"return": 7.17, "volatility": 1.15},
    "Money Market": {"return": 6.15, "volatility": 1.04},
    "Gold": {"return": 13.75, "volatility": 12.98},
}

# Correlation matrix (example; adjust based on real data)
correlation_matrix = np.array([
    [1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.3],
    [0.7, 1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.3],
    [0.6, 0.8, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.3],
    [0.5, 0.6, 0.5, 1.0, 0.3, 0.2, 0.1, 0.1, 0.2],
    [0.4, 0.5, 0.4, 0.3, 1.0, 0.2, 0.1, 0.1, 0.2],
    [0.3, 0.3, 0.3, 0.2, 0.2, 1.0, 0.5, 0.5, 0.4],
    [0.2, 0.2, 0.2, 0.1, 0.1, 0.5, 1.0, 0.6, 0.3],
    [0.2, 0.1, 0.1, 0.1, 0.1, 0.5, 0.6, 1.0, 0.2],
    [0.3, 0.3, 0.3, 0.2, 0.2, 0.4, 0.3, 0.2, 1.0],
])

# User inputs
amount_to_invest = float(input("Enter the amount you want to invest: "))  # Investment amount
returns = np.array([assets[asset]["return"] for asset in assets]) / 100
volatilities = np.array([assets[asset]["volatility"] for asset in assets]) / 100
covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

# Step 1: Get equity and debt allocation
print("\nSet overall allocation for asset classes:")
equity_allocation = float(input("Enter the allocation for equity (Large Cap to International Equity) in %: ")) / 100
debt_allocation = float(input("Enter the allocation for debt (Indian Treasury to Gold) in %: ")) / 100

# Validate allocation
if equity_allocation + debt_allocation != 1.0:
    print("\nInvalid allocations. The total must equal 100%. Exiting.")
    exit()

# Step 2: Ask if user wants to set individual bounds or use defaults
use_custom_bounds = input(
    "\nDo you want to set individual bounds for each asset? (yes/no): ").strip().lower() == "yes"

if use_custom_bounds:
    # User sets bounds manually
    bounds = []
    print("\nSet individual allocation bounds for each asset class:")
    for asset in assets:
        min_allocation = float(input(f"Minimum allocation for {asset} (0-100%): ")) / 100
        max_allocation = float(input(f"Maximum allocation for {asset} (0-100%): ")) / 100
        if min_allocation > max_allocation:
            print(f"Invalid bounds for {asset}. Minimum cannot be greater than maximum. Exiting.")
            exit()
        bounds.append((min_allocation, max_allocation))
else:
    # Set bounds automatically based on equity and debt constraints
    equity_assets = list(assets.keys())[:5]  # Large Cap to International Equity
    debt_assets = list(assets.keys())[5:]   # Indian Treasury to Gold
    bounds = []
    for asset in equity_assets:
        bounds.append((equity_allocation / len(equity_assets), equity_allocation / len(equity_assets)))
    for asset in debt_assets:
        bounds.append((debt_allocation / len(debt_assets), debt_allocation / len(debt_assets)))

# Validate total allocation bounds
total_min = sum([b[0] for b in bounds])
total_max = sum([b[1] for b in bounds])
if total_max < 1.0 or total_min > 1.0:
    print("\nInvalid individual allocation ranges. Ensure total allocations cover 100%. Exiting.")
    exit()

# Print bounds for verification
print("\nFinal Bounds (Weight Constraints):")
for i, asset in enumerate(assets):
    print(f"{asset}: Min = {bounds[i][0]*100:.2f}%, Max = {bounds[i][1]*100:.2f}%")

# Initial guess: evenly distributed weights
initial_guess = np.ones(len(assets)) / len(assets)

# Objective function: Maximize returns (minimize negative returns)
def objective(x):
    return -np.dot(x, returns)

# Constraint: Total allocation should be 100%
def sum_constraint(x):
    return np.sum(x) - 1

# Constraint: Equity allocation
def equity_constraint(x):
    equity_indices = slice(0, 5)  # First 5 assets are equity
    return np.sum(x[equity_indices]) - equity_allocation

# Constraint: Debt allocation
def debt_constraint(x):
    debt_indices = slice(5, 9)  # Last 4 assets are debt
    return np.sum(x[debt_indices]) - debt_allocation

# Constraints list
constraints = [
    {"type": "eq", "fun": sum_constraint},
    {"type": "eq", "fun": equity_constraint},
    {"type": "eq", "fun": debt_constraint},
]

# Run optimization
result = minimize(
    objective,
    initial_guess,
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000}
)

# Check and display results
if result.success:
    optimal_weights = result.x
    portfolio_return = np.dot(optimal_weights, returns) * 100
    portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))) * 100
    risk_free_rate = 3.0  # Example: Risk-free rate as 3%
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

    print("\nOptimization Successful!")
    print("Optimal Allocation (Weights):")
    for i, asset in enumerate(assets):
        print(f"{asset}: {optimal_weights[i] * 100:.2f}%")
    print("\nOptimal Allocation (Amounts):")
    for i, asset in enumerate(assets):
        print(f"{asset}: ₹{optimal_weights[i] * amount_to_invest:.2f}")
    print("\nExpected Portfolio Return: {:.2f}%".format(portfolio_return))
    print("Expected Portfolio Risk: {:.2f}%".format(portfolio_risk))
    print("Sharpe Ratio: {:.2f}".format(sharpe_ratio))

    # Portfolio evaluation
    if sharpe_ratio > 1:
        print("\nNICE PORTFOLIO..INVESTABLE")
    else:
        print("\nRISK MUCH GREATER THAN RETURN. CHANGE ALLOCATION.")
else:
    print("\nOptimization Failed.")
    print("Reason:", result.message)
