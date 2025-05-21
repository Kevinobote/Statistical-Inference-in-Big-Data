## CAT 1
####################################
# Question 1
####################################

# Loading MASS
library(MASS)

# Required Parameters:
r <- 5      # Number of heads required (successes)
x <- 10     # The 10th flip is where the 5th head occurs
p <- 0.5    # Probability of head (success) in a fair coin

# Applying the Negative Binomial Distribution formula:
# P(X = x) = choose(x - 1, r - 1) * p^r * (1 - p)^(x - r)

prob <- choose(x - 1, r - 1) * p^r * (1 - p)^(x - r)

# Display the result
cat("Question 1:\n")
cat("Probability of getting the 5th head on the 10th flip is:", prob, "\n")

# Convert to fractional form
cat("Fraction form (approximate):", paste0(prob * 1024, "/1024"), "\n")


####################################
# Question 2
####################################

# Given data
data <- c(6.9, 7.3, 6.7, 6.4, 6.3, 5.9, 7.0, 7.1, 6.5, 7.6, 7.2, 7.1, 6.1,
          7.3, 7.6, 7.6, 6.7, 6.3, 5.7, 6.7, 7.5, 5.3, 5.4, 7.4, 6.9)

# Sample size
n <- length(data)

# Calculate sample mean
sample_mean <- mean(data)

# Calculate deviations from mean and squared deviations
deviation <- data - sample_mean
squared_deviation <- deviation^2

# Create a data frame for the table
results_df <- data.frame(
  Observation = 1:n,
  X = data,
  `X - Mean` = round(deviation, 4),
  `(X - Mean)^2` = round(squared_deviation, 4)
)

# Print the table
print(results_df, row.names = FALSE)

# Sum of squared deviations
sum_squared_deviation <- sum(squared_deviation)

# Sample variance (unbiased estimator divides by n-1)
sample_variance <- sum_squared_deviation / (n - 1)

# Method of Moments Estimators for Gamma(α, β):
# E[X] = αβ, Var[X] = αβ²
# => α̂ = (mean^2) / variance
# => β̂ = variance / mean

alpha_hat <- sample_mean^2 / sample_variance
beta_hat <- sample_variance / sample_mean

# Display summary
cat("\nSample Mean (x̄):", round(sample_mean, 4), "\n")
cat("Sum of (x - x̄)^2:", round(sum_squared_deviation, 4), "\n")
cat("Sample Variance (s²):", round(sample_variance, 4), "\n\n")

cat("Method of Moments Estimates:\n")
cat("Alpha (α̂):", round(alpha_hat, 4), "\n")
cat("Beta (β̂):", round(beta_hat, 4), "\n")

####################################
# Question 3
####################################

# Simulate samples
set.seed(123)  # For reproducibility

# True parameters
mu <- 10
sigma_sq <- 4

# Sample sizes
n_x <- 10
n_y <- 10

# Generate samples
X <- rnorm(n_x, mean = mu, sd = sqrt(sigma_sq))
Y <- rnorm(n_y, mean = mu, sd = 2 * sqrt(sigma_sq))  # Var = 4σ² ⇒ sd = 2σ

# Sample means
X_bar <- mean(X)
Y_bar <- mean(Y)

# (a) Unbiased estimator: μ̂ = αX̄ + (1 - α)Ȳ
alpha <- 0.5
mu_hat <- alpha * X_bar + (1 - alpha) * Y_bar

cat("(a) Unbiased Estimator:\n")
cat("X̄ =", round(X_bar, 4), " Ȳ =", round(Y_bar, 4), "\n")
cat("μ̂ =", round(mu_hat, 4), "\n\n")

# (b) MSE of μ̂ = Var(μ̂), since it's unbiased
# Var(X̄) = σ² / n = 4 / 10
# Var(Ȳ) = 4σ² / n = 16 / 10

var_X_bar <- sigma_sq / n_x        # Var(X̄)
var_Y_bar <- 4 * sigma_sq / n_y    # Var(Ȳ)

mse_mu_hat <- alpha^2 * var_X_bar + (1 - alpha)^2 * var_Y_bar

cat("(b) MSE of μ̂:\n")
cat("Var(X̄) =", round(var_X_bar, 4), " Var(Ȳ) =", round(var_Y_bar, 4), "\n")
cat("MSE(μ̂) =", round(mse_mu_hat, 4), "\n\n")

# (c) Compare MSEs for μ̂ = X̄ vs μ̂ = 0.5X̄ + 0.5Ȳ
mse_X_bar <- var_X_bar  # When α = 1
mse_weighted <- (0.5)^2 * var_X_bar + (0.5)^2 * var_Y_bar  # α = 0.5

cat("(c) MSE Comparison:\n")
cat("MSE(X̄) =", round(mse_X_bar, 4), "\n")
cat("MSE(0.5X̄ + 0.5Ȳ) =", round(mse_weighted, 4), "\n")

if (mse_X_bar < mse_weighted) {
  cat("Conclusion: X̄ is preferable due to lower MSE.\n")
} else {
  cat("Conclusion: The weighted estimator is preferable.\n")
}

####################################
# Question 4
####################################

# Set parameters
mu <- 5         # True mean
sigma2 <- 4     # Known variance (i.e., σ²)
sigma <- sqrt(sigma2)
n <- 10         # Sample size

# Simulate data from N(mu, sigma^2)
set.seed(123)   # For reproducibility
x <- rnorm(n, mean = mu, sd = sigma)

# Estimator Y = (X1 + X2)/2
Y <- (x[1] + x[2]) / 2

# Sample mean using all n observations
x_bar <- mean(x)

# Variance of Y from its definition (only uses X1 and X2)
var_Y <- var(c(x[1], x[2])) / 2

# Cramer-Rao Lower Bound for estimating mu
crlb <- sigma2 / n

# Efficiency of Y
efficiency_Y <- crlb / var_Y

# Output results
cat("Simulated Sample:\n", round(x, 2), "\n\n")
cat("Y = (X1 + X2)/2:", round(Y, 4), "\n")
cat("Sample Mean (x̄):", round(x_bar, 4), "\n")
cat("Variance of Y:", round(var_Y, 4), "\n")
cat("CRLB:", round(crlb, 4), "\n")
cat("Efficiency of Y:", round(efficiency_Y, 4), "\n")

####################################
# Question 5:Neyman-Pearson Lemma
####################################
# Define parameters
theta0 <- 1  # Under H0
theta1 <- 2  # Under H1
n <- 20      # Sample size
alpha <- 0.05  # Significance level

# Simulate data under H0 for demonstration
set.seed(123)
data <- runif(n)  # Since f(x;1) = 1, X ~ Uniform(0,1)

# Compute the test statistic: product of xi
test_statistic <- prod(data)

# Compute the critical value c such that P(prod X_i >= c | H0) = alpha
# We approximate this using simulation under H0

simulate_critical_value <- function(n, alpha, B = 100000) {
  simulated_prods <- replicate(B, prod(runif(n)))
  critical_value <- quantile(simulated_prods, probs = 1 - alpha)
  return(critical_value)
}

# Calculate critical value for given n and alpha
critical_value <- simulate_critical_value(n, alpha)

# Print results
cat("Test Statistic (Π x_i):", round(test_statistic, 6), "\n")
cat("Critical Value at alpha =", alpha, ":", round(critical_value, 6), "\n")

# Decision Rule
if (test_statistic >= critical_value) {
  cat("Reject H0: Evidence suggests θ = 2\n")
} else {
  cat("Fail to Reject H0: Insufficient evidence to suggest θ = 2\n")
  
  ####################################
  # Question 6 :likelihood ratio
  ####################################
  # Sample size
  n <- 30
  
  # True values for simulation
  theta1 <- 5      # mean (unknown)
  theta2_prime <- 2  # H0 variance
  theta2_alt <- 5    # alternative variance
  
  # Simulate data under H0
  set.seed(123)
  x <- rnorm(n, mean = theta1, sd = sqrt(theta2_prime))
  
  # Compute test statistic: sample variance (with divisor n)
  x_bar <- mean(x)
  S2 <- sum((x - x_bar)^2) / n
  
  # Define critical values under H0: scaled chi-squared distribution
  alpha <- 0.05
  lower_crit <- theta2_prime * qchisq(alpha / 2, df = n - 1) / n
  upper_crit <- theta2_prime * qchisq(1 - alpha / 2, df = n - 1) / n
  
  # Decision rule
  if (S2 <= lower_crit || S2 >= upper_crit) {
    result <- "Reject H0: Variance is significantly different"
  } else {
    result <- "Fail to Reject H0: No significant difference"
  }
  
  # Print results
  cat("Sample Variance:", round(S2, 4), "\n")
  cat("Lower Critical Value:", round(lower_crit, 4), "\n")
  cat("Upper Critical Value:", round(upper_crit, 4), "\n")
  cat("Test Result:", result, "\n")
  
}

####################################
# Question 7
####################################
# Set sample size
n <- 5

# Define theta under H0
theta0 <- 0.5

# Define Binomial distribution under H0: Y ~ Binomial(n, theta0)
# Function to compute significance level for a given critical value c
significance_level <- function(c, n, theta) {
  prob <- pbinom(c, size = n, prob = theta)
  return(prob)
}

# Significance level when c = 1
alpha_c1 <- significance_level(1, n, theta0)
cat("Significance level (α) when c = 1:", alpha_c1, "\n")  # Output: 0.1875

# Significance level when c = 0
alpha_c0 <- significance_level(0, n, theta0)
cat("Significance level (α) when c = 0:", alpha_c0, "\n")  # Output: 0.03125

# Simulating the test: Generate data and apply test
set.seed(123)  # For reproducibility

simulate_test <- function(theta, n, c) {
  sample <- rbinom(n, size = 1, prob = theta)
  y <- sum(sample)
  reject_H0 <- y <= c
  list(sample = sample, Y = y, Reject_H0 = reject_H0)
}

# Simulate under theta = 0.3 (alternative hypothesis), c = 1
sim_result <- simulate_test(theta = 0.3, n = n, c = 1)
cat("Simulated Sample:", sim_result$sample, "\n")
cat("Sum Y:", sim_result$Y, "\n")
cat("Reject H0:", sim_result$Reject_H0, "\n")

####################################
# Question 8
####################################
# Observed frequencies
observed <- c(124, 30, 43, 11)

# Expected ratios based on Mendelian theory
expected_ratios <- c(9, 3, 3, 1)
total <- sum(observed)

# Expected counts
expected <- expected_ratios / sum(expected_ratios) * total

# Perform Chi-Square Test
chi_test <- chisq.test(x = observed, p = expected_ratios / sum(expected_ratios))

# Output results
print(chi_test)