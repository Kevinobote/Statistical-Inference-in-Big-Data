# Statistical Inference in Big Data

This repository contains solutions to various statistical inference problems implemented in R. Each question explores different concepts in probability theory, parameter estimation, hypothesis testing, and statistical distributions.

## Question 1: Negative Binomial Distribution

**Principles Used:**
- Negative Binomial Distribution: Models the probability of observing a specified number of successes before a specified number of failures
- Probability mass function calculation
- Combinatorial mathematics (using the choose function)

**Implementation:**
- Calculates the probability of getting the 5th head on the 10th flip of a fair coin
- Uses the negative binomial PMF: P(X = x) = choose(x - 1, r - 1) * p^r * (1 - p)^(x - r)
- Converts the decimal probability to fractional form

## Question 2: Method of Moments Estimation for Gamma Distribution

**Principles Used:**
- Method of Moments (MoM) estimation technique
- Sample statistics calculation (mean, variance)
- Parameter estimation for Gamma distribution

**Implementation:**
- Calculates sample mean and variance from given data
- Computes deviations from the mean and squared deviations
- Applies Method of Moments to estimate parameters α and β for a Gamma distribution:
  - α̂ = (mean²) / variance
  - β̂ = variance / mean

## Question 3: Unbiased Estimators and Mean Squared Error

**Principles Used:**
- Unbiased estimation
- Mean Squared Error (MSE) calculation
- Variance of linear combinations of random variables
- Efficiency comparison of estimators

**Implementation:**
- Simulates samples from normal distributions with different variances
- Constructs an unbiased estimator as a weighted average of two sample means
- Calculates and compares MSE for different estimators
- Demonstrates how to choose between estimators based on MSE

## Question 4: Efficiency and Cramer-Rao Lower Bound

**Principles Used:**
- Cramer-Rao Lower Bound (CRLB) theory
- Efficiency of estimators
- Variance calculation for estimators
- Comparison of different estimators for the mean

**Implementation:**
- Simulates data from a normal distribution
- Compares two estimators: Y = (X₁ + X₂)/2 and the sample mean
- Calculates the CRLB for estimating the mean
- Computes the efficiency of the estimator Y relative to the CRLB

## Question 5: Neyman-Pearson Lemma

**Principles Used:**
- Neyman-Pearson Lemma for hypothesis testing
- Likelihood ratio tests
- Critical value determination
- Type I error control (significance level α)

**Implementation:**
- Tests H₀: θ = 1 vs H₁: θ = 2 for a uniform distribution
- Computes the test statistic as the product of observations
- Determines critical value through simulation
- Makes a decision based on comparing the test statistic to the critical value

## Question 6: Likelihood Ratio Test for Variance

**Principles Used:**
- Likelihood ratio test methodology
- Chi-squared distribution for variance testing
- Two-sided hypothesis testing
- Critical region determination

**Implementation:**
- Tests H₀: σ² = 2 vs H₁: σ² ≠ 2
- Calculates the sample variance as the test statistic
- Determines critical values using the chi-squared distribution
- Makes a decision based on whether the test statistic falls in the rejection region

## Question 7: Binomial Hypothesis Testing

**Principles Used:**
- Binomial distribution properties
- Significance level calculation
- Power of a test
- One-sided hypothesis testing

**Implementation:**
- Tests H₀: θ = 0.5 vs H₁: θ < 0.5 for a binomial proportion
- Calculates significance levels for different critical values
- Simulates data under the alternative hypothesis
- Demonstrates the decision-making process in hypothesis testing

## Question 8: Chi-Square Goodness-of-Fit Test

**Principles Used:**
- Chi-square goodness-of-fit test
- Mendelian genetics ratios
- Expected vs observed frequencies comparison
- Statistical significance assessment

**Implementation:**
- Tests whether observed genetic frequencies follow Mendelian inheritance ratios (9:3:3:1)
- Calculates expected frequencies based on theoretical ratios
- Performs chi-square test to assess the fit between observed and expected frequencies
- Reports the test statistic and p-value for interpretation

## How to Use

1. Open the R script `Cat 1.R` in RStudio or any R environment
2. Run the entire script or individual question sections
3. Review the output for each question's calculations and interpretations

## Requirements

- R programming language
- MASS package (for certain probability distributions)