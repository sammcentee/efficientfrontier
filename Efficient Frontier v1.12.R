# Enable multithreading
library(doParallel) # speed up computations by using multiple processors
cl <- makeCluster(12) # I have 8 cores on my machine (keep 2 free)
registerDoParallel(cl) # start parallel computing backend

# Load necessary libraries
library(quantmod)
library(PerformanceAnalytics)
library(quadprog)
library(MCMCpack)
library(readxl)
library(readODS)
library(ggplot2)

# Specify the Excel file path with double backslashes
excel_file <- "~/Excel/spy_holdings.ods"

# Read data from the Excel file
data <- read_ods(excel_file)

# Extract the SPY ticker list from the Excel file
weird_data <- as.vector(data)
filtered_data <- weird_data$`Tickers`
indices_to_keep <- !is.na(filtered_data)
clean_data <- filtered_data[indices_to_keep]
symbols <- clean_data

getSymbols(symbols, src = "yahoo", from = "2020-01-01", to = "2024-01-01")

# Extract adjusted close prices and calculate daily returns
with_na_prices <- do.call(merge, lapply(symbols, function(sym) Ad(get(sym))))
with_na_returns <- Return.calculate(with_na_prices)[-1] # First Row is filled with NA after calculating returns, so we drop it
returns <- with_na_returns[,!apply(with_na_returns, 2, anyNA)]
prices <- with_na_prices[,!apply(with_na_returns, 2, anyNA)]
pricesframe <- as.data.frame(prices)

# Calculate mean returns and covariance matrix
mean_returns <- colMeans(returns) #This might be dangerous and not working properly!!!!!!!!
mean_returns <- as.numeric((pricesframe[length(pricesframe[,1]),]/pricesframe[1,])/(length(pricesframe[,1]-1)))

cov_matrix <- cov(returns)

# Number of portfolios to simulate
num_portfolios <- 10^7

# Function to calculate portfolio performance
portfolio_performance <- function(weights, mean_returns, cov_matrix, n_days = 252) {
  portfolio_return <- sum(weights * mean_returns) * n_days
  portfolio_std <- sqrt(t(weights) %*% cov_matrix %*% weights) * sqrt(n_days)
  return(c(portfolio_return, portfolio_std))
}

# Generate random portfolios
results <- matrix(nrow = num_portfolios, ncol = 3)
colnames(results) <- c("Return", "Risk", "Sharpe")
weights_record <- matrix(nrow = num_portfolios, ncol = length(returns[1,]))
colnames(weights_record) <- symbols[!apply(with_na_returns, 2, anyNA)]

  # Function to generate a vector of Dirichlet samples
  generate_dirichlet_vector <- function(n) {
    alpha <- rep(1/(n^2), n)  # Parameters for the Dirichlet distribution
    as.vector(rdirichlet(1, alpha))
  }
    # Use the Dirichlet samples to generate weighting of each portfolio
    # This is an efficient method for sampling possible weightings when the number of stocks is large
    set.seed(11111)
    for (i in 1:num_portfolios) {
      weights <- c(rep(0,length(returns[1,])))
      weights_numbers <- generate_dirichlet_vector(sample(5:40, 1))
      positions <- sample(1:length(returns[1,]), length(weights_numbers))
      weights[positions] <- weights_numbers
      weights_record[i, ] <- weights
      portfolio <- portfolio_performance(weights, mean_returns, cov_matrix)
      results[i, 1] <- portfolio[1]
      results[i, 2] <- portfolio[2]
      results[i, 3] <- portfolio[1] / portfolio[2]  # Sharpe ratio
    }

# Convert results to data frame and ensure numeric types
results_df <- as.data.frame(results)
results_df$Return <- as.numeric(as.character(results_df$Return))
results_df$Risk <- as.numeric(as.character(results_df$Risk))
results_df$Sharpe <- as.numeric(as.character(results_df$Sharpe))

# Convert results to data frame
weights_df <- as.data.frame(weights_record)

# Find the portfolio with the highest Sharpe ratio
max_sharpe_port <- results_df[which.max(results_df$Sharpe), ]
max_sharpe_weights <- weights_df[which.max(results_df$Sharpe), ]
max_sharpe_weights <- max_sharpe_weights[which(max_sharpe_weights > 0.001, arr.ind = T)]

# Find the portfolio with the minimum standard deviation
min_std_port <- results_df[which.min(results_df$Risk), ]
min_std_weights <- weights_df[which.min(results_df$Risk), ]
min_std_weights <- min_std_weights[which(min_std_weights > 0.001, arr.ind = T)]

# Find the portfolio with the highest returns
max_return_weights <- weights_df[which.max(results_df$Return), ]
max_return_weights <- max_return_weights[which(max_return_weights > 0.001, arr.ind = T)]

# Plot the efficient frontier
ggplot(results_df, aes(x = Risk, y = Return, color = Sharpe)) +
  geom_point(alpha = 0.5) +
  scale_color_gradient(low = "yellow", high = "red") +
  labs(title = "Efficient Frontier",
       x = "Volatility (Annualised Standard Deviation %)",
       y = "Return (Annualised %)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  annotate("point", x = max_sharpe_port$Risk, y = max_sharpe_port$Return, color = "blue", size = 3, shape = 17) +
  annotate("point", x = min_std_port$Risk, y = min_std_port$Return, color = "green", size = 3, shape = 17) +
  annotate("text", x = max_sharpe_port$Risk, y = max_sharpe_port$Return, label = "Max Sharpe Ratio", hjust = -0.1, vjust = -0.5) +
  annotate("text", x = min_std_port$Risk, y = min_std_port$Return, label = "Min Volatility", hjust = -0.1, vjust = -0.5) +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent)

print(max_sharpe_weights)
print(max_sharpe_port)
print(max_return_weights)

stopCluster(cl) # Turn off multithreading, not sure why but I'm supposed to

