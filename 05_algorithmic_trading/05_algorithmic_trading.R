######################################################################
## R Code for Algorithmic Trading Backtesting
## Incorporating Sentiment Indicators for Strategy Evaluation
## Results are commented
######################################################################

## Initialise environment
rm(list = ls())

## Set working directory
fileloc <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(fileloc)
rm(fileloc)

## Load required libraries
library(xts)
library(zoo)
library(quantmod)
library(PerformanceAnalytics)
library(matrixStats)
library(doParallel)
library(ggplot2)
library(tidyr)
library(dplyr)
library(TTR)

######################################################################
## Function: Performance Metrics for Trading Strategies
######################################################################
Performance <- function(x, ntrades = 1, cost = 0) {
  # Define a default NA performance vector for cases where 'x' is too short or invalid
  NA_PERF_VECTOR <- c(
    "Cumulative Return" = NA, "Annual Return" = NA, "Annualized Sharpe Ratio" = NA, "Win %" = NA,
    "Annualized Volatility" = NA, "Maximum Drawdown" = NA, "Max Length Drawdown" = NA, "n.trades" = NA
  )

  # Check if x has enough valid data points for calculations
  if (NROW(na.omit(x)) < 2) { # Many performance metrics require at least 2 points for std dev, etc.
    return(NA_PERF_VECTOR)
  }

  cumRetx = Return.cumulative(x, geometric = TRUE) - ntrades * cost
  annRetx = (cumRetx + 1)^(252 / length(x)) - 1
  sharpex = annRetx / sd.annualized(x, scale = 252)

  # Handle cases where denominator for Win % might be zero (no non-zero returns)
  non_zero_returns_count <- length(x[x != 0])
  winpctx <- ifelse(non_zero_returns_count > 0, length(x[x > 0]) / non_zero_returns_count, NA)

  annSDx = sd.annualized(x, scale = 252)

  # findDrawdowns can fail on very short or constant series
  DDs <- tryCatch({
    findDrawdowns(x)
  }, error = function(e) {
    list(return = NA, length = NA) # Return NA for drawdowns if calculation fails
  })

  maxDDx = ifelse(!is.null(DDs$return) && length(DDs$return) > 0, min(DDs$return), NA)
  maxLx = ifelse(!is.null(DDs$length) && length(DDs$length) > 0, max(DDs$length), NA)


  Perf = c(cumRetx, annRetx, sharpex, winpctx, annSDx, maxDDx, maxLx, ntrades)
  names(Perf) = c(
    "Cumulative Return", "Annual Return", "Annualized Sharpe Ratio", "Win %",
    "Annualized Volatility", "Maximum Drawdown", "Max Length Drawdown", "n.trades"
  )
  return(Perf)
}

######################################################################
## Trading Strategy Implementation
## Functions now accept Bollinger Band and RVT quantile parameters.
######################################################################

## 1) Full-Period Strategy Test Function
testMyStrategy <- function(myStock,
                           ev_pnlog, ev_rvt,
                           longshort = 0,
                           tcost = 0.001,
                           bb_n = 20,       # Bollinger Band lookback period
                           bb_sd = 1,       # Bollinger Band standard deviation multiplier
                           rvt_quantile = 0.60) { # RVT quantile threshold

  # Define a placeholder for a failed strategy performance output
  FAILED_PERF_MATRIX <- matrix(NA, nrow = 8, ncol = 2)
  rownames(FAILED_PERF_MATRIX) <- c("Cumulative Return", "Annual Return", "Annualized Sharpe Ratio", "Win %",
                                    "Annualized Volatility", "Maximum Drawdown", "Max Length Drawdown", "n.trades")
  colnames(FAILED_PERF_MATRIX) <- c("Me", "BH")

  # Calculate Bollinger Bands. Use tryCatch for robustness against problematic 'n' values.
  bb_result <- tryCatch({
    BBands(ev_pnlog, n = bb_n, sd = bb_sd)
  }, error = function(e) {
    message(paste("BBands calculation failed for bb_n=", bb_n, ", bb_sd=", bb_sd, ". Error:", e$message))
    return(NULL)
  })

  # If BBands calculation failed or resulted in all NAs for bands, return failed performance
  if (is.null(bb_result) || all(is.na(bb_result$up))) {
    return(FAILED_PERF_MATRIX)
  }

  # Calculate RVT threshold
  rvtThr <- quantile(ev_rvt, rvt_quantile, na.rm = TRUE)

  # Generate trading signal
  sig <- Lag(
    ifelse(ev_pnlog > bb_result$up & ev_rvt > rvtThr,
           1, longshort),
    1
  )

  # Count trades.
  valid_sig <- na.omit(sig)
  ntr <- if (length(valid_sig) > 0) length(rle(as.vector(valid_sig))$lengths) else 0

  # Compute benchmark (Buy and Hold) and strategy returns
  bmkR <- dailyReturn(myStock, type = "arithmetic")
  myR  <- bmkR * sig

  names(bmkR) <- "BH"
  names(myR) <- "Me"

  tt <- na.omit(merge(bmkR, myR))

  # Check if 'tt' has enough data points to compute performance meaningfully.
  if (NROW(tt) < 2) {
    message(paste("Insufficient valid data after merging returns for bb_n=", bb_n, ", bb_sd=", bb_sd, ", rvt_q=", rvt_quantile))
    return(FAILED_PERF_MATRIX)
  }

  # Ensure the Performance function call returns the expected matrix
  perf_output <- tryCatch({
    cbind(Me = Performance(tt$Me, ntr, tcost),
          BH = Performance(tt$BH, 1, tcost))
  }, error = function(e) {
    message(paste("Performance calculation failed for bb_n=", bb_n, ", bb_sd=", bb_sd, ", rvt_q=", rvt_quantile, ". Error:", e$message))
    return(FAILED_PERF_MATRIX)
  })

  return(perf_output)
}

## 2) Rolling-Window Strategy Test Function (Modified to accept tuning parameters)
RollingMyStrategy <- function(myStock, ev_pnlog, ev_rvt,
                              longshort = 0,
                              tcost = 0.001,
                              w_size = 252,
                              bb_n = 20,       # Bollinger Band lookback period
                              bb_sd = 1,       # Bollinger Band standard deviation multiplier
                              rvt_quantile = 0.60) { # RVT quantile threshold


  bmkR <- dailyReturn(myStock, type = "arithmetic")
  df   <- na.omit(merge(bmkR, ev_pnlog, ev_rvt))
  colnames(df) <- c("BH", "PNlog", "RVT")

  Nwin <- nrow(df) - w_size
  if (Nwin < 1) stop("Insufficient data for the specified window size.")

  allRes <- foreach(i = 1:Nwin, .combine = rbind) %do% {
    sub <- df[i:(i + w_size - 1), ]

    # Recalculate signals for the current window using passed parameters
    bb       <- BBands(sub$PNlog, n = bb_n, sd = bb_sd)
    rvtThr   <- quantile(sub$RVT, rvt_quantile, na.rm = TRUE)
    sig      <- Lag(ifelse(sub$PNlog > bb$up & sub$RVT > rvtThr, 1, longshort), 1)

    runs     <- rle(as.vector(na.omit(sig)))
    ntr      <- length(runs$lengths)

    stratR   <- sub$BH * sig
    df2      <- na.omit(merge(sub$BH, stratR))
    names(df2) <- c("BH", "Me")

    perf_output_window <- tryCatch({
      rbind(Me = Performance(df2$Me, ntr, tcost),
            BH = Performance(df2$BH, 1, tcost))
    }, error = function(e) {
      # Return a 2x8 matrix of NAs if calculation fails for this window
      matrix(NA, nrow = 2, ncol = 8,
             dimnames = list(c("Me", "BH"), c("Cumulative Return", "Annual Return", "Annualized Sharpe Ratio", "Win %",
                                              "Annualized Volatility", "Maximum Drawdown", "Max Length Drawdown", "n.trades")))
    })
    perf_output_window
  }

  meIdx <- seq(1, nrow(allRes), by = 2)
  bhIdx <- seq(2, nrow(allRes), by = 2)

  # Handle cases where colMeans might receive all NAs
  MeAvg <- tryCatch(colMeans(allRes[meIdx, ], na.rm = TRUE), error = function(e) rep(NA, ncol(allRes)))
  BHAvg <- tryCatch(colMeans(allRes[bhIdx, ], na.rm = TRUE), error = function(e) rep(NA, ncol(allRes)))

  rbind(Me = MeAvg, BH = BHAvg)
}

######################################################################
## Data Acquisition and Pre-processing
######################################################################

## 3) Load Financial Assets: DB and HSBC
# DB
x_DB   <- read.csv("data/DB.csv", stringsAsFactors = FALSE)
xt_DB  <- xts(x_DB$Adj.Close, order.by = as.Date(x_DB$Date))

pnDB   <- xts(0.5 * log((x_DB$positivePartscr + 1) / (x_DB$negativePartscr + 1)),
              order.by = as.Date(x_DB$Date))
rvtDB  <- xts(x_DB$RVT, order.by = as.Date(x_DB$Date))

library(zoo)
pnDB   <- na.locf(pnDB)
pnDB   <- na.locf(pnDB, fromLast = TRUE)
rvtDB  <- na.locf(rvtDB)
rvtDB  <- na.locf(rvtDB, fromLast = TRUE)

# HSBC
x_HSBC <- read.csv("data/HSBC.csv", stringsAsFactors = FALSE)
xt_H   <- xts(x_HSBC$Adj.Close, order.by = as.Date(x_HSBC$Date))

pnH    <- xts(0.5 * log((x_HSBC$positivePartscr + 1) / (x_HSBC$negativePartscr + 1)),
              order.by = as.Date(x_HSBC$Date))
rvtH   <- xts(x_HSBC$RVT, order.by = as.Date(x_HSBC$Date))

pnH    <- na.locf(pnH);     pnH    <- na.locf(pnH, fromLast = TRUE)
rvtH   <- na.locf(rvtH);     rvtH   <- na.locf(rvtH, fromLast = TRUE)

######################################################################
## Strategy Execution and Performance Reporting (Using Tuned Parameters)
######################################################################

# Optimal parameters identified from tuning for each asset
# DB Optimal: BB_n=5, BB_sd=1.5, RVT_Quantile=0.7
db_tuned_bb_n <- 5
db_tuned_bb_sd <- 1.5
db_tuned_rvt_quantile <- 0.7

# HSBC Optimal: BB_n=5, BB_sd=1.0, RVT_Quantile=0.5
hsbc_tuned_bb_n <- 5
hsbc_tuned_bb_sd <- 1.0
hsbc_tuned_rvt_quantile <- 0.5


## 4) Full-Period Performance Analysis: Long-Only vs. Long-Short
cat("\n--- Full-Period Performance Summary (Using Tuned Parameters) --- \n")
# DB performance with its tuned parameters
res_DB_lo <- testMyStrategy(xt_DB, pnDB, rvtDB, longshort = 0, tcost = 0.001,
                            bb_n = db_tuned_bb_n, bb_sd = db_tuned_bb_sd, rvt_quantile = db_tuned_rvt_quantile)
res_DB_ls <- testMyStrategy(xt_DB, pnDB, rvtDB, longshort = -1, tcost = 0.001,
                            bb_n = db_tuned_bb_n, bb_sd = db_tuned_bb_sd, rvt_quantile = db_tuned_rvt_quantile)

# HSBC performance with its tuned parameters
res_H_lo  <- testMyStrategy(xt_H, pnH, rvtH, longshort = 0, tcost = 0.001,
                            bb_n = hsbc_tuned_bb_n, bb_sd = hsbc_tuned_bb_sd, rvt_quantile = hsbc_tuned_rvt_quantile)
res_H_ls  <- testMyStrategy(xt_H, pnH, rvtH, longshort = -1, tcost = 0.001,
                            bb_n = hsbc_tuned_bb_n, bb_sd = hsbc_tuned_bb_sd, rvt_quantile = hsbc_tuned_rvt_quantile)

plot_cumulative_returns <- function(price_xts, pnlog, rvt, bb_n, bb_sd, rvt_quantile, file_name) {
  returns <- dailyReturn(price_xts, type = "arithmetic")
  bb_res <- BBands(pnlog, n = bb_n, sd = bb_sd)
  rvt_thr <- quantile(rvt, rvt_quantile, na.rm = TRUE)

  # Long-only: 1 or 0
  sig_long <- Lag(ifelse(pnlog > bb_res$up & rvt > rvt_thr, 1, 0), 1)

  # Long-short: 1, -1, or 0
  sig_ls <- Lag(ifelse(pnlog > bb_res$up & rvt > rvt_thr, 1,
                       ifelse(pnlog < bb_res$dn & rvt > rvt_thr, -1, 0)), 1)

  ret_long <- returns * sig_long
  ret_ls <- returns * sig_ls

  combined <- na.omit(merge(returns, ret_long, ret_ls))
  colnames(combined) <- c("BH", "Long", "LS")

  cum_returns <- cumprod(1 + combined)
  df <- fortify.zoo(cum_returns)
  df_long <- pivot_longer(df, -Index, names_to = "Strategy", values_to = "CumulativeReturn")

  p <- ggplot(df_long, aes(x = Index, y = CumulativeReturn, color = Strategy)) +
    geom_line(size = 1) +
    labs(x = "Date", y = "Cumulative Return", color = NULL) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black"),
      axis.text = element_text(color = "black"),
      legend.position = "top"
    )

  ggsave(file_name, plot = p, width = 7, height = 4.5)
}

# DB plot
plot_cumulative_returns(
  price_xts = xt_DB,
  pnlog = pnDB,
  rvt = rvtDB,
  bb_n = db_tuned_bb_n,
  bb_sd = db_tuned_bb_sd,
  rvt_quantile = db_tuned_rvt_quantile,
  file_name = "db_cumulative_returns.pdf"
)

# HSBC plot
plot_cumulative_returns(
  price_xts = xt_H,
  pnlog = pnH,
  rvt = rvtH,
  bb_n = hsbc_tuned_bb_n,
  bb_sd = hsbc_tuned_bb_sd,
  rvt_quantile = hsbc_tuned_rvt_quantile,
  file_name = "hsbc_cumulative_returns.pdf"
)

## 5) Rolling-Window Performance Analysis: Robustness Testing
cat("\n--- Rolling-Window Performance (254-Day Periods, Using Tuned Parameters) --- \n")
# DB rolling performance with its tuned parameters
rol_DB_254_lo <- t(RollingMyStrategy(xt_DB, pnDB, rvtDB, 0, 0.001, 254,
                                     bb_n = db_tuned_bb_n, bb_sd = db_tuned_bb_sd, rvt_quantile = db_tuned_rvt_quantile))
rol_DB_254_ls <- t(RollingMyStrategy(xt_DB, pnDB, rvtDB, -1, 0.001, 254,
                                     bb_n = db_tuned_bb_n, bb_sd = db_tuned_bb_sd, rvt_quantile = db_tuned_rvt_quantile))

# HSBC rolling performance with its tuned parameters
rol_H_254_lo  <- t(RollingMyStrategy(xt_H, pnH, rvtH, 0, 0.001, 254,
                                     bb_n = hsbc_tuned_bb_n, bb_sd = hsbc_tuned_bb_sd, rvt_quantile = hsbc_tuned_rvt_quantile))
rol_H_254_ls  <- t(RollingMyStrategy(xt_H, pnH, rvtH, -1, 0.001, 254,
                                     bb_n = hsbc_tuned_bb_n, bb_sd = hsbc_tuned_bb_sd, rvt_quantile = hsbc_tuned_rvt_quantile))

cat("\n--- Rolling-Window Performance (127-Day Periods, Using Tuned Parameters) --- \n")
# DB rolling performance with its tuned parameters
rol_DB_127_lo <- t(RollingMyStrategy(xt_DB, pnDB, rvtDB, 0, 0.001, 127,
                                     bb_n = db_tuned_bb_n, bb_sd = db_tuned_bb_sd, rvt_quantile = db_tuned_rvt_quantile))
rol_DB_127_ls <- t(RollingMyStrategy(xt_DB, pnDB, rvtDB, -1, 0.001, 127,
                                     bb_n = db_tuned_bb_n, bb_sd = db_tuned_bb_sd, rvt_quantile = db_tuned_rvt_quantile))

# HSBC rolling performance with its tuned parameters
rol_H_127_lo  <- t(RollingMyStrategy(xt_H, pnH, rvtH, 0, 0.001, 127,
                                     bb_n = hsbc_tuned_bb_n, bb_sd = hsbc_tuned_bb_sd, rvt_quantile = hsbc_tuned_rvt_quantile))
rol_H_127_ls  <- t(RollingMyStrategy(xt_H, pnH, rvtH, -1, 0.001, 127,
                                     bb_n = hsbc_tuned_bb_n, bb_sd = hsbc_tuned_bb_sd, rvt_quantile = hsbc_tuned_rvt_quantile))

######################################################################
## Results Presentation: Consolidated Performance Tables
######################################################################

makeTable <- function(h_lo, h_ls, db_lo, db_ls, title) {
  HSBC_Long      <- h_lo[ , "Me"]
  HSBC_LS        <- h_ls[ , "Me"]
  HSBC_BuyHold   <- h_lo[ , "BH"]

  DB_Long        <- db_lo[ , "Me"]
  DB_LS          <- db_ls[ , "Me"]
  DB_BuyHold     <- db_lo[ , "BH"]

  tab <- cbind(
    HSBC_Long, HSBC_LS, HSBC_BuyHold,
    DB_Long,   DB_LS,   DB_BuyHold
  )
  colnames(tab) <- c(
    "HSBC Long", "HSBC LS", "HSBC BuyHold",
    "DB Long",   "DB LS",   "DB BuyHold"
  )

  cat("\n===== ", title, " =====", "\n", sep = "")
  print(round(tab, 4))
  cat("\n")
}

# --- 1) Full Window Performance Results ---
makeTable(
  res_H_lo,  res_H_ls,
  res_DB_lo, res_DB_ls,
  "Full Window Performance Results (Using Tuned Parameters)"
)

# ===== Full Window Performance Results (Using Tuned Parameters) =====
#   HSBC Long  HSBC LS HSBC BuyHold  DB Long    DB LS DB BuyHold
# Cumulative Return          0.0295   1.0535      -0.4905   0.0573   0.7035    -0.5897
# Annual Return              0.0123   0.3528      -0.2466   0.0238   0.2521    -0.3135
# Annualized Sharpe Ratio    0.2684   1.5395      -1.0739   0.9300   0.5579    -0.6933
# Win %                      0.6341   0.5276       0.4908   0.8333   0.5249     0.4820
# Annualized Volatility      0.0458   0.2292       0.2296   0.0256   0.4520     0.4521
# Maximum Drawdown          -0.0370  -0.1695      -0.5213  -0.0076  -0.4593    -0.7146
# Max Length Drawdown      138.0000 197.0000     592.0000 219.0000 144.0000   587.0000
# n.trades                  79.0000  79.0000       1.0000  13.0000  13.0000     1.0000

# --- 2) Rolling 254-Day Performance Results (Annual View) ---
makeTable(
  rol_H_254_lo, rol_H_254_ls,
  rol_DB_254_lo, rol_DB_254_ls,
  "Rolling 254-Day Performance Results (Using Tuned Parameters)"
)

# ===== Rolling 254-Day Performance Results (Using Tuned Parameters) =====
#   HSBC Long  HSBC LS HSBC BuyHold  DB Long    DB LS DB BuyHold
# Cumulative Return          0.0128   0.1943      -0.1212   0.0062   0.2698    -0.2715
# Annual Return              0.0129   0.1951      -0.1216   0.0062   0.2710    -0.2724
# Annualized Sharpe Ratio    0.3714   0.9744      -0.6050     -Inf   0.7396    -0.7124
# Win %                      0.6548   0.5247       0.4990   0.8601   0.5231     0.4812
# Annualized Volatility      0.0363   0.1830       0.1834   0.0262   0.3950     0.3951
# Maximum Drawdown          -0.0205  -0.1387      -0.2279  -0.0101  -0.2767    -0.4388
# Max Length Drawdown      104.5661 129.1580     203.9770 237.2270 116.3190   223.8994
# n.trades                  37.1753  37.1753       1.0000   4.5431   4.5431     1.0000

# --- 3) Rolling 127-Day Performance Results (Semi-Annual View) ---
makeTable(
  rol_H_127_lo, rol_H_127_ls,
  rol_DB_127_lo, rol_DB_127_ls,
  "Rolling 127-Day Performance Results (Using Tuned Parameters)"
)

# ===== Rolling 127-Day Performance Results (Using Tuned Parameters) =====
#   HSBC Long HSBC LS HSBC BuyHold  DB Long   DB LS DB BuyHold
# Cumulative Return          0.0077  0.1150      -0.0783   0.0034  0.1155    -0.1391
# Annual Return              0.0162  0.2615      -0.1431   0.0073  0.2873    -0.2354
# Annualized Sharpe Ratio    0.4505  1.3422      -0.6940     -Inf  0.8322    -0.6312
# Win %                      0.6758  0.5280       0.4977   0.8500  0.5176     0.4869
# Annualized Volatility      0.0374  0.1852       0.1858   0.0190  0.3945     0.3946
# Maximum Drawdown          -0.0141 -0.0983      -0.1606  -0.0038 -0.2420    -0.3150
# Max Length Drawdown       77.5221 67.1326      98.2337 123.1916 70.9368    97.3789
# n.trades                  18.7747 18.7747       1.0000   2.4695  2.4695     1.0000

## Signal Overlay on Price Charts

# DB: Strategy signal (buy/sell)
bb_DB <- BBands(pnDB, n = db_tuned_bb_n, sd = db_tuned_bb_sd)
rvt_thr_DB <- quantile(rvtDB, db_tuned_rvt_quantile, na.rm = TRUE)
sig_DB_full <- Lag(ifelse(pnDB > bb_DB$up & rvtDB > rvt_thr_DB, 1,
                          ifelse(pnDB < bb_DB$dn & rvtDB > rvt_thr_DB, -1, 0)), 1)

pdf("db_signals_plot.pdf", width = 7, height = 4.5)
plot.zoo(xt_DB, main = "", col = "gray60", ylab = "Price", xlab = "Date")
points(index(xt_DB)[which(sig_DB_full == 1)], xt_DB[which(sig_DB_full == 1)], col = "green", pch = 16)
points(index(xt_DB)[which(sig_DB_full == -1)], xt_DB[which(sig_DB_full == -1)], col = "red", pch = 17)
legend("topright",
       legend = c("Buy Signal", "Sell Signal"),
       col = c("green", "red"),
       pch = c(16, 17),
       bty = "n")
dev.off()

# HSBC: Strategy signal (buy/sell)
bb_H <- BBands(pnH, n = hsbc_tuned_bb_n, sd = hsbc_tuned_bb_sd)
rvt_thr_H <- quantile(rvtH, hsbc_tuned_rvt_quantile, na.rm = TRUE)
sig_H_full <- Lag(ifelse(pnH > bb_H$up & rvtH > rvt_thr_H, 1,
                         ifelse(pnH < bb_H$dn & rvtH > rvt_thr_H, -1, 0)), 1)

pdf("hsbc_signals_plot.pdf", width = 7, height = 4.5)
plot.zoo(xt_H, main = "", col = "gray60", ylab = "Price", xlab = "Date")
points(index(xt_H)[which(sig_H_full == 1)], xt_H[which(sig_H_full == 1)], col = "green", pch = 16)
points(index(xt_H)[which(sig_H_full == -1)], xt_H[which(sig_H_full == -1)], col = "red", pch = 17)
legend("topright",
       legend = c("Buy Signal", "Sell Signal"),
       col = c("green", "red"),
       pch = c(16, 17),
       bty = "n")
dev.off()


######################################################################
## APPENDIX: Parameter Tuning for Full Window, Long-Only Strategy
## Objective: Maximize Cumulative Return for each asset.
## Note: This process is computationally intensive due to fine-grained ranges.
######################################################################

# Define precise parameter ranges for tuning
bb_n_values <-  seq(10, 30, by = 5)
bb_sd_values <- seq(0.5, 3.0, by = 0.5)
rvt_q_values <- seq(0.5, 0.9, by = 0.1)

# Initialize storage for tuning results for DB and HSBC
tuning_results_db <- list()
tuning_results_hsbc <- list()
counter <- 1

# Grid Search Execution
for (n_val in bb_n_values) {
  for (sd_val in bb_sd_values) {
    for (q_val in rvt_q_values) {

      # Test for DB
      perf_db <- testMyStrategy(
        myStock = xt_DB,
        ev_pnlog = pnDB,
        ev_rvt = rvtDB,
        longshort = 0, # Fixed: Long-Only strategy
        tcost = 0.001,
        bb_n = n_val,
        bb_sd = sd_val,
        rvt_quantile = q_val
      )
      # Store parameter values and the Cumulative Return for DB.
      tuning_results_db[[counter]] <- c(
        BB_n = n_val,
        BB_sd = sd_val,
        RVT_Quantile = q_val,
        Cumulative_Return = perf_db["Cumulative Return", "Me"]
      )

      # Test for HSBC
      perf_hsbc <- testMyStrategy(
        myStock = xt_H,
        ev_pnlog = pnH,
        ev_rvt = rvtH,
        longshort = 0, # Fixed: Long-Only strategy
        tcost = 0.001,
        bb_n = n_val,
        bb_sd = sd_val,
        rvt_quantile = q_val
      )
      # Store parameter values and the Cumulative Return for HSBC.
      tuning_results_hsbc[[counter]] <- c(
        BB_n = n_val,
        BB_sd = sd_val,
        RVT_Quantile = q_val,
        Cumulative_Return = perf_hsbc["Cumulative Return", "Me"]
      )

      counter <- counter + 1
    }
  }
}

# Consolidate results into data frames for easier analysis
tuning_df_db <- as.data.frame(do.call(rbind, tuning_results_db))
tuning_df_hsbc <- as.data.frame(do.call(rbind, tuning_results_hsbc))

# Identify the best parameters based on maximum Cumulative Return for each asset
# Corrected: Use max() with na.rm=TRUE, then which() to find the index of that max value.
# [1] is added to select the first combination if multiple combinations yield the same maximum return.
max_cum_ret_db <- max(tuning_df_db$Cumulative_Return, na.rm = TRUE)
best_params_db <- tuning_df_db[which(tuning_df_db$Cumulative_Return == max_cum_ret_db)[1], ]

max_cum_ret_hsbc <- max(tuning_df_hsbc$Cumulative_Return, na.rm = TRUE)
best_params_hsbc <- tuning_df_hsbc[which(tuning_df_hsbc$Cumulative_Return == max_cum_ret_hsbc)[1], ]

# Present Tuning Results
cat("\n######################################################################\n")
cat("## APPENDIX: Parameter Tuning Results (Full Window, Long-Only)      ##\n")
cat("## Optimization Objective: Maximize Cumulative Return               ##\n")
cat("######################################################################\n")

cat("\nOptimal Parameters for Deutsche Bank (DB):\n")
print(round(best_params_db, 4))
cat("\n")

# Optimal Parameters for Deutsche Bank (DB):
# BB_n BB_sd RVT_Quantile Cumulative_Return
# 51   15   2.5          0.5            0.0138

cat("Optimal Parameters for HSBC Holdings (HSBC):\n")
print(round(best_params_hsbc, 4))
cat("\n")

# Optimal Parameters for HSBC Holdings (HSBC):
# BB_n BB_sd RVT_Quantile Cumulative_Return
# 71   20   1.5          0.5            0.0266

cat("Full Tuning Grid Results (DB) - Sorted by Cumulative Return:\n")
# Sort the entire tuning data frame by Cumulative Return in descending order, putting NAs last
print(round(tuning_df_db[order(-tuning_df_db$Cumulative_Return, na.last = TRUE), ], 4))
cat("\n")

# Full Tuning Grid Results (DB) - Sorted by Cumulative Return:
# BB_n BB_sd RVT_Quantile Cumulative_Return
# 51    15   2.5          0.5            0.0138
# 52    15   2.5          0.6            0.0138
# 53    15   2.5          0.7            0.0138
# 73    20   1.5          0.7            0.0138
# 76    20   2.0          0.5            0.0138
# 77    20   2.0          0.6            0.0138
# 78    20   2.0          0.7            0.0138
# 81    20   2.5          0.5            0.0138
# 82    20   2.5          0.6            0.0138
# 83    20   2.5          0.7            0.0138
# 131   30   1.5          0.5            0.0138
# 132   30   1.5          0.6            0.0138
# 133   30   1.5          0.7            0.0138
# 16    10   2.0          0.5            0.0128
# 17    10   2.0          0.6            0.0128
# 18    10   2.0          0.7            0.0128
# 43    15   1.5          0.7            0.0128
# 46    15   2.0          0.5            0.0128
# 47    15   2.0          0.6            0.0128
# 48    15   2.0          0.7            0.0128
# 103   25   1.5          0.7            0.0128
# 13    10   1.5          0.7            0.0031
# 98    25   1.0          0.7            0.0031
# 14    10   1.5          0.8           -0.0010
# 15    10   1.5          0.9           -0.0010
# 19    10   2.0          0.8           -0.0010
# 20    10   2.0          0.9           -0.0010
# 21    10   2.5          0.5           -0.0010
# 22    10   2.5          0.6           -0.0010
# 23    10   2.5          0.7           -0.0010
# 24    10   2.5          0.8           -0.0010
# 25    10   2.5          0.9           -0.0010
# 26    10   3.0          0.5           -0.0010
# 27    10   3.0          0.6           -0.0010
# 28    10   3.0          0.7           -0.0010
# 29    10   3.0          0.8           -0.0010
# 30    10   3.0          0.9           -0.0010
# 44    15   1.5          0.8           -0.0010
# 45    15   1.5          0.9           -0.0010
# 49    15   2.0          0.8           -0.0010
# 50    15   2.0          0.9           -0.0010
# 54    15   2.5          0.8           -0.0010
# 55    15   2.5          0.9           -0.0010
# 56    15   3.0          0.5           -0.0010
# 57    15   3.0          0.6           -0.0010
# 58    15   3.0          0.7           -0.0010
# 59    15   3.0          0.8           -0.0010
# 60    15   3.0          0.9           -0.0010
# 70    20   1.0          0.9           -0.0010
# 74    20   1.5          0.8           -0.0010
# 75    20   1.5          0.9           -0.0010
# 79    20   2.0          0.8           -0.0010
# 80    20   2.0          0.9           -0.0010
# 84    20   2.5          0.8           -0.0010
# 85    20   2.5          0.9           -0.0010
# 86    20   3.0          0.5           -0.0010
# 87    20   3.0          0.6           -0.0010
# 88    20   3.0          0.7           -0.0010
# 89    20   3.0          0.8           -0.0010
# 90    20   3.0          0.9           -0.0010
# 99    25   1.0          0.8           -0.0010
# 100   25   1.0          0.9           -0.0010
# 104   25   1.5          0.8           -0.0010
# 105   25   1.5          0.9           -0.0010
# 106   25   2.0          0.5           -0.0010
# 107   25   2.0          0.6           -0.0010
# 108   25   2.0          0.7           -0.0010
# 109   25   2.0          0.8           -0.0010
# 110   25   2.0          0.9           -0.0010
# 111   25   2.5          0.5           -0.0010
# 112   25   2.5          0.6           -0.0010
# 113   25   2.5          0.7           -0.0010
# 114   25   2.5          0.8           -0.0010
# 115   25   2.5          0.9           -0.0010
# 116   25   3.0          0.5           -0.0010
# 117   25   3.0          0.6           -0.0010
# 118   25   3.0          0.7           -0.0010
# 119   25   3.0          0.8           -0.0010
# 120   25   3.0          0.9           -0.0010
# 130   30   1.0          0.9           -0.0010
# 134   30   1.5          0.8           -0.0010
# 135   30   1.5          0.9           -0.0010
# 136   30   2.0          0.5           -0.0010
# 137   30   2.0          0.6           -0.0010
# 138   30   2.0          0.7           -0.0010
# 139   30   2.0          0.8           -0.0010
# 140   30   2.0          0.9           -0.0010
# 141   30   2.5          0.5           -0.0010
# 142   30   2.5          0.6           -0.0010
# 143   30   2.5          0.7           -0.0010
# 144   30   2.5          0.8           -0.0010
# 145   30   2.5          0.9           -0.0010
# 146   30   3.0          0.5           -0.0010
# 147   30   3.0          0.6           -0.0010
# 148   30   3.0          0.7           -0.0010
# 149   30   3.0          0.8           -0.0010
# 150   30   3.0          0.9           -0.0010
# 71    20   1.5          0.5           -0.0016
# 72    20   1.5          0.6           -0.0016
# 41    15   1.5          0.5           -0.0027
# 42    15   1.5          0.6           -0.0027
# 101   25   1.5          0.5           -0.0027
# 102   25   1.5          0.6           -0.0027
# 10    10   1.0          0.9           -0.0079
# 68    20   1.0          0.7           -0.0119
# 11    10   1.5          0.5           -0.0123
# 12    10   1.5          0.6           -0.0123
# 128   30   1.0          0.7           -0.0242
# 69    20   1.0          0.8           -0.0252
# 129   30   1.0          0.8           -0.0280
# 40    15   1.0          0.9           -0.0338
# 96    25   1.0          0.5           -0.0390
# 39    15   1.0          0.8           -0.0452
# 38    15   1.0          0.7           -0.0470
# 97    25   1.0          0.6           -0.0540
# 9     10   1.0          0.8           -0.0615
# 66    20   1.0          0.5           -0.0616
# 126   30   1.0          0.5           -0.0634
# 67    20   1.0          0.6           -0.0782
# 35    15   0.5          0.9           -0.0797
# 127   30   1.0          0.6           -0.0800
# 36    15   1.0          0.5           -0.0888
# 8     10   1.0          0.7           -0.0895
# 65    20   0.5          0.9           -0.0917
# 95    25   0.5          0.9           -0.0971
# 37    15   1.0          0.6           -0.0997
# 5     10   0.5          0.9           -0.1076
# 6     10   1.0          0.5           -0.1283
# 125   30   0.5          0.9           -0.1305
# 7     10   1.0          0.6           -0.1386
# 34    15   0.5          0.8           -0.2153
# 94    25   0.5          0.8           -0.2205
# 64    20   0.5          0.8           -0.2255
# 4     10   0.5          0.8           -0.2556
# 124   30   0.5          0.8           -0.2659
# 2     10   0.5          0.6           -0.2987
# 3     10   0.5          0.7           -0.3059
# 1     10   0.5          0.5           -0.3346
# 63    20   0.5          0.7           -0.3361
# 123   30   0.5          0.7           -0.3467
# 93    25   0.5          0.7           -0.3496
# 33    15   0.5          0.7           -0.3563
# 121   30   0.5          0.5           -0.3809
# 32    15   0.5          0.6           -0.3893
# 122   30   0.5          0.6           -0.4046
# 91    25   0.5          0.5           -0.4049
# 62    20   0.5          0.6           -0.4083
# 92    25   0.5          0.6           -0.4099
# 61    20   0.5          0.5           -0.4318
# 31    15   0.5          0.5           -0.4556

cat("Full Tuning Grid Results (HSBC) - Sorted by Cumulative Return:\n")
# Sort the entire tuning data frame by Cumulative Return in descending order, putting NAs last
print(round(tuning_df_hsbc[order(-tuning_df_hsbc$Cumulative_Return, na.last = TRUE), ], 4))
cat("\n")

# Full Tuning Grid Results (HSBC) - Sorted by Cumulative Return:
#   BB_n BB_sd RVT_Quantile Cumulative_Return
# 71    20   1.5          0.5            0.0266
# 11    10   1.5          0.5            0.0259
# 136   30   2.0          0.5            0.0210
# 137   30   2.0          0.6            0.0210
# 138   30   2.0          0.7            0.0210
# 98    25   1.0          0.7            0.0159
# 72    20   1.5          0.6            0.0157
# 12    10   1.5          0.6            0.0149
# 101   25   1.5          0.5            0.0148
# 126   30   1.0          0.5            0.0148
# 16    10   2.0          0.5            0.0137
# 17    10   2.0          0.6            0.0137
# 18    10   2.0          0.7            0.0137
# 46    15   2.0          0.5            0.0137
# 47    15   2.0          0.6            0.0137
# 48    15   2.0          0.7            0.0137
# 51    15   2.5          0.5            0.0137
# 52    15   2.5          0.6            0.0137
# 53    15   2.5          0.7            0.0137
# 81    20   2.5          0.5            0.0137
# 82    20   2.5          0.6            0.0137
# 83    20   2.5          0.7            0.0137
# 111   25   2.5          0.5            0.0137
# 112   25   2.5          0.6            0.0137
# 113   25   2.5          0.7            0.0137
# 141   30   2.5          0.5            0.0137
# 142   30   2.5          0.6            0.0137
# 143   30   2.5          0.7            0.0137
# 127   30   1.0          0.6            0.0126
# 73    20   1.5          0.7            0.0087
# 131   30   1.5          0.5            0.0085
# 13    10   1.5          0.7            0.0079
# 97    25   1.0          0.6            0.0076
# 128   30   1.0          0.7            0.0054
# 96    25   1.0          0.5            0.0054
# 44    15   1.5          0.8            0.0044
# 74    20   1.5          0.8            0.0044
# 104   25   1.5          0.8            0.0044
# 102   25   1.5          0.6            0.0040
# 41    15   1.5          0.5            0.0038
# 99    25   1.0          0.8            0.0015
# 14    10   1.5          0.8           -0.0005
# 15    10   1.5          0.9           -0.0010
# 19    10   2.0          0.8           -0.0010
# 20    10   2.0          0.9           -0.0010
# 21    10   2.5          0.5           -0.0010
# 22    10   2.5          0.6           -0.0010
# 23    10   2.5          0.7           -0.0010
# 24    10   2.5          0.8           -0.0010
# 25    10   2.5          0.9           -0.0010
# 26    10   3.0          0.5           -0.0010
# 27    10   3.0          0.6           -0.0010
# 28    10   3.0          0.7           -0.0010
# 29    10   3.0          0.8           -0.0010
# 30    10   3.0          0.9           -0.0010
# 45    15   1.5          0.9           -0.0010
# 49    15   2.0          0.8           -0.0010
# 50    15   2.0          0.9           -0.0010
# 54    15   2.5          0.8           -0.0010
# 55    15   2.5          0.9           -0.0010
# 56    15   3.0          0.5           -0.0010
# 57    15   3.0          0.6           -0.0010
# 58    15   3.0          0.7           -0.0010
# 59    15   3.0          0.8           -0.0010
# 60    15   3.0          0.9           -0.0010
# 75    20   1.5          0.9           -0.0010
# 79    20   2.0          0.8           -0.0010
# 80    20   2.0          0.9           -0.0010
# 84    20   2.5          0.8           -0.0010
# 85    20   2.5          0.9           -0.0010
# 86    20   3.0          0.5           -0.0010
# 87    20   3.0          0.6           -0.0010
# 88    20   3.0          0.7           -0.0010
# 89    20   3.0          0.8           -0.0010
# 90    20   3.0          0.9           -0.0010
# 100   25   1.0          0.9           -0.0010
# 105   25   1.5          0.9           -0.0010
# 109   25   2.0          0.8           -0.0010
# 110   25   2.0          0.9           -0.0010
# 114   25   2.5          0.8           -0.0010
# 115   25   2.5          0.9           -0.0010
# 116   25   3.0          0.5           -0.0010
# 117   25   3.0          0.6           -0.0010
# 118   25   3.0          0.7           -0.0010
# 119   25   3.0          0.8           -0.0010
# 120   25   3.0          0.9           -0.0010
# 130   30   1.0          0.9           -0.0010
# 135   30   1.5          0.9           -0.0010
# 139   30   2.0          0.8           -0.0010
# 140   30   2.0          0.9           -0.0010
# 144   30   2.5          0.8           -0.0010
# 145   30   2.5          0.9           -0.0010
# 146   30   3.0          0.5           -0.0010
# 147   30   3.0          0.6           -0.0010
# 148   30   3.0          0.7           -0.0010
# 149   30   3.0          0.8           -0.0010
# 150   30   3.0          0.9           -0.0010
# 106   25   2.0          0.5           -0.0015
# 107   25   2.0          0.6           -0.0015
# 134   30   1.5          0.8           -0.0018
# 132   30   1.5          0.6           -0.0022
# 103   25   1.5          0.7           -0.0029
# 129   30   1.0          0.8           -0.0037
# 42    15   1.5          0.6           -0.0070
# 76    20   2.0          0.5           -0.0083
# 77    20   2.0          0.6           -0.0083
# 78    20   2.0          0.7           -0.0083
# 108   25   2.0          0.7           -0.0083
# 133   30   1.5          0.7           -0.0090
# 43    15   1.5          0.7           -0.0138
# 4     10   0.5          0.8           -0.0614
# 94    25   0.5          0.8           -0.0625
# 95    25   0.5          0.9           -0.0659
# 6     10   1.0          0.5           -0.0727
# 34    15   0.5          0.8           -0.0759
# 64    20   0.5          0.8           -0.0759
# 124   30   0.5          0.8           -0.0759
# 69    20   1.0          0.8           -0.0778
# 10    10   1.0          0.9           -0.0794
# 40    15   1.0          0.9           -0.0794
# 70    20   1.0          0.9           -0.0794
# 8     10   1.0          0.7           -0.0825
# 9     10   1.0          0.8           -0.0827
# 7     10   1.0          0.6           -0.0856
# 37    15   1.0          0.6           -0.0856
# 39    15   1.0          0.8           -0.0873
# 35    15   0.5          0.9           -0.0883
# 65    20   0.5          0.9           -0.0883
# 125   30   0.5          0.9           -0.0883
# 36    15   1.0          0.5           -0.0884
# 38    15   1.0          0.7           -0.0894
# 66    20   1.0          0.5           -0.0923
# 67    20   1.0          0.6           -0.0937
# 68    20   1.0          0.7           -0.0980
# 5     10   0.5          0.9           -0.1063
# 123   30   0.5          0.7           -0.1178
# 3     10   0.5          0.7           -0.1180
# 93    25   0.5          0.7           -0.1192
# 63    20   0.5          0.7           -0.1324
# 122   30   0.5          0.6           -0.1586
# 33    15   0.5          0.7           -0.1635
# 2     10   0.5          0.6           -0.1661
# 92    25   0.5          0.6           -0.1661
# 62    20   0.5          0.6           -0.1816
# 121   30   0.5          0.5           -0.1852
# 91    25   0.5          0.5           -0.1921
# 32    15   0.5          0.6           -0.1962
# 61    20   0.5          0.5           -0.2114
# 1     10   0.5          0.5           -0.2278
# 31    15   0.5          0.5           -0.2302

######################################################################
## End of Script
######################################################################
