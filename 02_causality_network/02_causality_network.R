##### 02_causality_network.R ################################################

##### Set Up ################################################################
rm(list=ls())

# Packages
library(xts)
library(quantmod)
library(lmtest)
library(TTR)
library(igraph)
library(tidyverse)
library(tidyquant)
library(timetk)
library(vars)
library(bruceR)
library(stargazer)

# Index to country mapping in original order
index_to_country <- c(
  BSESN = "India",
  BVSP  = "Brazil",
  FTSE  = "UK",
  GDAXI = "Germany",
  GSPC  = "USA",
  HSCE  = "China",
  IBEX  = "Spain",
  JKSE  = "Indonesia",
  MXX   = "Mexico",
  N225  = "Japan",
  TWII  = "Taiwan",
  VIX   = "VIX",
  VLIC  = "VLIC"
)

# Import
data_env <- readRDS("WorldMarkts99_20.RDS")
markets <- names(index_to_country)
country_names <- unname(index_to_country)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
##### Treatment ################################################################
# Extract closes
adjusted_prices <- markets %>%
  set_names(country_names) %>%
  imap(~ get(.x, envir = data_env) %>%
         Ad() %>%
         magrittr::set_colnames(.y)) %>%
  reduce(~ merge(.x, .y, join = "inner")) %>%
  na.omit()

# convert to tibble with date column
price_tbl <- adjusted_prices %>%
  tk_tbl(preserve_index = TRUE, rename_index = "date")

# weekly arithmetic returns
rets_weekly <- price_tbl %>%
  pivot_longer(-date, names_to = "symbol", values_to = "price") %>%
  group_by(symbol) %>%
  tq_transmute(
    select     = price,
    mutate_fun = periodReturn,
    period     = "weekly",
    type       = "arithmetic",
    col_rename = "return"
  ) %>%
  pivot_wider(names_from = symbol, values_from = return) %>%
  tk_xts(date_var = "date")

# monthly arithmetic returns
rets_monthly <- price_tbl %>%
  pivot_longer(-date, names_to = "symbol", values_to = "price") %>%
  group_by(symbol) %>%
  tq_transmute(
    select     = price,
    mutate_fun = periodReturn,
    period     = "monthly",
    type       = "arithmetic",
    col_rename = "return"
  ) %>%
  pivot_wider(names_from = symbol, values_from = return) %>%
  tk_xts(date_var = "date")

# set decay lambda and calculate smoothing K=1-lambda
lambda    <- 0.94
smoothing <- 1 - lambda

# COMPUTE WEEKLY VOLATILITY
vol_weekly <- rets_weekly
for (sym in country_names) {
  vol_weekly[, sym] <- sqrt(EMA(rets_weekly[, sym]^2, n = 1, ratio = smoothing))
}
vol_weekly <- na.omit(vol_weekly)

# COMPUTE MONTHLY VOLATILITY
vol_monthly <- rets_monthly
for (sym in country_names) {
  vol_monthly[, sym] <- sqrt(EMA(rets_monthly[, sym]^2, n = 1, ratio = smoothing))
}
vol_monthly <- na.omit(vol_monthly)

# Subset returns to our analysis window
epoch = "2005-06-01/2009-06-30"
rets_weekly  <- rets_weekly[epoch]
rets_monthly <- rets_monthly[epoch]
vol_weekly <- vol_weekly[epoch]
vol_monthly <- vol_monthly[epoch]

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
##### Analysis ################################################################
# build tidy causality matrix with 3-vector entries like (1,0,1)
get_causality_matrix <- function(data, lags = 3, alpha = 0.05) {
  vars <- colnames(data)

  expand_grid(cause = vars, effect = vars) %>%
    filter(cause != effect) %>%
    mutate(cell = pmap_chr(list(cause, effect), function(c, e) {
      result <- map_int(1:lags, function(lag) {
        gt <- try(grangertest(data[, e] ~ data[, c], order = lag), silent = TRUE)
        if (inherits(gt, "try-error")) return(0)
        p <- suppressWarnings(as.numeric(gt$`Pr(>F)`[2]))
        if (!is.na(p) && p < alpha) 1 else 0
      })
      paste0("(", paste(result, collapse = ","), ")")
    })) %>%
    pivot_wider(names_from = effect, values_from = cell) %>%
    column_to_rownames("cause") %>%
    .[country_names, country_names]  # maintain original order
}

gc_rets_wk  <- get_causality_matrix(rets_weekly)
gc_rets_mo  <- get_causality_matrix(rets_monthly)
gc_vol_wk   <- get_causality_matrix(vol_weekly)
gc_vol_mo   <- get_causality_matrix(vol_monthly)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
##### FIRST ANALYSIS: Among which countries there is true causality? ####
# Convert matrix with entries like "(1,0,1)" into binary 1/0
binary_matrix <- function(mat) {
  mat %>%
    as.data.frame() %>%
    mutate(across(everything(), ~ as.integer(grepl("1", .)))) %>%
    as.matrix()
}

# 1. Binary matrix summary: country → country (1 if any lag causes effect, 0 otherwise)
print_binary_causality_matrix <- function(mat, title) {
  bin <- binary_matrix(mat)
  cat("\n\n###", title, "\n")
  print(as.data.frame(bin))
}

print_binary_causality_matrix(gc_rets_wk, "Returns – Weekly")
print_binary_causality_matrix(gc_rets_mo, "Returns – Monthly")
print_binary_causality_matrix(gc_vol_wk,  "Volatility – Weekly")
print_binary_causality_matrix(gc_vol_mo,  "Volatility – Monthly")

# 2. Rank by causes (rows) and being caused (columns)
rank_causal_matrix <- function(mat) {
  bin <- binary_matrix(mat)
  tibble(
    Country = rownames(bin),
    Causes = rowSums(bin),
    Caused = colSums(bin)
  ) %>% arrange(desc(Causes))
}

cat("\n\n### Rankings Within Each Metric\n")
cat("\n> Returns – Weekly\n"); print(rank_causal_matrix(gc_rets_wk))
cat("\n> Returns – Monthly\n"); print(rank_causal_matrix(gc_rets_mo))
cat("\n> Volatility – Weekly\n"); print(rank_causal_matrix(gc_vol_wk))
cat("\n> Volatility – Monthly\n"); print(rank_causal_matrix(gc_vol_mo))


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
##### SECOND ANALYSIS: Among which countries there is contemporaneous correlation? ####

# Set threshold
corr_threshold <- 0.6

# Function: convert correlation matrix into binary 0/1 (1 if |corr| > threshold)
get_binary_corr_matrix <- function(cor_matrix, threshold = corr_threshold) {
  bin <- abs(cor_matrix) > threshold
  diag(bin) <- FALSE  # exclude self-correlation
  storage.mode(bin) <- "numeric"
  bin[rownames(cor_matrix), colnames(cor_matrix)]
}

# Correlation matrices
cor_matrices <- list(
  rets_wk = cor(rets_weekly),
  rets_mo = cor(rets_monthly),
  vol_wk  = cor(vol_weekly),
  vol_mo  = cor(vol_monthly)
)

# 1. Print binary 0/1 correlation matrices
print_binary_corr_matrix <- function(mat, title) {
  cat("\n\n###", title, "\n")
  print(as.data.frame(get_binary_corr_matrix(mat)))
}

print_binary_corr_matrix(cor_matrices$rets_wk, "Returns – Weekly")
print_binary_corr_matrix(cor_matrices$rets_mo, "Returns – Monthly")
print_binary_corr_matrix(cor_matrices$vol_wk,  "Volatility – Weekly")
print_binary_corr_matrix(cor_matrices$vol_mo,  "Volatility – Monthly")

# 2. Rank countries by number of strong correlations (within each matrix)
rank_corr_influence <- function(mat) {
  bin <- get_binary_corr_matrix(mat)
  tibble(
    Country = rownames(bin),
    Strong_Correlations = rowSums(bin)
  ) %>% arrange(desc(Strong_Correlations))
}

# compute for each matrix the per‐country count of |ro|>0.6
corr_counts <- cor_matrices %>%
  imap(~ {
    bin <- get_binary_corr_matrix(.x)
    tibble(
      Country = rownames(bin),
      !!.y    := rowSums(bin)
    )
  }) %>%
  reduce(full_join, by = "Country") %>%
  # rename columns to nice labels
  rename(
    `Returns - Weekly`     = rets_wk,
    `Returns - Monthly`    = rets_mo,
    `Volatility - Weekly`  = vol_wk,
    `Volatility - Monthly` = vol_mo
  ) %>%
  arrange(desc(`Returns - Weekly`))  # or whatever ordering you like

print(corr_counts)


##### THIRD ANALYSIS: PLOT #####################################################
plot_circle_network <- function(causality_matrix, title = "Causality Network") {
  adj_matrix <- causality_matrix %>%
    as.data.frame() %>%
    mutate(across(everything(), ~ as.integer(grepl("1", .)))) %>%
    as.matrix()

  g <- graph_from_adjacency_matrix(adj_matrix, mode = "directed", diag = FALSE)
  layout <- layout_in_circle(g)

  label_pos <- layout * 1.05  # keep labels outside

  plot(
    g,
    layout = layout,
    main = title,
    vertex.size = 10,                 # reduced size here
    vertex.color = "white",
    vertex.frame.color = "black",
    vertex.label = NA,
    edge.arrow.size = 0.4
  )

  text(
    x = label_pos[, 1],
    y = label_pos[, 2],
    labels = V(g)$name,
    cex = 0.9,
    font = 2
  )
}

par(mfrow = c(2, 2))
plot_circle_network(gc_rets_wk,  "Returns – Weekly")
plot_circle_network(gc_rets_mo,  "Returns – Monthly")
plot_circle_network(gc_vol_wk,   "Volatility – Weekly")
plot_circle_network(gc_vol_mo,   "Volatility – Monthly")
par(mfrow = c(1, 1))  # Reset layout

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

### Latex ####
############ 1)
library(knitr)
library(kableExtra)
library(purrr)

gc_list <- list(
  "Returns – Weekly"    = gc_rets_wk,
  "Returns – Monthly"   = gc_rets_mo,
  "Volatility – Weekly" = gc_vol_wk,
  "Volatility – Monthly"= gc_vol_mo
)

imap(gc_list, ~ {
  cat("\n% --------- ", .y, " ---------\n")
  # convert to data.frame and replace any actual R NAs just in case
  df <- as.data.frame(.x, stringsAsFactors = FALSE)
  df[is.na(df)] <- "-"

  # print LaTeX table with na = "-" so kable also handles them
  cat(
    kable(
      df,
      format    = "latex",
      booktabs  = TRUE,
      caption   = paste0("Granger causality — ", .y),
      align     = c("l", rep("c", ncol(df))),
      na        = "-"           # fill any remaining NAs with hyphens
    ) %>%
      kable_styling(
        latex_options = c("hold_position","scale_down"),
        font_size     = 7
      ),
    "\n\n"
  )
})

############ 2)
library(knitr)
library(kableExtra)
library(purrr)

gc_list <- list(
  "Returns – Weekly"    = print_binary_causality_matrix(gc_rets_wk, "Returns – Weekly"),
  "Returns – Monthly"   = print_binary_causality_matrix(gc_rets_mo, "Returns – Monthly"),
  "Volatility – Weekly" = print_binary_causality_matrix(gc_vol_wk,  "Volatility – Weekly"),
  "Volatility – Monthly"= print_binary_causality_matrix(gc_vol_mo,  "Volatility – Monthly")
)

imap(gc_list, ~ {
  cat("\n% --------- ", .y, " ---------\n")
  # convert to data.frame and replace any actual R NAs just in case
  df <- as.data.frame(.x, stringsAsFactors = FALSE)
  df[is.na(df)] <- "-"

  # print LaTeX table with na = "-" so kable also handles them
  cat(
    kable(
      df,
      format    = "latex",
      booktabs  = TRUE,
      caption   = paste0("Granger causality — ", .y),
      align     = c("l", rep("c", ncol(df))),
      na        = "-"           # fill any remaining NAs with hyphens
    ) %>%
      kable_styling(
        latex_options = c("hold_position","scale_down"),
        font_size     = 7
      ),
    "\n\n"
  )
})

############ 3)
# assuming you have `cor_matrices` (rets_wk, rets_mo, vol_wk, vol_mo)
# and the helper `get_binary_corr_matrix()`
# build corr_counts as before...
corr_counts <- cor_matrices %>%
  imap(~ {
    bin <- get_binary_corr_matrix(.x)
    tibble(
      Country = rownames(bin),
      !!.y    := rowSums(bin)
    )
  }) %>%
  reduce(full_join, by = "Country") %>%
  rename(
    `Returns - Weekly`     = rets_wk,
    `Returns - Monthly`    = rets_mo,
    `Volatility - Weekly`  = vol_wk,
    `Volatility - Monthly` = vol_mo
  ) %>%
  arrange(desc(`Returns - Weekly`))

# emit LaTeX with bold headers and bold first column
latex_table <- kable(
  corr_counts,
  format    = "latex",
  booktabs  = TRUE,
  caption   = "Number of strong contemporaneous correlations per country and criterion ($|\\rho|>0.6$)",
  label     = "tab:corr_counts",
  align     = c("c", rep("c", ncol(corr_counts)-1)),
  escape    = TRUE
) %>%
  kable_styling(
    latex_options = c("hold_position","scale_down"),
    font_size     = 8
  ) %>%
  row_spec(0, bold = TRUE) %>%            # bold header row
  column_spec(1, bold = F)             # bold first (Country) column

cat(latex_table)
