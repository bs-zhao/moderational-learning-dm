rm(list = ls())

suppressPackageStartupMessages({
  library(rstan)
  library(dplyr)
})

rstan_options(auto_write = TRUE)
options(mc.cores = max(1, parallel::detectCores() - 1))

TASK <- "RiskC"
MODE <- "context"
FOLD_MIN <- 0L
FOLD_MAX <- 9L
FOLD_TEST <- 0L

stan_file_context <- file.path("stan", paste0(TASK, "_context.stan"))
stan_file_test <- file.path("stan", paste0(TASK, "_test.stan"))

ITER <- 4000L
WARMUP <- 1000L
CHAINS <- 4L
ADAPT_DELTA <- 0.95
SEED_BASE <- 2025L

STAN_REFRESH <- 0L

this_file <- NULL
cmd_args <- commandArgs(trailingOnly = FALSE)
needle <- "--file="
for (i in seq_along(cmd_args)) {
  if (startsWith(cmd_args[[i]], needle)) {
    this_file <- substring(cmd_args[[i]], nchar(needle) + 1)
  }
}
if (!is.null(this_file) && file.exists(this_file)) {
  setwd(dirname(normalizePath(this_file)))
} else if (requireNamespace("rstudioapi", quietly = TRUE)) {
  ctx <- try(rstudioapi::getActiveDocumentContext(), silent = TRUE)
  if (!inherits(ctx, "try-error") && !is.null(ctx$path) && file.exists(ctx$path)) {
    setwd(dirname(normalizePath(ctx$path)))
  }
}
message(": ", getwd())

ta <- commandArgs(trailingOnly = TRUE)
if (length(ta) >= 1) TASK <- ta[[1]]
if (length(ta) >= 2) MODE <- tolower(ta[[2]])
if (length(ta) >= 3) FOLD_TEST <- as.integer(ta[[3]])

if (!MODE %in% c("context", "test")) {
  stop("MODE  context  test: ", MODE)
}

stan_file_context <- file.path("stan", paste0(TASK, "_context.stan"))
stan_file_test <- file.path("stan", paste0(TASK, "_test.stan"))

need_cols <- c("subj", "trial", "choice", "a_amount", "a_prob", "b_amount", "b_prob")

pars_summary <- c("logit_alpha", "log_beta", "log_gamma", "tau", "alpha", "beta", "gamma")

stan_data_riskc_from_df <- function(df_sub, subj_label = NULL) {
  sfx <- if (is.null(subj_label)) "" else paste0("subj=", subj_label, "")

  N <- as.integer(nrow(df_sub))
  if (N < 1L) {
    stop("stan_data_riskc_from_df: N < 1", sfx)
  }

  a_amt <- as.numeric(df_sub$a_amount)
  b_amt <- as.numeric(df_sub$b_amount)
  a_prob <- as.numeric(df_sub$a_prob)
  b_prob <- as.numeric(df_sub$b_prob)

  if (any(!is.finite(a_amt)) || any(!is.finite(b_amt)) ||
      any(!is.finite(a_prob)) || any(!is.finite(b_prob))) {
    stop(" NA/Inf", sfx)
  }

  if (any(a_amt < 0) || any(b_amt < 0)) {
    stop("a_amount / b_amount  >= 0", sfx)
  }

  a_prob <- pmax(0, pmin(1, a_prob))
  b_prob <- pmax(0, pmin(1, b_prob))

  choice <- as.integer(round(df_sub$choice))
  if (anyNA(choice) || any(choice < 0L | choice > 1L)) {
    stop(
      "choice  0/1Stan: 0=A, 1=B",
      sfx,
      " choice : ",
      paste(unique(df_sub$choice[!is.na(df_sub$choice)]), collapse = ", ")
    )
  }

  list(
    N = N,
    a_amt = as.vector(a_amt),
    a_prob = as.vector(a_prob),
    b_amt = as.vector(b_amt),
    b_prob = as.vector(b_prob),
    choice = choice
  )
}

fit_one_subject <- function(df_sub, sm, subj_id, seed, meta = NULL) {
  df_sub <- df_sub %>% dplyr::arrange(trial)
  if (nrow(df_sub) == 0) {
    return(NULL)
  }

  n_tr <- nrow(df_sub)
  stan_data <- stan_data_riskc_from_df(df_sub, subj_label = subj_id)

  cores_use <- min(CHAINS, max(1L, getOption("mc.cores", 1L)))
  if (!is.null(meta)) {
    message(sprintf(
      "  [%d/%d] %s | fold=%s | subj=%s | trials=%d | chains=%d cores=%d | iter=%d warmup=%d | …",
      meta$i,
      meta$n,
      meta$phase,
      meta$fold,
      subj_id,
      n_tr,
      CHAINS,
      cores_use,
      ITER,
      WARMUP
    ))
    flush.console()
  }

  t_samp0 <- proc.time()
  fit <- sampling(
    sm,
    data = stan_data,
    iter = ITER,
    warmup = WARMUP,
    chains = CHAINS,
    cores = cores_use,
    seed = seed,
    control = list(adapt_delta = ADAPT_DELTA),
    refresh = as.integer(STAN_REFRESH)
  )
  t_samp <- (proc.time() - t_samp0)[["elapsed"]]

  t_post0 <- proc.time()
  smry <- summary(fit, pars = pars_summary)$summary
  waic_val <- NA_real_
  log_lik <- try(
    if (requireNamespace("loo", quietly = TRUE)) loo::extract_log_lik(fit) else NULL,
    silent = TRUE
  )
  if (!inherits(log_lik, "try-error") && !is.null(log_lik) && requireNamespace("loo", quietly = TRUE)) {
    waic_val <- tryCatch({
      w <- loo::waic(log_lik)
      w$estimates["waic", "Estimate"]
    }, error = function(e) NA_real_)
  }

  row_base <- tibble::tibble(
    subj = subj_id,
    waic = waic_val,
    n_trials = nrow(df_sub)
  )
  for (p in pars_summary) {
    if (p %in% rownames(smry)) {
      row_base[[paste0(p, "_mean")]] <- smry[p, "mean"]
      row_base[[paste0(p, "_sd")]] <- smry[p, "sd"]
    }
  }

  pm <- tibble::tibble(subj = subj_id)
  ex <- rstan::extract(fit, pars = c("alpha", "beta", "gamma", "tau"))
  pm$alpha_mean <- mean(ex$alpha)
  pm$beta_mean <- mean(ex$beta)
  pm$gamma_mean <- mean(ex$gamma)
  pm$tau_mean <- mean(ex$tau)
  pm$logit_alpha_mean <- mean(rstan::extract(fit, pars = "logit_alpha")$logit_alpha)
  pm$log_beta_mean <- mean(rstan::extract(fit, pars = "log_beta")$log_beta)
  pm$log_gamma_mean <- mean(rstan::extract(fit, pars = "log_gamma")$log_gamma)
  t_post <- (proc.time() - t_post0)[["elapsed"]]

  if (!is.null(meta)) {
    message(sprintf(
      "  [%d/%d] %s | fold=%s | subj=%s |  |  %.1fs | / %.1fs",
      meta$i,
      meta$n,
      meta$phase,
      meta$fold,
      subj_id,
      t_samp,
      t_post
    ))
    flush.console()
  }

  list(summary_row = row_base, posterior_means = pm)
}

run_fold_context <- function(fold, sm) {
  dir_data <- file.path("save", "hcog", "getdata", TASK, paste0("f_", fold))
  dir_out <- file.path("save", "hcog", "fit_context", TASK, paste0("f_", fold))
  dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)

  path_df <- file.path(dir_data, "df_train.csv")
  if (!file.exists(path_df)) {
    warning(" fold ", fold, " ", path_df)
    return(invisible(NULL))
  }

  df <- read.csv(path_df, stringsAsFactors = FALSE)
  if (!all(need_cols %in% names(df))) {
    stop(path_df, " ", paste(setdiff(need_cols, names(df)), collapse = ", "))
  }
  df <- df %>%
    dplyr::filter(complete.cases(
      subj, trial, choice, a_amount, a_prob, b_amount, b_prob
    ))

  subj_list <- sort(unique(df$subj))
  n_subj <- length(subj_list)
  message("[context] ", TASK, " fold ", fold, " | : ", n_subj, " | : ", path_df)
  tc <- df %>% dplyr::count(subj)
  message(sprintf(
    "  /: min=%d median=%.0f max=%d | =%d",
    min(tc$n),
    stats::median(tc$n),
    max(tc$n),
    nrow(df)
  ))
  flush.console()

  summary_all <- list()
  post_means <- list()

  for (k in seq_len(n_subj)) {
    subj_id <- subj_list[[k]]
    df_sub <- df %>% dplyr::filter(subj == subj_id)
    seed <- SEED_BASE + as.integer(subj_id) * 1000L + fold
    meta <- list(
      phase = "context",
      fold = as.character(fold),
      i = k,
      n = n_subj
    )
    out <- tryCatch(
      fit_one_subject(df_sub, sm, subj_id, seed, meta = meta),
      error = function(e) {
        message(sprintf("  [%d/%d] context | fold=%s | subj=%s | : %s", k, n_subj, fold, subj_id, e$message))
        flush.console()
        NULL
      }
    )
    if (is.null(out)) {
      warning(" ", subj_id, " ")
      next
    }
    summary_all[[length(summary_all) + 1]] <- out$summary_row
    post_means[[length(post_means) + 1]] <- out$posterior_means
  }

  if (length(summary_all) > 0) {
    summary_all_df <- dplyr::bind_rows(summary_all)
    write.csv(summary_all_df, file.path(dir_out, "summary_all.csv"), row.names = FALSE)
    pm_df <- dplyr::bind_rows(post_means)
    write.csv(pm_df, file.path(dir_out, "posterior_means.csv"), row.names = FALSE)
    message("[context] fold ", fold, "  | : ", nrow(summary_all_df), " / ", n_subj)
    message("  : ", file.path(dir_out, "summary_all.csv"))
    message("  : ", file.path(dir_out, "posterior_means.csv"))
    message(
      "   summary_all  _mean / _sd  stan/", TASK, "_test.stan "
    )
    flush.console()
  } else {
    warning("fold ", fold, " ")
  }
}

run_fold_test <- function(fold, sm) {
  dir_data <- file.path("save", "hcog", "getdata", TASK, paste0("f_", fold))
  dir_out <- file.path("save", "hcog", "fit_test", TASK, paste0("f_", fold))
  dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)

  path_df <- file.path(dir_data, "df_test_inference.csv")
  if (!file.exists(path_df)) {
    stop("test : ", path_df)
  }

  df <- read.csv(path_df, stringsAsFactors = FALSE)
  if (!all(need_cols %in% names(df))) {
    stop(path_df, " ", paste(setdiff(need_cols, names(df)), collapse = ", "))
  }
  df <- df %>%
    dplyr::filter(complete.cases(
      subj, trial, choice, a_amount, a_prob, b_amount, b_prob
    ))

  subj_list <- sort(unique(df$subj))
  n_subj <- length(subj_list)
  message("[test] ", TASK, " fold ", fold, " | : ", n_subj, " | : ", path_df)
  tc <- df %>% dplyr::count(subj)
  message(sprintf(
    "  /: min=%d median=%.0f max=%d | =%d",
    min(tc$n),
    stats::median(tc$n),
    max(tc$n),
    nrow(df)
  ))
  flush.console()

  summary_all <- list()
  post_means <- list()

  for (k in seq_len(n_subj)) {
    subj_id <- subj_list[[k]]
    df_sub <- df %>% dplyr::filter(subj == subj_id)
    seed <- SEED_BASE + as.integer(subj_id) * 1000L + fold + 17L
    meta <- list(
      phase = "test",
      fold = as.character(fold),
      i = k,
      n = n_subj
    )
    out <- tryCatch(
      fit_one_subject(df_sub, sm, subj_id, seed, meta = meta),
      error = function(e) {
        message(sprintf("  [%d/%d] test | fold=%s | subj=%s | : %s", k, n_subj, fold, subj_id, e$message))
        flush.console()
        NULL
      }
    )
    if (is.null(out)) {
      warning(" ", subj_id, " ")
      next
    }
    summary_all[[length(summary_all) + 1]] <- out$summary_row
    post_means[[length(post_means) + 1]] <- out$posterior_means
  }

  if (length(summary_all) > 0) {
    summary_all_df <- dplyr::bind_rows(summary_all)
    write.csv(summary_all_df, file.path(dir_out, "summary_all.csv"), row.names = FALSE)
    pm_df <- dplyr::bind_rows(post_means)
    write.csv(pm_df, file.path(dir_out, "posterior_means.csv"), row.names = FALSE)
    message("[test] fold ", fold, "  | : ", nrow(summary_all_df), " / ", n_subj)
    message("  : ", file.path(dir_out, "summary_all.csv"))
    message("  : ", file.path(dir_out, "posterior_means.csv"))
    message("   hcog_s3_eval_generation.py  df_test_generation  accuracy / MSE")
    flush.console()
  } else {
    warning("test fold ", fold, " ")
  }
}

if (MODE == "context") {
  stan_path <- stan_file_context
  if (!file.exists(stan_path)) {
    stop(" Stan : ", normalizePath(stan_path, mustWork = FALSE))
  }
  message("========== hcog_s2 | TASK=", TASK, " MODE=context | folds ", FOLD_MIN, ":", FOLD_MAX, " ==========")
  message(" Stan: ", normalizePath(stan_path))
  flush.console()
  sm <- stan_model(stan_path)
  for (fold in seq(FOLD_MIN, FOLD_MAX)) {
    message("----------  fold ", fold, " / ", FOLD_MAX, " ----------")
    flush.console()
    run_fold_context(fold, sm)
  }
} else {
  stan_path <- stan_file_test
  if (!file.exists(stan_path)) {
    stop(" Stan : ", normalizePath(stan_path, mustWork = FALSE))
  }
  message("========== hcog_s2 | TASK=", TASK, " MODE=test | fold=", FOLD_TEST, " ==========")
  message(" Stan: ", normalizePath(stan_path))
  flush.console()
  sm <- stan_model(stan_path)
  run_fold_test(FOLD_TEST, sm)
}

message(" | TASK=", TASK, " MODE=", MODE)
