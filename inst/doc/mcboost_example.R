## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, message = FALSE---------------------------------------------------
library(tidyverse)
library(PracTools)
library(ranger)
library(neuralnet)
library(formattable)
library(mlr3)
library(mlr3learners)
library(mcboost)

## -----------------------------------------------------------------------------
data(nhis.large)
?nhis.large

## -----------------------------------------------------------------------------
categorical <- c("age.grp", "parents", "educ", "inc.grp", "doing.lw",
  "limited", "sex", "hisp", "race")

nhis <- nhis.large %>%
  mutate_at(categorical, as.factor) %>%
  mutate_at(categorical, fct_explicit_na) %>%
  drop_na(notcov) %>%
  select(all_of(categorical), notcov, svywt, ID)

nhis$notcov <- factor(ifelse(nhis$notcov == 1, "notcov", "cov"))

nhis_enc <- data.frame(model.matrix(notcov ~ ., data = nhis)[,-1])
nhis_enc$notcov <- nhis$notcov
nhis_enc$sex <- nhis$sex
nhis_enc$hisp <- nhis$hisp
nhis_enc$race <- nhis$race
nhis_enc$inv_wt <- (1 / nhis$svywt)

## -----------------------------------------------------------------------------
set.seed(2953)

test <- nhis_enc %>% slice_sample(prop = 0.25, weight_by = inv_wt)

nontest_g <- nhis_enc %>% anti_join(test, by = "ID")

train_g <- nontest_g %>% slice_sample(prop = 0.75)

post <- nontest_g %>% anti_join(train_g, by = "ID") %>% select(-ID, -svywt, -inv_wt, -c(sex:race))

train <- train_g %>% select(-ID, -svywt, -inv_wt, -c(sex:race), -c(sex2:race3))

## -----------------------------------------------------------------------------
train_g %>% summarise_at(vars(sex2:race3), mean)
# hispanic individuals
1 - sum(train_g %>% summarise_at(vars(hisp2:hisp4), mean))

## -----------------------------------------------------------------------------
post %>% summarise_at(vars(sex2:race3), mean)
# hispanic individuals
1 - sum(post %>% summarise_at(vars(hisp2:hisp4), mean))

## -----------------------------------------------------------------------------
test %>% summarise_at(vars(sex2:race3), mean)
# hispanic individuals
1 - sum(test %>% summarise_at(vars(hisp2:hisp4), mean))

## ---- message = FALSE---------------------------------------------------------
nnet <- neuralnet(notcov ~ .,
  hidden = 5,
  linear.output = FALSE,
  err.fct = 'ce',
  threshold = 0.5,
  lifesign = 'full',
  data = train
)

## -----------------------------------------------------------------------------
init_nnet = function(data) {
  predict(nnet, data)[, 2]
}

## -----------------------------------------------------------------------------
d1 <- select(post, -c(notcov, sex2:race3))
d2 <- select(post, -notcov)
l <- 1 - one_hot(post$notcov)

## -----------------------------------------------------------------------------
ridge = LearnerAuditorFitter$new(lrn("regr.glmnet", alpha = 0, lambda = 2 / nrow(post)))

pops = SubpopAuditorFitter$new(list("sex2", "hisp2", "hisp3", "hisp4", "race2", "race3"))

## -----------------------------------------------------------------------------
nnet_mc_ridge = MCBoost$new(init_predictor = init_nnet,
                            auditor_fitter = ridge,
                            multiplicative = TRUE,
                            partition = TRUE,
                            max_iter = 15)
nnet_mc_ridge$multicalibrate(d1, l)

nnet_mc_subpop = MCBoost$new(init_predictor = init_nnet,
                             auditor_fitter = pops,
                             partition = TRUE,
                             max_iter = 15)
nnet_mc_subpop$multicalibrate(d2, l)

## -----------------------------------------------------------------------------
test$nnet <- predict(nnet, newdata = test)[, 2]
test$nnet_mc_ridge <- nnet_mc_ridge$predict_probs(test)
test$nnet_mc_subpop <- nnet_mc_subpop$predict_probs(test)

test$c_nnet <- round(test$nnet)
test$c_nnet_mc_ridge <- round(test$nnet_mc_ridge)
test$c_nnet_mc_subpop <- round(test$nnet_mc_subpop)
test$label <- 1 - one_hot(test$notcov)

## -----------------------------------------------------------------------------
mean(test$c_nnet == test$label)
mean(test$c_nnet_mc_ridge == test$label)
mean(test$c_nnet_mc_subpop == test$label)

## ---- warning = FALSE---------------------------------------------------------
test <- test %>%
  group_by(sex, hisp) %>%
  mutate(sex_hisp = cur_group_id()) %>%
  group_by(sex, race) %>%
  mutate(sex_race = cur_group_id()) %>%
  group_by(hisp, race) %>%
  mutate(hisp_race = cur_group_id()) %>%
  ungroup()

grouping_vars <- c("sex", "hisp", "race", "sex_hisp", "sex_race", "hisp_race")

eval <- map(grouping_vars, group_by_at, .tbl = test) %>%
  map(summarise,
      'accuracy_nnet' = mean(c_nnet == label),
      'accuracy_nnet_mc_ridge' = mean(c_nnet_mc_ridge == label),
      'accuracy_nnet_mc_subpop' = mean(c_nnet_mc_subpop == label),
      'size' = n()) %>%
  bind_rows()

## -----------------------------------------------------------------------------
eval %>%
  arrange(desc(size)) %>%
  select(size, accuracy_nnet:accuracy_nnet_mc_subpop) %>%
  round(., digits = 3) %>%
  formattable(., lapply(1:nrow(eval), function(row) {
  area(row, col = 2:4) ~ color_tile("transparent", "lightgreen")
    }))

## -----------------------------------------------------------------------------
rf <- ranger(notcov ~ ., data = train, probability = TRUE)

## -----------------------------------------------------------------------------
init_rf = function(data) {
  predict(rf, data)$prediction[, 2]
}

## -----------------------------------------------------------------------------
ridge = LearnerAuditorFitter$new(lrn("regr.glmnet", alpha = 0, lambda = 2 / nrow(post)))

lasso = LearnerAuditorFitter$new(lrn("regr.glmnet", alpha = 1, lambda = 40 / nrow(post)))

## -----------------------------------------------------------------------------
rf_mc_ridge = MCBoost$new(init_predictor = init_rf,
                          auditor_fitter = ridge,
                          multiplicative = TRUE,
                          partition = TRUE,
                          max_iter = 15)
rf_mc_ridge$multicalibrate(d1, l)

rf_mc_lasso = MCBoost$new(init_predictor = init_rf,
                          auditor_fitter = lasso,
                          multiplicative = TRUE,
                          partition = TRUE,
                          max_iter = 15)
rf_mc_lasso$multicalibrate(d2, l)

## -----------------------------------------------------------------------------
test$rf <- predict(rf, test)$prediction[, 2]
test$rf_mc_ridge <- rf_mc_ridge$predict_probs(test)
test$rf_mc_lasso <- rf_mc_lasso$predict_probs(test)

test$c_rf <- round(test$rf)
test$c_rf_mc_ridge <- round(test$rf_mc_ridge)
test$c_rf_mc_lasso <- round(test$rf_mc_lasso)

## -----------------------------------------------------------------------------
mean(test$c_rf == test$label)
mean(test$c_rf_mc_ridge == test$label)
mean(test$c_rf_mc_lasso == test$label)

## -----------------------------------------------------------------------------
eval <- map(grouping_vars, group_by_at, .tbl = test) %>%
  map(summarise,
      'bias_rf' = abs(mean(rf) - mean(label))*100,
      'bias_rf_mc_ridge' = abs(mean(rf_mc_ridge) - mean(label))*100,
      'bias_rf_mc_lasso' = abs(mean(rf_mc_lasso) - mean(label))*100,
      'size' = n()) %>%
  bind_rows()

## -----------------------------------------------------------------------------
eval %>%
  arrange(desc(size)) %>%
  select(size, bias_rf:bias_rf_mc_lasso) %>%
  round(., digits = 3) %>%
  formattable(., lapply(1:nrow(eval), function(row) {
  area(row, col = 2:4) ~ color_tile("lightgreen", "transparent")
    }))

