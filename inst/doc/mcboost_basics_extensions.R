## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library("mcboost")
library("mlr3")
set.seed(83007)

## -----------------------------------------------------------------------------
tsk = tsk("sonar")
d = tsk$data(cols = tsk$feature_names)
l = tsk$data(cols = tsk$target_names)[[1]]
mc = MCBoost$new(auditor_fitter = "TreeAuditorFitter")
mc$multicalibrate(d[1:200,], l[1:200])

## -----------------------------------------------------------------------------
mc$predict_probs(d[201:208,])

## -----------------------------------------------------------------------------
library(data.table)
adult_train = fread(
  "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/adult_train.csv",
  stringsAsFactors = TRUE
)
adult_train$Country = NULL
adult_train$fnlwgt = NULL
train_tsk = TaskClassif$new("adult_train", adult_train, target = "Target")

## -----------------------------------------------------------------------------
library(mlr3pipelines)
pipe = po("collapsefactors", no_collapse_above_prevalence = 0.0006) %>>%
  po("fixfactors") %>>%
  po("encode") %>>%
  po("imputehist")
prep_task = pipe$train(train_tsk)[[1]]

## -----------------------------------------------------------------------------
prep_task$set_col_roles(c("Race.Amer.Indian.Eskimo", "Race.Asian.Pac.Islander", "Race.Black", "Race.Other", "Race.White"), remove_from = "feature")

## -----------------------------------------------------------------------------
library(mlr3learners)
l = lrn("classif.ranger", num.trees = 10L, predict_type = "prob")
l$train(prep_task)

## -----------------------------------------------------------------------------
init_predictor = function(data) {
  l$predict_newdata(data)$prob[, 2]
}

## -----------------------------------------------------------------------------
data = prep_task$data(cols = prep_task$feature_names)
labels = 1 - one_hot(prep_task$data(cols = prep_task$target_names)[[1]])

## -----------------------------------------------------------------------------
mc = MCBoost$new(auditor_fitter = "RidgeAuditorFitter", init_predictor = init_predictor)
mc$multicalibrate(data, labels)

## -----------------------------------------------------------------------------
mc

## -----------------------------------------------------------------------------
adult_test = fread(
  "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/adult_test.csv",
  stringsAsFactors = TRUE
)
adult_test$Country = NULL
adult_test$fnlwgt = NULL

# The first row seems to have an error
adult_test = adult_test[Target != "",]
adult_test$Target = droplevels(adult_test$Target)

# Note, that we have to convert columns from numeric to integer here:
sdc = train_tsk$feature_types[type == "integer", id]
adult_test[, (sdc) := lapply(.SD, as.integer), .SDcols = sdc]

test_tsk = TaskClassif$new("adult_test", adult_test, target = "Target")
prep_test = pipe$predict(test_tsk)[[1]]

## -----------------------------------------------------------------------------
test_data = prep_test$data(cols = prep_test$feature_names)
test_labels = 1 - one_hot(prep_test$data(cols = prep_test$target_names)[[1]])

## -----------------------------------------------------------------------------
prs = mc$predict_probs(test_data)

## -----------------------------------------------------------------------------
mean(round(prs) == test_labels)

## -----------------------------------------------------------------------------
mean(round(init_predictor(test_data)) == test_labels)

## -----------------------------------------------------------------------------
# Get bias per subgroup for multi-calibrated predictor
adult_test$biasmc = (prs - test_labels)
adult_test[, .(abs(mean(biasmc)), .N), by = .(Race)]
# Get bias per subgroup for initial predictor
adult_test$biasinit = (init_predictor(test_data) - test_labels)
adult_test[, .(abs(mean(biasinit)), .N), by = .(Race)]

## -----------------------------------------------------------------------------
ae = mc$auditor_effect(test_data)
hist(ae)

## -----------------------------------------------------------------------------
effect = apply(test_data[ae >= median(ae[ae>0]),], 2, quantile)
no_effect  = apply(test_data[ae < median(ae[ae>0]),], 2, quantile)
difference = apply((effect-no_effect), 2, mean)
difference[difference > 0.1]

## -----------------------------------------------------------------------------
test_data[ae >= median(ae[ae>0]), names(which(difference > 0.1)), with = FALSE]

## -----------------------------------------------------------------------------
prs = mc$predict_probs(test_data, t = 3L)

## -----------------------------------------------------------------------------
tsk = tsk("sonar")
data = tsk$data()[, Class := as.integer(Class) - 1L]
mod = glm(data = data, formula = Class ~ .)

## -----------------------------------------------------------------------------
init_predictor = function(data) {
  predict(mod, data)
}

## -----------------------------------------------------------------------------
d = data[, -1]
l = data$Class
mc = MCBoost$new(init_predictor = init_predictor)
mc$multicalibrate(d[1:200,], l[1:200])
mc$predict_probs(d[201:208,])

## -----------------------------------------------------------------------------
tsk = tsk("sonar")

## -----------------------------------------------------------------------------
learner = lrn("classif.ranger", predict_type = "prob")
learner$train(tsk)
init_predictor = mlr3_init_predictor(learner)

## -----------------------------------------------------------------------------
d = data[, -1]
l = data$Class
mc = MCBoost$new(init_predictor = init_predictor, auditor_fitter=CVTreeAuditorFitter$new(), max_iter = 2L)
mc$multicalibrate(d[1:200,], l[1:200])
mc$predict_probs(d[201:208,])

## -----------------------------------------------------------------------------
tsk = tsk("sonar")

## -----------------------------------------------------------------------------
learner = lrn("classif.ranger", predict_type = "prob")
learner$train(tsk)
init_predictor = mlr3_init_predictor(learner)

## -----------------------------------------------------------------------------
d = data[, -1]
l = data$Class
mc = MCBoost$new(
  init_predictor = init_predictor,
  auditor_fitter= TreeAuditorFitter$new(),
  iter_sampling = "bootstrap"
)
mc$multicalibrate(d[1:200,], l[1:200])
mc$predict_probs(d[201:208,])

## -----------------------------------------------------------------------------
tsk = tsk("sonar")
data = tsk$data(cols = tsk$feature_names)
labels = tsk$data(cols = tsk$target_names)[[1]]

## -----------------------------------------------------------------------------
rf = LearnerAuditorFitter$new(lrn("regr.rpart", minsplit = 10L))
mc = MCBoost$new(auditor_fitter = rf)
mc$multicalibrate(data, labels)

## -----------------------------------------------------------------------------
data[, Bin := sample(c(1, 0), nrow(data), replace = TRUE)]

## -----------------------------------------------------------------------------
rf = SubpopAuditorFitter$new(list(
  "Bin",
  function(data) {data[["V1"]] > 0.2},
  function(data) {data[["V1"]] > 0.2 | data[["V3"]] < 0.29}
))

## -----------------------------------------------------------------------------
mc = MCBoost$new(auditor_fitter = rf)
mc$multicalibrate(data, labels)

## -----------------------------------------------------------------------------
mc$predict_probs(data)

## -----------------------------------------------------------------------------
rf = SubgroupAuditorFitter$new(list(
  rep(c(0, 1), 104),
  rep(c(1, 1, 1, 0), 52)
))

## -----------------------------------------------------------------------------
mc = MCBoost$new(auditor_fitter = rf)
mc$multicalibrate(data, labels)

## -----------------------------------------------------------------------------
predict_masks = list(
  rep(c(0, 1), 52),
  rep(c(1, 1, 1, 0), 26)
)

## -----------------------------------------------------------------------------
mc$predict_probs(data[1:104,], subgroup_masks = predict_masks)

## -----------------------------------------------------------------------------
tsk = tsk("penguins")
# first we convert to a binary task
row_ids = tsk$data(cols = c("species", "..row_id"))[species %in% c("Adelie", "Gentoo")][["..row_id"]]
tsk$filter(row_ids)$droplevels()
tsk

## -----------------------------------------------------------------------------
library("mlr3pipelines")
library("mlr3learners")

# Convert task to X,y
X = tsk$data(cols = tsk$feature_names)
y = tsk$data(cols = tsk$target_names)

# Our inital model is a pipeline that imputes missings and encodes categoricals
init_model = as_learner(po("encode") %>>% po("imputehist") %>>%
  lrn("classif.glmnet", predict_type = "prob"))
# And we fit it on a subset of the data in order to simulate a poorly performing model.
init_model$train(tsk$clone()$filter(row_ids[c(1:9, 160:170)]))
init_model$predict(tsk)$score()

# We define a pipeline that imputes missings and encodes categoricals
auditor = as_learner(po("encode") %>>% po("imputehist") %>>% lrn("regr.rpart"))

mc = MCBoost$new(auditor_fitter = auditor, init_predictor = init_model)
mc$multicalibrate(X, y)

## -----------------------------------------------------------------------------
mc

## -----------------------------------------------------------------------------
library(data.table)
library(mlr3oml)
oml = OMLData$new(42730)
data = oml$data

tsk = TaskRegr$new("communities_crime", data, target = "ViolentCrimesPerPop")

## -----------------------------------------------------------------------------
summary(data$ViolentCrimesPerPop)

## -----------------------------------------------------------------------------
tsk$set_row_roles(sample(tsk$row_roles$use, 500), "validation")

## -----------------------------------------------------------------------------
library(mlr3pipelines)
pipe =  po("imputehist")
prep_task = pipe$train(list(tsk))[[1]]

prep_task$set_col_roles(c("racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "community"), remove_from = "feature")

## -----------------------------------------------------------------------------
library(mlr3learners)
l = lrn("regr.ranger", num.trees = 10L)
l$train(prep_task)

## -----------------------------------------------------------------------------
init_predictor = function(data) {
  l$predict_newdata(data)$response
}

## -----------------------------------------------------------------------------
data = prep_task$data(cols = prep_task$feature_names)
labels = prep_task$data(cols = prep_task$target_names)[[1]]

## -----------------------------------------------------------------------------
mc = MCBoost$new(auditor_fitter = "RidgeAuditorFitter", init_predictor = init_predictor, eta = 0.1)
mc$multicalibrate(data, labels)

## -----------------------------------------------------------------------------
test_task = tsk$clone()
test_task$row_roles$use = test_task$row_roles$validation
test_task = pipe$predict(list(test_task))[[1]]
test_data = test_task$data(cols = tsk$feature_names)
test_labels = test_task$data(cols = tsk$target_names)[[1]]

## -----------------------------------------------------------------------------
prs = mc$predict_probs(test_data)

## -----------------------------------------------------------------------------
mean((prs - test_labels)^2)

## -----------------------------------------------------------------------------
mean((init_predictor(test_data) - test_labels)^2)

## -----------------------------------------------------------------------------
test_data$se_mcboost = (prs - test_labels)^2
test_data$se_init = (init_predictor(test_data) - test_labels)^2

test_data[, .(mcboost = mean(se_mcboost), initial = mean(se_init), .N), by = .(racepctblack > 0.5)]

