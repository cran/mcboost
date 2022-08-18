## ---- echo = FALSE------------------------------------------------------------
NOT_CRAN = identical(tolower(Sys.getenv("NOT_CRAN")), "true")
HAS_DEPS = identical(tolower(Sys.getenv("_R_CHECK_DEPENDS_ONLY_")), "true")
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  purl = (NOT_CRAN & !HAS_DEPS),
  eval = (NOT_CRAN & !HAS_DEPS)
)

