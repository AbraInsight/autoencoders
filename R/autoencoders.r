install.packages(pkgs = "https://cran.r-project.org/bin/macosx/el-capitan/contrib/3.4/insuranceData_1.0.tgz", 
                 lib = .libPaths()[1], 
                 repos = NULL)

libs.to.load <- list("insuranceData",
                     "data.table")

invisible(lapply(X = libs.to.load, 
                 FUN = function(x) library(package = x, 
                                           lib.loc = .libPaths()[1],
                                           character.only = TRUE)))

data(ClaimsLong)
ins.data.dt <- data.table(ClaimsLong)