
for (seed in 0:2) {
    for (dataset in c(
        "iris",
        "glass",
        "wine",
        "breast-cancer",
        "abalone",
        "dry-beans",
        "change_in_mean",
        "change_in_covariance",
        "dirichlet"
    )) {
        file_name = paste0(c("dataset_caches/", dataset, "_", seed, ".csv"), collapse="")
        print(file_name)
        X <- as.matrix(read.table(file_name, sep=","))
        min_size = max(2, round(nrow(X)/100))
        tic = Sys.time()
        ecp_result <- ecp::e.divisive(X, min.size=min_size)
        toc = Sys.time()
        print(ecp_result$estimates - 1)
        print(as.numeric(toc - tic))
    }
}