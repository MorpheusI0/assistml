library(plumber)

r <- plumb("assist.R")
r$run(port = 8080, host = "0.0.0.0")
