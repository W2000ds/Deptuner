wrk.method = "POST"
wrk.body = string.rep("a", 8192)
wrk.headers["Content-Type"] = "application/octet-stream"
