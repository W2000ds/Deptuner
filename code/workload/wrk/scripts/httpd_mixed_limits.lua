local long_path = "/test.html?dep=" .. string.rep("a", 12000)

request = function()
  local headers = {
    ["X-DepTest-Large-Header"] = string.rep("b", 12000),
    ["Content-Type"] = "application/octet-stream"
  }
  return wrk.format("POST", long_path, headers, string.rep("c", 8192))
end
