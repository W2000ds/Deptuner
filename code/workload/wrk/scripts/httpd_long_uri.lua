local long_query = "/test.html?dep=" .. string.rep("a", 12000)

request = function()
  return wrk.format("GET", long_query)
end
