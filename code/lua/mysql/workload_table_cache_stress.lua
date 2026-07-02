-- workload_table_cache_stress.lua
-- A sysbench workload designed to stress MySQL's table cache, file descriptors, and connection limits.

require("oltp_common")

local DEFAULT_TABLE_COUNT = 200
local DEFAULT_TABLE_SIZE = 0  -- 此脚本不批量插入数据，保持 0 即可

local function get_table_count()
  return tonumber(sysbench.opt.tables) or DEFAULT_TABLE_COUNT
end

local function create_table_stmt(tbl)
  return string.format("CREATE TABLE IF NOT EXISTS %s (id INT PRIMARY KEY AUTO_INCREMENT, pad CHAR(100)) ENGINE=InnoDB", tbl)
end

local function ensure_table(con, tbl)
  con:query(create_table_stmt(tbl))
end

local function safe_query(con, tbl, sql, opts)
  local ok, err = pcall(con.query, con, sql)
  if not ok then
    local recreate = opts and opts.recreate_on_missing
    if recreate and string.find(err, "doesn't exist", 1, true) then
      ensure_table(con, tbl)
      return
    end
    error(err)
  end
end

local function custom_prepare()
  db_connect()
  local table_count = get_table_count()
  print(string.format("Creating %d small tables...", table_count))
  for i = 1, table_count do
    local tbl = string.format("t_%d", i)
    db_query(create_table_stmt(tbl))
  end
  db_disconnect()
end

local function custom_cleanup()
  db_connect()
  local table_count = get_table_count()
  print(string.format("Dropping %d tables created by stress workload...", table_count))
  for i = 1, table_count do
    db_query(string.format("DROP TABLE IF EXISTS t_%d", i))
  end
  db_disconnect()
end

-- 覆盖默认命令，将 prepare/cleanup 指向自定义逻辑
sysbench.cmdline.options.tables = {"Number of custom tables", DEFAULT_TABLE_COUNT}
sysbench.cmdline.options.table_size = {"Rows per table (unused)", DEFAULT_TABLE_SIZE}
sysbench.cmdline.commands.prepare = {custom_prepare, sysbench.cmdline.PARALLEL_COMMAND}
sysbench.cmdline.commands.cleanup = {custom_cleanup, sysbench.cmdline.PARALLEL_COMMAND}

function thread_init()
  drv = sysbench.sql.driver()
  con = drv:connect()
end

function thread_done()
  con:disconnect()
end

function event()
  -- randomly pick a table
  local table_count = get_table_count()
  local id = math.random(1, table_count)
  local tbl = string.format("t_%d", id)

  local op = math.random(1, 10)
  if op <= 6 then
    -- normal query
    safe_query(con, tbl, string.format("SELECT COUNT(*) FROM %s", tbl), {recreate_on_missing = true})
  elseif op <= 8 then
    -- metadata ops to stress table cache and FD usage
    safe_query(con, tbl, "SHOW TABLE STATUS LIKE 't_%'", {recreate_on_missing = false})
  elseif op == 9 then
    -- force table reopen path
    safe_query(con, tbl, "FLUSH TABLES", {recreate_on_missing = false})
  else
    -- drop & recreate occasionally (simulate DDL)
    safe_query(con, tbl, string.format("DROP TABLE IF EXISTS %s", tbl), {recreate_on_missing = false})
    ensure_table(con, tbl)
  end

  -- simulate reconnect pressure
  if math.random() < 0.02 then
    con:disconnect()
    con = drv:connect()
  end
end
