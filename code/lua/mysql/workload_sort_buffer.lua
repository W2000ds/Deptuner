-- workload_sort_buffer.lua
-- 专门用于放大 sort_buffer_size / max_sort_length 依赖

local drv = sysbench.sql.driver()
local con

local PAYLOAD_LENGTH = 1024
local INSERT_ROWS = 2500
local DISTINCT_LEN = math.min(PAYLOAD_LENGTH, 768)
local SUBSTR_LEN = math.min(PAYLOAD_LENGTH, 768)
local LIMIT_ROWS = 100

local function random_payload(length)
    local template = string.rep("##########", math.max(1, math.floor(length / 10)))
    return sysbench.rand.string(template)
end

function thread_init()
    con = drv:connect()
    con:query("SET autocommit=1")
end

function thread_done()
    if con then
        con:disconnect()
        con = nil
    end
end

function prepare()
    print("Creating test table for sort_buffer dependency workload...")
    local prepare_con = drv:connect()
    prepare_con:query([[
        CREATE TABLE IF NOT EXISTS sb_sort_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            txt LONGTEXT,
            pad CHAR(10)
        ) ENGINE=InnoDB
    ]])

    print("Populating data...")
    prepare_con:query("TRUNCATE TABLE sb_sort_test")
    for i = 1, INSERT_ROWS do
        local rnd = random_payload(PAYLOAD_LENGTH)
        prepare_con:query(string.format("INSERT INTO sb_sort_test (txt, pad) VALUES ('%s','x')", rnd))
    end
    print("Data ready.")
    prepare_con:disconnect()
end

function event()
    local qtype = sysbench.rand.uniform(1, 4)
    if qtype == 1 then
        con:query(string.format("SELECT id, LEFT(txt, %d) FROM sb_sort_test ORDER BY txt DESC LIMIT %d", SUBSTR_LEN, LIMIT_ROWS))
    elseif qtype == 2 then
        con:query("SELECT pad, COUNT(*) FROM sb_sort_test GROUP BY pad ORDER BY COUNT(*) DESC LIMIT 10")
    elseif qtype == 3 then
        con:query(string.format("SELECT DISTINCT LEFT(txt, %d) FROM sb_sort_test ORDER BY LEFT(txt, %d) LIMIT %d", DISTINCT_LEN, DISTINCT_LEN, LIMIT_ROWS))
    else
        con:query(string.format("SELECT id FROM sb_sort_test ORDER BY SUBSTRING(txt,1,%d) LIMIT %d", SUBSTR_LEN, LIMIT_ROWS))
    end
end

function cleanup()
    print("Dropping sort_buffer dependency test table...")
    local cleanup_con = drv:connect()
    cleanup_con:query("DROP TABLE IF EXISTS sb_sort_test")
    cleanup_con:disconnect()
end
