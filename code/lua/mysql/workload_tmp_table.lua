-- workload_tmp_table.lua
-- 用于放大 tmp_table_size / max_heap_table_size 对内部临时表落盘的影响

local drv = sysbench.sql.driver()
local con

local INSERT_ROWS = 5000
local PAYLOAD_LENGTH = 1200
local GROUP_KEY_LENGTH = 300
local LIMIT_ROWS = 300

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
    print("Creating test table for tmp_table dependency workload...")
    local prepare_con = drv:connect()
    prepare_con:query([[
        CREATE TABLE IF NOT EXISTS sb_tmp_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            txt LONGTEXT,
            pad CHAR(16)
        ) ENGINE=InnoDB
    ]])

    print("Populating data...")
    prepare_con:query("TRUNCATE TABLE sb_tmp_test")
    for i = 1, INSERT_ROWS do
        local rnd = random_payload(PAYLOAD_LENGTH)
        prepare_con:query(string.format("INSERT INTO sb_tmp_test (txt, pad) VALUES ('%s','pad')", rnd))
    end
    print("Data ready.")
    prepare_con:disconnect()
end

function event()
    local qtype = sysbench.rand.uniform(1, 3)
    if qtype == 1 then
        con:query(string.format([[
            SELECT LEFT(txt, %d) AS grp_key, COUNT(*) AS c
            FROM sb_tmp_test
            GROUP BY grp_key
            ORDER BY c DESC
            LIMIT %d
        ]], GROUP_KEY_LENGTH, LIMIT_ROWS))
    elseif qtype == 2 then
        con:query(string.format([[
            SELECT DISTINCT LEFT(txt, %d)
            FROM sb_tmp_test
            ORDER BY LEFT(txt, %d)
            LIMIT %d
        ]], GROUP_KEY_LENGTH, GROUP_KEY_LENGTH, LIMIT_ROWS))
    else
        con:query(string.format([[
            SELECT id
            FROM sb_tmp_test
            ORDER BY LEFT(txt, %d)
            LIMIT %d
        ]], GROUP_KEY_LENGTH, LIMIT_ROWS))
    end
end

function cleanup()
    print("Dropping tmp_table dependency test table...")
    local cleanup_con = drv:connect()
    cleanup_con:query("DROP TABLE IF EXISTS sb_tmp_test")
    cleanup_con:disconnect()
end
