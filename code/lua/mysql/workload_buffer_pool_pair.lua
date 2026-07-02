-- workload_buffer_pool_pair.lua
-- 用于验证 innodb_buffer_pool_size 与 innodb_buffer_pool_instances 的联动影响

local drv = sysbench.sql.driver()
local con

local INSERT_ROWS = 8000
local PAYLOAD_LENGTH = 1536
local RANGE_WIDTH = 500

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
    print("Creating test table for buffer_pool dependency workload...")
    local prepare_con = drv:connect()
    prepare_con:query([[
        CREATE TABLE IF NOT EXISTS sb_bp_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            k INT NOT NULL,
            payload LONGTEXT NOT NULL,
            KEY idx_k (k)
        ) ENGINE=InnoDB
    ]])

    print("Populating data...")
    prepare_con:query("TRUNCATE TABLE sb_bp_test")
    for i = 1, INSERT_ROWS do
        local k = sysbench.rand.uniform(1, INSERT_ROWS)
        local p = random_payload(PAYLOAD_LENGTH)
        prepare_con:query(string.format(
            "INSERT INTO sb_bp_test (k, payload) VALUES (%d, '%s')",
            k, p
        ))
    end
    print("Data ready.")
    prepare_con:disconnect()
end

function event()
    local qtype = sysbench.rand.uniform(1, 3)
    local left = sysbench.rand.uniform(1, INSERT_ROWS - RANGE_WIDTH)
    local right = left + RANGE_WIDTH
    local rid = sysbench.rand.uniform(1, INSERT_ROWS)

    if qtype == 1 then
        con:query(string.format(
            "SELECT SQL_NO_CACHE COUNT(*) FROM sb_bp_test WHERE id BETWEEN %d AND %d",
            left, right
        ))
    elseif qtype == 2 then
        con:query(string.format(
            "SELECT SQL_NO_CACHE id FROM sb_bp_test WHERE k BETWEEN %d AND %d ORDER BY id DESC LIMIT 300",
            left, right
        ))
    else
        con:query(string.format(
            "SELECT SQL_NO_CACHE LEFT(payload, 256) FROM sb_bp_test WHERE id = %d",
            rid
        ))
    end
end

function cleanup()
    print("Dropping buffer_pool dependency test table...")
    local cleanup_con = drv:connect()
    cleanup_con:query("DROP TABLE IF EXISTS sb_bp_test")
    cleanup_con:disconnect()
end
