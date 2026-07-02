-- workload_durability_write.lua
-- 用于放大 sync_binlog / innodb_flush_log_at_trx_commit 在写事务上的影响

local drv = sysbench.sql.driver()
local con

local SEED_ROWS = 3000
local PAYLOAD_LENGTH = 120

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
    print("Creating test table for durability write workload...")
    local prepare_con = drv:connect()
    prepare_con:query([[
        CREATE TABLE IF NOT EXISTS sb_write_test (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            k INT NOT NULL,
            v BIGINT NOT NULL DEFAULT 0,
            pad VARCHAR(256) NOT NULL,
            KEY idx_k (k)
        ) ENGINE=InnoDB
    ]])

    print("Populating seed data...")
    prepare_con:query("TRUNCATE TABLE sb_write_test")
    for i = 1, SEED_ROWS do
        local k = sysbench.rand.uniform(1, SEED_ROWS)
        local pad = random_payload(PAYLOAD_LENGTH)
        prepare_con:query(string.format(
            "INSERT INTO sb_write_test (k, v, pad) VALUES (%d, 0, '%s')",
            k, pad
        ))
    end
    print("Data ready.")
    prepare_con:disconnect()
end

function event()
    local target_id = sysbench.rand.uniform(1, SEED_ROWS)
    local k = sysbench.rand.uniform(1, SEED_ROWS)
    local pad = random_payload(PAYLOAD_LENGTH)

    con:query("BEGIN")
    con:query(string.format("UPDATE sb_write_test SET v = v + 1 WHERE id = %d", target_id))
    con:query(string.format("INSERT INTO sb_write_test (k, v, pad) VALUES (%d, 1, '%s')", k, pad))
    con:query("COMMIT")
end

function cleanup()
    print("Dropping durability dependency test table...")
    local cleanup_con = drv:connect()
    cleanup_con:query("DROP TABLE IF EXISTS sb_write_test")
    cleanup_con:disconnect()
end
