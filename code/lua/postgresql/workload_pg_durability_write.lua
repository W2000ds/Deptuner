-- PostgreSQL write-heavy workload for WAL/checkpoint/group-commit dependency validation.

sysbench.cmdline.options = {
    ct_cleanup = {"Drop benchmark table during cleanup (on/off)", "off"},
}

local drv = sysbench.sql.driver()
local con

local SEED_ROWS = 5000
local PAYLOAD_LENGTH = 160

local function random_payload(length)
    local template = string.rep("##########", math.max(1, math.floor(length / 10)))
    return sysbench.rand.string(template)
end

function thread_init()
    con = drv:connect()
end

function thread_done()
    if con then
        con:disconnect()
        con = nil
    end
end

function prepare()
    print("Creating PostgreSQL write dependency table...")
    local prepare_con = drv:connect()
    prepare_con:query([[
        CREATE TABLE IF NOT EXISTS sb_pg_write_test (
            id BIGSERIAL PRIMARY KEY,
            k INTEGER NOT NULL,
            v BIGINT NOT NULL DEFAULT 0,
            pad TEXT NOT NULL
        )
    ]])
    prepare_con:query("CREATE INDEX IF NOT EXISTS idx_sb_pg_write_k ON sb_pg_write_test (k)")

    local row_count = 0
    local rs = prepare_con:query("SELECT COUNT(*) AS cnt FROM sb_pg_write_test")
    if rs then
        local row = rs:fetch_row()
        if row and row.cnt then
            row_count = tonumber(row.cnt) or 0
        end
    end

    if row_count >= SEED_ROWS then
        print(string.format(
            "Reusing existing PostgreSQL write dependency table with %d rows.",
            row_count
        ))
        prepare_con:query("ANALYZE sb_pg_write_test")
        prepare_con:disconnect()
        return
    end

    prepare_con:query("TRUNCATE TABLE sb_pg_write_test RESTART IDENTITY")

    for i = 1, SEED_ROWS do
        local k = sysbench.rand.uniform(1, SEED_ROWS)
        local pad = random_payload(PAYLOAD_LENGTH)
        prepare_con:query(string.format(
            "INSERT INTO sb_pg_write_test (k, v, pad) VALUES (%d, 0, '%s')",
            k, pad
        ))
    end
    prepare_con:query("ANALYZE sb_pg_write_test")
    prepare_con:disconnect()
end

function event()
    local target_id = sysbench.rand.uniform(1, SEED_ROWS)
    local sibling_id = sysbench.rand.uniform(1, SEED_ROWS)
    local k = sysbench.rand.uniform(1, SEED_ROWS)
    local pad = random_payload(PAYLOAD_LENGTH)

    con:query("BEGIN")
    con:query(string.format("UPDATE sb_pg_write_test SET v = v + 1 WHERE id = %d", target_id))
    con:query(string.format(
        "UPDATE sb_pg_write_test SET k = %d, pad = '%s' WHERE id = %d",
        k, pad, sibling_id
    ))
    con:query("COMMIT")
end

function cleanup()
    if tostring(sysbench.opt.ct_cleanup or "off") ~= "on" then
        print("Skipping PostgreSQL write dependency cleanup (ct_cleanup=off).")
        return
    end

    print("Dropping PostgreSQL write dependency table...")
    local cleanup_con = drv:connect()
    cleanup_con:query("DROP TABLE IF EXISTS sb_pg_write_test")
    cleanup_con:disconnect()
end
