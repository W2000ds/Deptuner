-- PostgreSQL workload for work_mem/temp-buffer dependency validation.

local drv = sysbench.sql.driver()
local con

local INSERT_ROWS = 8000
local PAYLOAD_LENGTH = 1024
local LIMIT_ROWS = 200

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
    print("Creating PostgreSQL sort spill dependency table...")
    local prepare_con = drv:connect()
    prepare_con:query([[
        CREATE TABLE IF NOT EXISTS sb_pg_sort_test (
            id BIGSERIAL PRIMARY KEY,
            k INTEGER NOT NULL,
            txt TEXT NOT NULL,
            pad TEXT NOT NULL
        )
    ]])
    prepare_con:query("TRUNCATE TABLE sb_pg_sort_test RESTART IDENTITY")

    for i = 1, INSERT_ROWS do
        local k = sysbench.rand.uniform(1, INSERT_ROWS)
        local txt = random_payload(PAYLOAD_LENGTH)
        local pad = random_payload(120)
        prepare_con:query(string.format(
            "INSERT INTO sb_pg_sort_test (k, txt, pad) VALUES (%d, '%s', '%s')",
            k, txt, pad
        ))
    end
    prepare_con:query("ANALYZE sb_pg_sort_test")
    prepare_con:disconnect()
end

function event()
    local qtype = sysbench.rand.uniform(1, 4)
    if qtype == 1 then
        con:query(string.format(
            "SELECT id, left(txt, 512) FROM sb_pg_sort_test ORDER BY txt DESC LIMIT %d",
            LIMIT_ROWS
        ))
    elseif qtype == 2 then
        con:query("SELECT k, count(*) FROM sb_pg_sort_test GROUP BY k ORDER BY count(*) DESC LIMIT 100")
    elseif qtype == 3 then
        con:query(string.format(
            "SELECT DISTINCT left(txt, 512) FROM sb_pg_sort_test ORDER BY left(txt, 512) LIMIT %d",
            LIMIT_ROWS
        ))
    else
        con:query([[
            CREATE TEMP TABLE IF NOT EXISTS sb_pg_temp AS
            SELECT id, k, left(txt, 128) AS txt FROM sb_pg_sort_test LIMIT 1000
        ]])
        con:query("TRUNCATE sb_pg_temp")
        con:query("INSERT INTO sb_pg_temp SELECT id, k, left(txt, 128) FROM sb_pg_sort_test ORDER BY txt LIMIT 1000")
        con:query("SELECT k, count(*) FROM sb_pg_temp GROUP BY k ORDER BY count(*) DESC LIMIT 50")
    end
end

function cleanup()
    print("Dropping PostgreSQL sort spill dependency table...")
    local cleanup_con = drv:connect()
    cleanup_con:query("DROP TABLE IF EXISTS sb_pg_sort_test")
    cleanup_con:disconnect()
end
