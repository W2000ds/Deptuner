import json


def build_context_parts(sys_name, rule_ids, config, factors, workload_controller=None):
    """Build deterministic evidence from the evaluated config and workload context."""
    if not rule_ids:
        return []
    config = config or {}
    factors = factors or {}
    lines = [
        f"WORKLOAD_EVIDENCE sys={sys_name} rules={','.join(rule_ids)} "
        f"factors={json.dumps(factors, sort_keys=True)}"
    ]
    for rule_id in rule_ids:
        line = _rule_workload_line(sys_name, rule_id, config, factors, workload_controller)
        if line:
            lines.append(line)
    return [{"source": "workload_description", "text": "\n".join(lines)}]


def _rule_workload_line(sys_name, rule_id, config, factors, workload_controller):
    if sys_name == "mysql":
        lua_path = str(getattr(workload_controller, "lua_path", ""))
        if "durability_write" in lua_path:
            return (
                f"MYSQL_WORKLOAD_EVIDENCE {rule_id} active: InnoDB write transaction workload issues "
                "BEGIN/COMMIT and writes rows; durability-related binlog and redo-log settings are on the execution path."
            )
    if sys_name == "postgresql":
        lua_path = str(getattr(workload_controller, "lua_path", ""))
        if rule_id == "postgres_work_mem_temp_buffers_low_pair" and "sort_spill" in lua_path:
            return (
                "POSTGRES_WORKLOAD_EVIDENCE postgres_work_mem_temp_buffers_low_pair active: "
                "sort/hash/temp-table workload is selected, so work_mem and temp_buffers are eligible to be activated."
            )
        if rule_id == "postgres_commit_delay_always_wait":
            threads = _safe_int(factors.get("threads"), 0)
            commit_siblings = _safe_int(config.get("commit_siblings"), 0)
            if threads > commit_siblings:
                return (
                    "POSTGRES_WORKLOAD_EVIDENCE postgres_commit_delay_always_wait concurrency_satisfies_commit_siblings: "
                    f"threads={threads} exceeds commit_siblings={commit_siblings}; group commit delay condition can be reached."
                )
            return (
                "POSTGRES_WORKLOAD_EVIDENCE postgres_commit_delay_always_wait concurrency_below_commit_siblings: "
                f"threads={threads} does not exceed commit_siblings={commit_siblings}; group commit delay condition is unlikely."
            )
    if sys_name == "httpd":
        if rule_id == "httpd_keepalive_mkar_coupling":
            return (
                "HTTPD_WORKLOAD_EVIDENCE httpd_keepalive_mkar_coupling active: HTTP keep-alive workload reuses "
                "persistent connections; MaxKeepAliveRequests controls how many requests each kept-alive connection may serve."
            )
        if rule_id == "httpd_keepalive_timeout_coupling":
            return (
                "HTTPD_WORKLOAD_EVIDENCE httpd_keepalive_timeout_coupling active: HTTP keep-alive workload leaves "
                "persistent connections open between requests; KeepAliveTimeout controls idle lifetime."
            )
    if sys_name == "tomcat":
        if rule_id == "tomcat_request_header_size_inherits_max_http_header_size":
            return (
                "TOMCAT_WORKLOAD_EVIDENCE tomcat_request_header_size_inherits_max_http_header_size active: "
                "HTTP request workload exercises Connector request-header parsing; maxHttpRequestHeaderSize is compared with maxHttpHeaderSize."
            )
        if rule_id == "tomcat_response_header_size_inherits_max_http_header_size":
            return (
                "TOMCAT_WORKLOAD_EVIDENCE tomcat_response_header_size_inherits_max_http_header_size active: "
                "HTTP response workload exercises Connector response-header limits; maxHttpResponseHeaderSize is compared with maxHttpHeaderSize."
            )
        if rule_id == "tomcat_processor_cache_vs_max_threads":
            return (
                "TOMCAT_WORKLOAD_EVIDENCE tomcat_processor_cache_vs_max_threads active: HTTP concurrency workload exercises "
                "request processors; processorCache should scale with maxThreads."
            )
    if sys_name == "x265":
        return _x265_workload_line(rule_id, factors)
    return ""


def _x265_workload_line(rule_id, factors):
    factor_text = json.dumps(factors, sort_keys=True)
    prefix = f"X265_WORKLOAD_EVIDENCE {rule_id} active:"
    descriptions = {
        "x265_ctu_16_should_keep_rskip": (
            "video encoding workload exercises CU recursion; ctu=16 increases CU recursion and rskip controls early exit from recursion."
        ),
        "x265_rdoq_disabled_should_disable_psy_rdoq": (
            "video encoding workload exercises quantization; rdoq-level=0 disables RDOQ so psy-rdoq has no RDOQ path."
        ),
        "x265_low_rd_should_disable_psy_rd": (
            "video encoding workload exercises mode decisions; rd<3 does not use RDO-based mode decisions so psy-rd has no RDO mode-decision path."
        ),
        "x265_trellis_b_adapt_should_limit_bframes": (
            "video encoding workload exercises lookahead; b-adapt=2 enables trellis lookahead and bframes controls B-frame lookahead work."
        ),
        "x265_high_rd_should_avoid_amp": (
            "video encoding workload exercises inter analysis; rd>=5 uses expensive analysis and amp expands inter partition analysis."
        ),
        "x265_high_ref_should_limit_refs": (
            "video encoding workload exercises reference search; ref>=6 expands reference search and limit-refs controls reference analysis limiting."
        ),
        "x265_aq_disabled_should_disable_aq_strength": (
            "video encoding workload exercises adaptive quantization; aq-mode=0 disables adaptive quantization so aq-strength has no AQ path."
        ),
        "x265_full_me_should_limit_subme": (
            "video encoding workload exercises motion estimation; me=full uses exhaustive motion search and subme adds subpel refinement work."
        ),
        "x265_ctu_16_should_limit_tu_depth": (
            "video encoding workload exercises transform-unit recursion; ctu=16 constrains transform-unit depth."
        ),
        "x265_deep_tu_should_enable_limit_tu": (
            "video encoding workload exercises transform-unit recursion; tu-inter-depth>=4 enables deeper TU recursion and limit-tu controls TU recursion early exit."
        ),
    }
    return f"{prefix} factors={factor_text}; {descriptions.get(rule_id, 'video encoding workload exercises this encoder dependency.')}"


def _safe_int(value, default):
    try:
        return int(float(value))
    except Exception:
        return default
