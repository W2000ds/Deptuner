class RuntimeEvidenceExtractor:
    def __init__(self, workload_controller, target_system, dep_manager):
        self.workload_controller = workload_controller
        self.target_system = target_system
        self.dep_manager = dep_manager

    def extract(self, violated_ids):
        detailed = self.extract_with_details(violated_ids)
        evidence = {}
        for rid, detail in detailed["details"].items():
            if detail["signal"] != "none":
                evidence[rid] = detail["signal"]
        return evidence

    def extract_with_details(self, violated_ids, extra_parts=None, include_live_sources=True):
        if not violated_ids or not self.dep_manager:
            return {"details": {}, "parts": []}

        parts = self._collect_parts(extra_parts=extra_parts, include_live_sources=include_live_sources)
        generic_support = (
            "warning", "warn", "error", "fatal", "timeout", "timed out",
            "crash", "failed", "failure", "degraded", "out of", "cannot"
        )
        generic_contradict = (
            "disabled", "not used", "ignored", "skip", "skipped", "bypass"
        )

        details = {}
        for rule in self.dep_manager.rules:
            rid = rule["id"]
            if rid not in violated_ids:
                continue

            support_keywords = [str(x).lower() for x in rule.get("support_keywords", [])]
            contradict_keywords = [str(x).lower() for x in rule.get("contradict_keywords", [])]
            use_generic_support = not support_keywords
            use_generic_contradict = not contradict_keywords

            matched_support = []
            matched_contradict = []
            snippets = []
            source_names = set()

            for part in parts:
                source = str(part.get("source", "runtime"))
                text = str(part.get("text", ""))
                lowered = text.lower()
                support_hits = self._matched_keywords(lowered, support_keywords)
                contradict_hits = self._matched_keywords(lowered, contradict_keywords)
                if use_generic_support and self._contains_any(lowered, generic_support):
                    support_hits.append("__generic_support__")
                if use_generic_contradict and self._contains_any(lowered, generic_contradict):
                    contradict_hits.append("__generic_contradict__")

                if support_hits or contradict_hits:
                    source_names.add(source)
                    matched_support.extend([kw for kw in support_hits if kw != "__generic_support__"])
                    matched_contradict.extend([kw for kw in contradict_hits if kw != "__generic_contradict__"])
                    for kw in support_hits + contradict_hits:
                        if kw.startswith("__generic_"):
                            continue
                        snippets.append(
                            {
                                "source": source,
                                "keyword": kw,
                                "excerpt": self._excerpt(text, kw),
                            }
                        )

            has_support = bool(matched_support) or (use_generic_support and any(
                self._contains_any(str(part.get("text", "")).lower(), generic_support) for part in parts
            ))
            has_contradict = bool(matched_contradict) or (use_generic_contradict and any(
                self._contains_any(str(part.get("text", "")).lower(), generic_contradict) for part in parts
            ))

            signal = "none"
            if has_support and not has_contradict:
                signal = "support"
            elif has_contradict and not has_support:
                signal = "contradict"

            details[rid] = {
                "signal": signal,
                "matched_support_keywords": sorted(set(matched_support)),
                "matched_contradict_keywords": sorted(set(matched_contradict)),
                "source_names": sorted(source_names),
                "snippets": snippets[:12],
            }

        return {"details": details, "parts": parts}

    def _collect_parts(self, extra_parts=None, include_live_sources=True):
        parts = []
        if include_live_sources:
            for attr, source in (("output", "workload_stdout"), ("error", "workload_stderr")):
                value = getattr(self.workload_controller, attr, None)
                if value is None:
                    continue
                try:
                    text = value.decode("utf-8", errors="ignore")
                except Exception:
                    text = str(value)
                if text:
                    parts.append({"source": source, "text": text})

            if hasattr(self.target_system, "collect_runtime_evidence"):
                try:
                    system_parts = self.target_system.collect_runtime_evidence() or []
                    parts.extend(self._normalize_parts(system_parts, default_source="system"))
                except Exception:
                    pass

        if extra_parts:
            parts.extend(self._normalize_parts(extra_parts, default_source="extra"))

        return parts

    @staticmethod
    def _normalize_parts(raw_parts, default_source):
        normalized = []
        for part in raw_parts:
            if part is None:
                continue
            if isinstance(part, dict):
                text = str(part.get("text", ""))
                if text:
                    normalized.append({"source": str(part.get("source", default_source)), "text": text})
            else:
                text = str(part)
                if text:
                    normalized.append({"source": default_source, "text": text})
        return normalized

    @staticmethod
    def _matched_keywords(text, keywords):
        return [kw for kw in keywords if kw and kw in text]

    @staticmethod
    def _excerpt(text, keyword, window=120):
        lowered = text.lower()
        idx = lowered.find(keyword.lower())
        if idx < 0:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        return text[start:end].replace("\n", " ").strip()

    @staticmethod
    def _contains_any(text, keywords):
        for kw in keywords:
            if kw and kw in text:
                return True
        return False
