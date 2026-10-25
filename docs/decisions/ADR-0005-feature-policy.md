# ADR-0005: Identifier removal and the ports sensitivity analysis

Status: accepted (2026-10-25).

Flow ids, IPs, timestamps and ports are removed from the feature matrix in the primary analysis: they encode the
testbed (addresses, schedules) rather than attack behaviour. Destination port also carries real service
information, so sensitivity S2 adds it back. The comparison is reported either way and is never used to choose
the primary pipeline.
