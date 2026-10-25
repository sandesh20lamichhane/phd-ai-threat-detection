# ADR-0003: "Attempted" flows in the corrected CIC releases

Status: accepted (2026-10-25).

The corrected CICIDS2017 and CSE-CIC-IDS2018 releases mark flows produced by an attack tool that never carried
the attack (for example a handshake with no payload) as "<attack> - Attempted". At the flow level these look
benign. Primary analysis: excluded from training and evaluation, and counted. Sensitivity S3: relabelled benign.
Counting them as attacks would reward a model for memorising tool artefacts.
