# Reference simulation evidence

Author: Kazmir Fahrier

These measurements were generated on September 7, 2026 with Python 3.12.14 on macOS using the dependency versions recorded in `reference_summary.json`. The CSVs preserve every run, including incomplete missions and contacts.

The summary records the base Git revision and `working_tree_dirty: true` because validation ran before the repair commit was created. Its `source_sha256` identifies the actual Python source used. The README check recomputes that fingerprint and rejects evidence from different source. Output paths in the summary describe the original local run directory; the CSVs are archived here, while the mission illustration is in the parent documentation directory. Fresh CI measurements are retained as workflow artifacts for 14 days.

The default seed 7 mission serviced seven of seven goals with no contacts. Across the five default seeds, mean requested goal completion was 74.3 percent with one worker contact: seed 2 ended on contact and seed 3 exhausted daylight after five goals.

In the larger thirty row experiment, nineteen of twenty runs serviced all ten priority goals. Seed 71 serviced three goals before a worker contact. Mean completion was 96.5 percent, with mean selected pose error of 0.064 m. These results establish bounded simulation behavior, not a guarantee of collision avoidance or field performance.

To verify the README against this archive, run:

```bash
python docs/update_readme_kpis.py --summary docs/validation/reference_summary.json --check
```
