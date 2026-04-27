# TerraScout Design Notes

These notes document the six autonomy layers described in the project plan. They are written
as reviewable Markdown sources and can be rendered to PDFs with:

```bash
python docs/design/render_design_pdfs.py
```

The renderer writes PDF copies to `docs/design/pdf/`.

## Layer Notes

- [L0 PID Control](l0_pid_control.md)
- [L1 Kalman Tracking](l1_kalman_tracking.md)
- [L2 MCL Localization](l2_mcl_localization.md)
- [L3 EKF-SLAM Mapping](l3_ekf_slam.md)
- [L4 Hybrid A* Planning](l4_hybrid_astar.md)
- [L5 MDP Scheduling](l5_mdp_scheduler.md)

## References

The implementation follows standard public robotics references: Astrom and Murray for PID
feedback, Thrun/Burgard/Fox for probabilistic robotics and MCL, Durrant-Whyte and Bailey for
EKF-SLAM, Dolgov et al. for Hybrid A*, LaValle for graph search and planning, and Russell and
Norvig plus Sutton and Barto for MDP/value-iteration foundations.
