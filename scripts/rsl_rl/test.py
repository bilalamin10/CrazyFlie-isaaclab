# reset environment
obs, _ = env.get_observations()
timestep = 0

# --- Eval metrics accumulators ---
metric_keys = [
    "Metrics/tracking_err_mean",
    "Metrics/tracking_err_p95",
    "Metrics/success_rate",
]
metric_sums = {k: 0.0 for k in metric_keys}
metric_counts = {k: 0 for k in metric_keys}
episodes_completed = 0
num_envs = env.unwrapped.num_envs
prev_dones = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)

# simulate environment
while simulation_app.is_running():
    start_time = time.time()
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, extras = env.step(actions)

    # --- Accumulate metrics from env.extras (logged by _log_eval_metrics) ---
    log_dict = extras.get("log", {}) if isinstance(extras, dict) else {}
    for k in metric_keys:
        if k in log_dict:
            v = log_dict[k]
            metric_sums[k] += float(v)
            metric_counts[k] += 1

    # # --- Count episodes (rising edges of done flags) ---
    # if isinstance(dones, torch.Tensor):
    #     new_dones = dones & ~prev_dones
    #     episodes_completed += int(new_dones.sum().item())
    #     prev_dones = dones.clone()

    # # --- Stop if we've collected enough episodes ---
    # if args_cli.num_episodes is not None and episodes_completed >= args_cli.num_episodes:
    #     print(f"[INFO] Reached {episodes_completed} episodes, stopping eval.")
    #     break

    if args_cli.video:
        timestep += 1
        if timestep == args_cli.video_length:
            break

    sleep_time = dt - (time.time() - start_time)
    if args_cli.real_time and sleep_time > 0:
        time.sleep(sleep_time)

# --- Write metrics JSON ---
if args_cli.metrics_out is not None:
    import json
    means = {
        k: (metric_sums[k] / metric_counts[k]) if metric_counts[k] > 0 else float("nan")
        for k in metric_keys
    }
    means["num_episodes"] = episodes_completed
    means["num_envs"] = num_envs
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics_out)), exist_ok=True)
    with open(args_cli.metrics_out, "w") as f:
        json.dump(means, f, indent=2)
    print(f"[INFO] Saved metrics to {args_cli.metrics_out}")
    print(json.dumps(means, indent=2))

# close the simulator
env.close()