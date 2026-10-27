# dashboard: watch training from any computer

Serves the desktop's TensorBoard on the web through a Cloudflare tunnel. Nothing is opened
on the router: `cloudflared` dials out to Cloudflare, and TensorBoard listens on 127.0.0.1.

```bash
cd simulation_infra/dashboard
./serve.sh            # prints "Public dashboard: https://<random>.trycloudflare.com" (also in URL.txt)
./stop.sh
```

What you see (`AS-Tello-Room-v0`, under **Scalars**, `Episode/`):

| Scalar | Meaning |
|---|---|
| `success_rate` | share of episodes where all 4 drones landed within 0.3 m of their spots |
| `duration_s` | time one environment's episode lasted (sim seconds) |
| `drones_landed`, `drones_landed_on_spot`, `drones_reached_waypoint`, `collision_rate`, `crash_rate`, `tag_losses`, `tag_fix_frac`, `spot_err_m`, `min_separation_m` | the rest of the outcome |

Under `Rewards/`, one graph per reward term, each the **average per drone, per episode**
(the term summed over an episode's steps), so you can see what drives learning:

| Scalar | Term |
|---|---|
| `avg_closer_reward_per_episode` | +10 per metre gained toward the drone's waypoint (1.5 m above its spot) until it reaches it; 0 after |
| `avg_landing_loss_per_episode` | -0.5 per step not landed on the spot; a drone that ends early (crash, landed off the spot) pays the rest of the episode |
| `avg_no_tag_loss_per_episode` | -0.5 per step a flying drone has no AprilTag in view |
| `avg_crash_loss_per_episode` | -100 per contact / crash / leaving the room |
| `avg_separation_loss_per_episode` | closer than 0.5 m to another drone |
| `avg_action_rate_loss_per_episode` | jerky sticks |
| `avg_land_accuracy_reward_per_episode` | up to +100 at touchdown, 0 at 0.5 m or more off the spot |
| `avg_formation_reward_per_episode` | +200 when all four landed on their spots |
| `avg_total_per_episode` | the sum of all of the above |

## Who can see it

The quick tunnel needs no account, but its URL is public: anyone who has the link can view
(read-only) the training curves. It changes on every `./serve.sh`.

For a fixed address behind a login: in the Cloudflare dashboard create a tunnel
(Zero Trust -> Networks -> Tunnels), point a public hostname at `http://127.0.0.1:6006`, add a
Cloudflare Access application on that hostname (e.g. one-time PIN to your email), then:

```bash
CF_TUNNEL_TOKEN=<token from the dashboard> ./serve.sh
```

`cloudflared` lives in `~/.local/bin/` (official release binary).


## Render page (videos of the simulation)

`./render_serve.sh` starts `render_loop.py`, which records one episode of the newest
checkpoint (at most every 30 min, `--min_gap`), plus a page on `127.0.0.1:8790`
(`render_site/`) and a second quick tunnel. The URL is written to `RENDER_URL.txt`.
`./render_stop.sh` stops all three. Logs are in `run/render_loop.log` and
`run/render_record.log` (the last recording).
