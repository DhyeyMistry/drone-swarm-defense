# Watching the sim from your laptop

Training runs headless by default, which is what you want -- rendering slows it down.
When you want to *see* the drones, Isaac Sim can stream its viewport over WebRTC.

## On the server

```bash
/mnt/data/isaac/isaaclab-stream.sh scripts/demos/quadrupeds.py
```

or for a kit task, add `--livestream 1` to any train/play command. The script prints the
address to connect to. It looks up the server's public IP each time, so it keeps working
if the machine is restarted.

## On your laptop

Install the **Isaac Sim WebRTC Streaming Client** (the build matching Isaac Sim 5.1) from
NVIDIA's Isaac Sim download page, enter the server address, and press Connect.

## Two traps

1. **Firewall.** The streaming ports must be open to your IP in the instance's AWS
   security group: **TCP 49100** and **UDP 47998**. Home IP addresses change; if the stream
   stops connecting, check yours first (`curl https://checkip.amazonaws.com`) and ask the
   organizer to update the rule.
2. **Do not launch the client from a VS Code terminal.** VS Code sets
   `ELECTRON_RUN_AS_NODE=1`, which makes the client start as a plain Node process and exit
   with no window. Launch it from Spotlight, Finder, or the Start menu instead.
