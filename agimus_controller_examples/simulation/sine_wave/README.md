You can typically run the following commands.

In one terminal:
```bash
meshcat-server
```

then in another terminal
```bash
./run_simulation.py --ocp ./ocp.yaml --u-noise-level 1 --mass-noise-level 1 --use-low-level-controller
```
