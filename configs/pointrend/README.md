# Usage

Add this line to `def setup` in the regular train script

```diff
+ from point_rend import add_pointrend_config

... # Code

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
+   add_pointrend_config(cfg)
    cfg.merge_from_list(args.opts)

... # Code

```

Adjust your import statement's path with respect to your train script location. For example, if the point_rend code is placed under a folder named `baselines` at the folder's root, the import must read `from baselines.point_rend import add pointrend_config`. 