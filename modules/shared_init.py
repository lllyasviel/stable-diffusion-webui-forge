import os

import torch

from modules import shared
from modules.shared import cmd_opts


def initialize():
    """Initializes fields inside the shared module in a controlled manner.

    Should be called early because some other modules you can import mingt need these fields to be already set.
    """
    from modules import timer
    startup_timer = timer.startup_timer

    with startup_timer.subcategory("shared initialization"):
        os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)
        startup_timer.record("create hypernetwork directory")

        from modules import options, shared_options
        shared.options_templates = shared_options.options_templates
        shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
        shared.restricted_opts = shared_options.restricted_opts
        try:
            shared.opts.load(shared.config_filename)
        except FileNotFoundError:
            pass
        startup_timer.record("load options")

        from modules import devices
        shared.device = devices.device
        shared.weight_load_location = None if cmd_opts.lowram else "cpu"
        startup_timer.record("setup devices")

        from modules import shared_state
        shared.state = shared_state.State()
        startup_timer.record("initialize state")

        from modules import styles
        shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)
        startup_timer.record("load prompt styles")

        from modules import interrogate
        shared.interrogator = interrogate.InterrogateModels("interrogate")
        startup_timer.record("initialize interrogator")

        from modules import shared_total_tqdm
        shared.total_tqdm = shared_total_tqdm.TotalTQDM()
        startup_timer.record("initialize tqdm")

        from modules import memmon, devices
        shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
        shared.mem_mon.start()
        startup_timer.record("start memory monitor")

