# CONFIGURATOR/RUN SCHEDULER
# fmt: off
from conf.script_1_config import Script1Config as CONFIG_CLASS ## IMPORT DESIRED CONFIGURATION SCHEMA HERE
import hydra, sys; from conf import create_launch, is_main, configure_run
if is_main(sys.argv): hydra.main(version_base=None, config_path="conf", config_name="conf")(lambda conf: create_launch(conf))(); sys.exit()
# fmt: on

# ACTUAL SCRIPT - running in apptainer
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from transformers import set_seed
from src.utils import set_runtime_cfg

@configure_run(CONFIG_CLASS)
def main(conf: CONFIG_CLASS):
    set_runtime_cfg(conf)
    set_seed(conf.training.seed)
    log.info(f"Running script 1 with config: {conf}")


if __name__ == "__main__":
    main()
