from src.experiment import ExperimentMeta
from omegaconf import DictConfig
import hydra

@hydra.main(version_base="1.1", config_path="hydra_conf", config_name="config")
def main(opt: DictConfig) -> None:
    Experiment = ExperimentMeta.REGISTRY[opt.exp_type + "experiment"]
    experiment = Experiment(opt, **opt.config)
    experiment.run()

if __name__ == "__main__":
    main()