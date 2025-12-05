
import jsonargparse

from crossgoose.data_utils import make_dataset, make_synth_dataset
from crossgoose.eval import eval_models



if __name__ == "__main__":
    jsonargparse.auto_cli(
        [make_dataset, make_synth_dataset, eval_models],
        as_positional=False)
