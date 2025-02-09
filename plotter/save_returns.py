from typing import Optional, List
import pickle
import os

def gen_filepth(
    model_name: str,
    environment_name: str,
    additional_name: Optional[str],
    save_dir: str = "./data/returns/"
) -> str:
    if additional_name:
        fn = '_'.join([model_name, environment_name, additional_name]) + ".pickle"
    else:
        fn = '_'.join([model_name, environment_name]) + ".pickle"
    return os.path.join(save_dir, fn)

def save_returns(returns: List[float], file_name: str):
    serialized = pickle.dumps(returns)
    with open(file_name, 'wb') as f:
        f.write(serialized)