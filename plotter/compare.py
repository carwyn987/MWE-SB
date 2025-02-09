import matplotlib.pyplot as plt
import argparse
import pickle
import os
import numpy as np

def moving_average(data, window_size=20):
    if window_size < 1 or len(data) <= window_size:
        return data  # No smoothing if window_size is invalid
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

def compare_all(returns_dir: str, save_dir: str, smoothed: bool):
    # Load all
    files = os.listdir(returns_dir)
    all_data = {}
    for filename in files:
        try:
            with open(os.path.join(returns_dir, filename), 'rb') as f:
                return_list = pickle.load(f)
            if isinstance(return_list, list) and len(return_list) > 0:
                all_data[filename] = return_list
            else:
                raise Exception(f"Error loading file {filename}")
        except Exception as err:
            print(f"Unexpected {err=} while loading returns, {type(err)=}")
    
    if len(all_data) <= 0:
        raise Exception("No data loaded")
    
    fig, ax = plt.subplots(figsize=(plt.rcParams["figure.figsize"][0] * 2, plt.rcParams["figure.figsize"][1] * 2))
    # Maximize the window if supported by the backend
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()  # For Qt-based backends
    except AttributeError:
        pass  # Not all backends support window maximization

    for filename, returns in all_data.items():
        if smoothed:
            returns = moving_average(returns)
        ax.plot(returns, label=filename.split('.')[0])
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Normalized Return")
    ax.set_title("Model Comparison")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "comparison.png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns_dir", default="../data/returns/")
    parser.add_argument("--save_dir", default="../data/media/")
    parser.add_argument("--smoothed", action="store_true", help="Enable smoothing")
    args = parser.parse_args()

    compare_all(args.returns_dir, args.save_dir, args.smoothed)