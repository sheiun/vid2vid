from glob import glob
from time import time

from tqdm import tqdm

from generate import generate

if __name__ == "__main__":
    start = time()
    name = "o2d_512"
    size = int(name.partition("_")[-1])

    # test_dir = "test" if "2i" in name else "test_o2d"
    test_dir = "test"
    roots = glob(f"datasets/{test_dir}/*")
    for root in tqdm(roots):
        idx = root.split("/")[-1]
        kwargs = {
            "dataset_mode": "pose",
            "ngf": 64,
            "input_nc": 6 if "od" in name else 3,
            "resize_or_crop": "scaleHeight",
            "use_real_img": True,
            "name": name,
            "dataroot": root,
            "loadSize": size,
            "n_scales_spatial": size // 256,
            "openpose_only": False if "od" in name else True,
            "no_flow": "od" in name,
            "which_epoch": 3,
            "results_dir": f"./results/{name}/{idx}/",
            # "results_dir": f"./results/nxo2d_256/{idx}/",
        }
        generate(kwargs)
    print("Time:", (time() - start) / len(roots), name)
    # o2i_256, Time: 16.512991401553155
    # od2i_256, Time: 12.67669531404972
    # o2d_256, Time: 6.1980992808938025
    # od2i_256_ours, Time: 6.591896048188209 + 0.35719615817070005 (no resize) + 6.1980992808938025
