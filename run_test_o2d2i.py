from glob import glob
from time import time

from tqdm import tqdm

from generate import generate

if __name__ == "__main__":
    start = time()
    name = "od2i_512"
    size = int(name.partition("_")[-1])

    test_dir = f"test_pte_o2d2i"  # test_o2d2i | test_pte_o2d2i
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
            "results_dir": f"./results/o2d2i_512_pte/{idx}/",  # o2d2i_512 | o2d2i_512_pte
        }
        generate(kwargs)
    print("Time:", (time() - start) / len(roots), name)

    # Time: 6.636090524494648 od2i_256
