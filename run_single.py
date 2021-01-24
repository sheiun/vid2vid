from generate import generate

if __name__ == "__main__":
    name = "o2d_256"
    # TODO: make every test set into one folder
    kwargs = {
        "dataset_mode": "pose",
        "ngf": 64,
        "input_nc": 6 if "od" in name else 3,
        "openpose_only": False if "od" in name else True,
        "resize_or_crop": "scaleHeight",
        "use_real_img": True,
        "name": name,
        "dataroot": "datasets/test_1041_o2d",
        "loadSize": int(name.partition("_")[-1]),
        "no_flow": "od" in name,
        "which_epoch": 3,
        "results_dir": "datasets/test_1041_o2d/results",  # "./results/" + name,
    }
    generate(kwargs)
