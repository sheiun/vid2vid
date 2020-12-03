from os import makedirs

from torch.autograd import Variable

import util.util as util
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.test_options import TestOptions


def generate():
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    for i, data in enumerate(dataset):
        if data["change_seq"]:
            model.fake_B_prev = None

        _, _, height, width = data["A"].size()
        A = Variable(data["A"]).view(1, -1, opt.input_nc, height, width)
        B = Variable(data["B"]).view(1, -1, opt.output_nc, height, width)

        generated = model.inference(A, B, None)
        fake_B = util.tensor2im(generated[0].data[0])

        makedirs(opt.results_dir, exist_ok=True)
        suffix = ".jpg"
        if "2d" in opt.name:
            suffix = "_IUV.png"
        util.save_image(fake_B, f"{opt.results_dir}/{i:05}{suffix}")
        if i >= min(len(dataset) - opt.start_frame, opt.how_many) - 1:
            break


if __name__ == "__main__":
    generate()
