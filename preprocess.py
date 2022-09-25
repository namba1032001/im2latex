
from os.path import join
import argparse

from PIL import Image
from torchvision import transforms
import torch
import torchvision

def preprocess(data_dir, split):
    assert split in ["train", "validate", "test"]

    print("Process {} dataset...".format(split))
    images_dir = join(data_dir, "formula_images_processed")

    formulas_file = join(data_dir, "im2latex_formulas.norm.lst")
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    split_file = join(data_dir, "im2latex_{}_filter.lst".format(split))
    pairs = []
    transform = transforms.ToTensor()

    with open(split_file, 'r') as f:
        i = 0
        for line in f:
            formula_id ,img_name, temp = line.strip('\n').split()
            img_name = str(img_name) + '.png'
            # load img and its corresponding formula
            img_path = join(images_dir, img_name)
            # img_tensor = torchvision.io.read_image(img_path)
            # img_tensor = img_tensor.resize_(200,160)
            img = Image.open(img_path)
            img_tensor = transform(img)
            formula = formulas[int(formula_id)]
            # print(img_tensor.shape)
            pair = (img_tensor, formula)
            # print(type(img_tensor))
            pairs.append(pair)
            if((split == "train") & (i == 10000)):
                break;
            elif(((split == "validate") | (split == "test")) & (i == 1000)):
                break;
            i+=1

        pairs.sort(key=img_size)

    out_file = join(data_dir, "{}.pkl".format(split))
    torch.save(pairs, out_file)
    print("Save {} dataset to {}".format(split, out_file))


def img_size(pair):
    img, formula = pair
    return tuple(img.size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Im2Latex Data Preprocess Program")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    args = parser.parse_args()

    splits = ["validate", "test", "train"]
    for s in splits:
        preprocess(args.data_path, s)
