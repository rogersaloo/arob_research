from PIL import Image, ImageStat

def is_grayscale(path):

    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)

    if sum(stat.sum)/3 == stat.sum[0]:
        return True
    else:
        return False

is_grayscale("00000005_001.png")
