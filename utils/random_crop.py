
import numpy as np
from skimage.util.shape import view_as_windows

def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs,
                              (1, output_size, output_size, 1))[..., 0, :, :,
                                                                0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


if __name__ == '__main__':

    from PIL import Image
    import cv2

    img = cv2.imread("../test_data/sep_imgs/highspeed+cloudy-rgb.jpg")
    # cv2.imshow("111", img)
    # cv2.waitKey(0)
    print(img.shape)
    img = img[np.newaxis, ...]
    print(img.shape)
    img = np.transpose(img, (0, 3, 1, 2))
    print(img.shape)
    new_img = random_crop(img, 300)
    print(new_img.shape)
    new_img = new_img.squeeze()
    print(new_img.shape)

    cv2.imshow("original", np.transpose(img.squeeze(), (1, 2, 0)))
    cv2.imshow("cropped", np.transpose(new_img, (1, 2, 0)))
    cv2.waitKey(0)


