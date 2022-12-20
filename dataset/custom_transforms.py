""" Multi-version of torch.transforms """
import torch
import torchvision
import torchvision.transforms as tf
import torchvision.transforms.functional as F
class ComposeWithMultiTensor(tf.Compose):
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __call__(self, tensor_list):
        for t in self.transforms:
            tensor_list = t(tensor_list)
        return tensor_list

class RandomApplyWithMultiTensor(tf.RandomApply):
    """Apply randomly a list of transformations with a given probability.
        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """
    def forward(self, tensor_list):
        if self.p < torch.rand(1):
            return tensor_list
        for t in self.transforms:
            tensor_list = t(tensor_list)
        return tensor_list

class RandomResizedCropWithMultiTensor(tf.RandomResizedCrop):
    def forward(self, tensor_list):
        img = tensor_list[0]
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        return list(map(
            lambda x: F.resized_crop(x, i, j, h, w, self.size, self.interpolation)
            , tensor_list))

class RandomHorizontalFlipWithMultiTensor(tf.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def forward(self, tensor_list):
        if torch.rand(1) < self.p:
            return list(map( lambda x: F.hflip(x), tensor_list))
        return tensor_list

class RandomVerticalFlipWithMultiTensor(tf.RandomVerticalFlip):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def forward(self, tensor_list):
        if torch.rand(1) < self.p:
            return list(map( lambda x: F.vflip(x), tensor_list))
        return tensor_list

class RandomGrayscaleWithMultiTensor(tf.RandomGrayscale):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    """
    def forward(self, tensor_list):
        def rgb_to_gray(img):
            """ Returns:
            PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
            with probability (1-p).
            - If input image is 1 channel: return 
            - If input image is 3 channel: grayscale version is 3 channel with r == g == b
            """
            num_output_channels = F._get_image_num_channels(img)
            if num_output_channels ==1:
                return img 
            return F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        if torch.rand(1) < self.p:
            return list(map( lambda x: rgb_to_gray(x), tensor_list))
        return tensor_list    

if __name__ == '__main__':
    from PIL import Image
    img = Image.open('/workspace/inpaint_mask/data/warpData/CIHP/Training/tps_dgrid_p16/origin/0038267.jpg')
    img_tensor = tf.ToTensor()(img)
    f = ComposeWithMultiTensor([
        RandomGrayscaleWithMultiTensor(p=1.0),
        RandomVerticalFlipWithMultiTensor(p=1.0),
        RandomHorizontalFlipWithMultiTensor(p=1.0),
        RandomApplyWithMultiTensor([RandomResizedCropWithMultiTensor(size= (256,256))], p=1.0)
    ])
    result_list = f([img_tensor])
    to_pillow_f = tf.ToPILImage()
    img.save('./sample.jpg')
    to_pillow_f(result_list[0]).save('./sample_result.jpg')