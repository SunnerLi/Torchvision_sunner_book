# torchvision_sunner.transforms.function

This pages shows the parameters of the function which ``torchvision_sunner`` provide. Unlike the augmentation class, these function call be call directly. Also, you just need to import for single module:
```python
import torchvision_sunner.transforms as sunnertransforms
```

## torchvision_sunner.transforms.function.asImg [[source](https://github.com/SunnerLi/Torchvision_sunner2/blob/master/torchvision_sunner/transforms/function.py#L11)]

This function can transfer the computed tensor into numpy image directly, and resize the image into ideal size. This function may be appropriatly used after the final result of network. You should notice the value range should locate in [-1, 1]. Besides, the rank format of input tensor should be ``BCHW``. Here is the usage:
```python
import cv2
rendered_img = asImg(net_output_img, [320, 640])
cv2.imshow('show_img_window', rendered_img)
cv2.waitKey()
```

#### Parameters
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 
* **size (tuple or list) -** The size of output image

## torchvision_sunner.utils.quiet [[source](https://github.com/SunnerLi/Torchvision_sunner2/blob/master/torchvision_sunner/utils.py#L3)]

This function can mute all information preview in the terminal. Here is the usage:
```python
sunnertransforms.quiet()
```