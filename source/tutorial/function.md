# torchvision_sunner.transforms.function

This pages shows the parameters of the function which ``torchvision_sunner`` provide. Unlike the augmentation class, these function call be call directly. Also, you just need to import for single module:
```python
import torchvision_sunner.transforms as sunnertransforms
```

## torchvision_sunner.transforms.function.asImg [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/function.py#L17)]

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

## torchvision_sunner.utils.quiet [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/utils.py#L9)]

This function can mute all information preview in the terminal. Here is the usage:
```python
sunnertransforms.quiet()
```

## torchvision_sunner.transforms.function.show [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/function.py#L58)]

This function utilize the previous function, and show the image directly. This function may be appropriatly used after the final result of network. You should notice the value range should locate in [-1, 1]. Besides, the rank format of input tensor should be ``BCHW``. Here is the usage:
```python
# Do the forwarding first
img = net(data)

# Show it in 2 * 4 grid
show(img[:8], row=2, column=4)
```

#### Parameters
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 
* **row (Int) -** The number of row you want to extend
* **column (Int) -** The number of column you want to extend
* **title (Int) -** The title string in the demonstration image
* **sec (Int) -** The time you want to pause. Set -1 if you want to wait until press any key