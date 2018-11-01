# torchvision_sunner.transforms

This module define the augmentation functions. In this page, we will intorduce the parameters one by one! On the other hand, we also provide the API to deal with categorical data. You can refer the parameters and usage in [here](categorical.html). But you just need to import the library as following:
```python
import torchvision_sunner.transforms as sunnertransforms
```
In default, the whole augmentations are covered as the class interface. So you should create the operation instance first, and use it to do the augmentation next. The ``torchvision_sunner.transforms`` also defines some function that the user can call directly. You can check [here](function.html) for detail.

## torchvision_sunner.transforms.OP [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/base.py#L10)]

There two kinds of operators toward augmentations. The first one is simple category. The operation is the same with arbitrary rank format or input type. The other one is complex category, we should consider for different cases toward different rank format or input type. This function provide an interface to deal with complex operation. The instance of this function cannot be create directly. But you can inherit this class, and create the new augmentation class in the future. 

### torchvision_sunner.transforms.OP.\_\_call\_\_ 

This function define the process while the instance is called. 

#### Parameters
* **tensor (arbitrary) -** The tensor you want to deal with. The type of this parameter can be ``torch.Tensor`` or ``np.ndarray``. Also, the length of tensor rank can be 4 or 5. 

## torchvision_sunner.transforms.ToTensor [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/simple.py#L13)]

Change the type of tensor as ``torch.Tensor``. If the type of input is already ``torch.Tensor``, the function will return the parameters directly. Here is the usage:
```python
# Use it uniquely
op = sunnertransforms.ToTensor()
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.ToTensor()
])
input = op(input)
``` 

#### Parameters
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.ToFloat [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/simple.py#L28)]

Change the type of tensor as float type, which means ``torch.FloatTensor``. You should notice that this function should be called **after you wnsure the type of tensor is ``torch.Tensor``**. Here is the usage:
```python
# Use it uniquely
op = sunnertransforms.ToFloat()
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.ToTensor(),
    sunnertransforms.ToFloat()
])
input = op(input)
``` 

#### Parameters
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.Transpose [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/simple.py#L41)]

Transpose the rank format toward the given tensor. You should be careful to ensure the rank format while you use this function. Here is the usage:

```python
# Use it uniquely
op = sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW)
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW)
])
input = op(input)
``` 

#### Parameters
(constructor)
* **direction (int) -** The constant which is defined in ``torchvision_sunner.constant``. We only provide two direction now:
    * ``sunnertransforms.BHWC2BCHW``: Transfer the rank format from [batch, height, width, channel] to [batch, channel, height, width]
    * ``sunnertransforms.BCHW2BHWC``: Transfer the rank format from [batch, channel, height, width] to [batch, height, width, channel]    

(inference)
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.RandomHorizontalFlip [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/simple.py#L68)]

Flip the tensor toward horizontal direction randomly. Here is the usage:

* **Notice :** You should transfer the tensor into rank format ``BCHW`` first.

```python
# Use it uniquely
op = sunnertransforms.RandomHorizontalFlip()
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
    sunnertransforms.RandomHorizontalFlip()
])
input = op(input)
``` 

#### Parameters
* **p  (float)  -** The probability you want to flip, and the value should locate in [0.0, 1.0]

(inference)
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.RandomVerticalFlip [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/simple.py#L92)]

Flip the tensor toward vertical direction randomly. Here is the usage:

* **Notice :** You should transfer the tensor into rank format ``BCHW`` first.

```python
# Use it uniquely
op = sunnertransforms.RandomVerticalFlip()
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
    sunnertransforms.RandomVerticalFlip()
])
input = op(input)
``` 

#### Parameters
* **p  (float)  -** The probability you want to flip, and the value should locate in [0.0, 1.0]

(inference)
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.GrayStack [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/simple.py#L116)]

Stack the gray-scale image for 3 times to become RGB image. If the input is already RGB image, this function do nothing. This function also accept the input tensor whose channel is 1. Here is the usage:

```python
import torch

# Use it uniquely
op = sunnertransforms.GrayStack(sunnertransforms.BHW2BCHW)
input = torch.randn(32, 28, 28)
input = op(input)
print(input.size())
# >> [32, 3, 28, 28]

# The channel=1 case is Ok!
input = torch.randn(32, 1, 28, 28)
input = op(input)
print(input.size())
# >> [32, 3, 28, 28]

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
    sunnertransforms.GrayStack(sunnertransforms.BHW2BCHW)
])
input = torch.randn(32, 28, 28)
input = op(input)
``` 

#### Parameters
(constructor)
* **direction (int) -** The constant which is defined in ``torchvision_sunner.constant``. We only provide two direction now:
    * ``sunnertransforms.BHW2BCHW``: Transfer the rank format from [batch, height, width] to [batch, 3, height, width]
    * ``sunnertransforms.BTHW2BTCHW``: Transfer the rank format from [batch, time_step, height, width] to [batch, time_step, 3, height, width]    

(inference)
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.Resize [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/complex.py#L14)]

This function is complex operation. Resize the tensor into corresponding size. You **don't** need to normalize the tensor before you call this function. Here is the usage:

* **Notice :** You should transfer the tensor into rank format ``BCHW`` first.

```python
# Use it uniquely
op = sunnertransforms.Resize(output_size = (320, 640))
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
    sunnertransforms.Resize(output_size = (320, 640))
])
input = op(input)
``` 

#### Parameters
(constructor)
* **output_size (tuple) -** The tuple to represent the resized size. The format of tuple is ``(Height, width)``

(inference)
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.Normalize [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/complex.py#L53)]

This function is complex operation. Normalize the tensor for the given tensor. If you don't assign mean and std, then we treat the range of input tensor is [0, 255]. Here is the usage:

* **Notice :** You should transfer the tensor into rank format ``BCHW`` first.
* **Notice :** You should call ``ToFloat()`` first.

```python
# Use it uniquely
op = sunnertransforms.Normalize(mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5])
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
    sunnertransforms.ToFloat(),
    sunnertransforms.Normalize(mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5]),
])
input = op(input)
``` 

#### Parameters
(constructor)
* **mean (list) -** The mean of pixel in RGB order. You should make sure that the length of list should be the same as the channel number of given tensor. The default is ``[127.5, 127.5, 127.5]``
* **std (list) -** The std of pixel in RGB order. You should make sure that the length of list should be the same as the channel number of given tensor. The default is ``[127.5, 127.5, 127.5]``

(inference)
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.UnNormalize [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/complex.py#L93)]

This function is complex operation. Un-normalize the tensor for the given tensor. If you don't assign mean and std, then we treat the range of input tensor is [0, 255]. Here is the usage:

* **Notice :** You should transfer the tensor into rank format ``BCHW`` first.

```python
# Use it uniquely
op = sunnertransforms.UnNormalize(mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5])
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
    sunnertransforms.UnNormalize(mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5]),
])
input = op(input)
``` 

#### Parameters
(constructor)
* **mean (list) -** The mean of pixel in RGB order. You should make sure that the length of list should be the same as the channel number of given tensor. The default is ``[127.5, 127.5, 127.5]``
* **std (list) -** The std of pixel in RGB order. You should make sure that the length of list should be the same as the channel number of given tensor. The default is ``[127.5, 127.5, 127.5]``

(inference)
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 

## torchvision_sunner.transforms.ToGray [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/complex.py#L125)]

This function is complex operation. Transfer the tensor into gray-scale. Here is the usage:

* **Notice :** You should transfer the tensor into rank format ``BCHW`` first.

```python
# Use it uniquely
op = sunnertransforms.ToGray()
input = op(input)

# Use it with other augmentation
op = torchvision.transforms.Compose([
    sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
    sunnertransforms.ToGray()
])
input = op(input)
``` 

#### Parameters
* **tensor (np.ndarray or torch.Tensor) -** The tensor you want to deal with. 