# F & Q

**Q1: It's confusing to trace for every rank notice in augmentation!**

For simplify, almost every augmentations require ``BCHW`` rank format. Especially for ``GrayStack``. Because ``Transpose`` function cannot support the gray-scale tensor. The reason is that we cannot distinguish [ABC] is the rank of ``BHWC`` in image or ``BTHW`` in gray-scale video, and most of other function will use ``Transpose`` function indirectly. As the result, we should force the user to transfer the image into RGB format at first. You should stack the operation just like this:

```python
transforms.Compose([
    sunnerTransforms.GrayStack(sunnerTransforms.BHW2BHWC),
    sunnerTransforms.ToTensor(),
    sunnerTransforms.Transpose(sunnerTransforms.BHWC2BCHW),
    # other augmentation...
])
```

**Q2: Which augmentations only accept the value range of [-1, 1]?**

The ``CategoricalTranspose`` is the only augmentation which need to do the normalization first. We recommand you to call this function at last. For example:
```python
transforms.Compose([
    # Do something augmentation first
    sunnerTransforms.Normalize(),
    sunnerTransforms.CategoricalTranspose(sunnerTransforms.COLOR2ONEHOT)
])
```

**Q3: Why the Normalize function cannot work sometimes?**

After you read the image, the type of value in the array is ``np.uint8``. The pytorch function cannot divide the interger type. As the result, we recommand you call the ``ToFloat`` function before you call the ``Normalize`` function. For example:
```python
transforms.Compose([
    sunnerTransforms.ToTensor(),
    sunnerTransforms.ToFloat(),
    sunnerTransforms.Normalize(),
    # other augmentation...
])
```