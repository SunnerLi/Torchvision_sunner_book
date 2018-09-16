# F & Q

**Q1: It's confusing to trace for every rank notice in augmentation!**

For simplify, almost every augmentations require ``BCHW`` rank format. Especially for ``Resize``. Because we use ``skimage`` package to deal with the computation, but it only accept ``BHWC`` rank format. As the result, it's recommand to use ``Resize`` at first, and then concatenate the other augmentation after that. Just like this:
```python
transforms.Compose([
    sunnerTransforms.Resize((h, w)),
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