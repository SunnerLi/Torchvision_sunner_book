# Constant

This pages shows some constants which are defined in ``torchvision_sunner``. You can import the module and call like this:
```python
import torchvision_sunner.transforms as sunnertransforms
print(sunnertransforms.UNDER_SAMPLING)
>> 0
``` 

### Numeric constant

* ``UNDER_SAMPLING`` - The numeric value of this constant is ``0``. This constant represent the under sampling toward the data un-balance. You can refer to [here](data.html#class-torchvision-sunner-data-imagefolder-source) for further usage
* ``OVER_SAMPLING`` - The numeric value of this constant is ``1``. This constant represent the over sampling toward the data un-balance. You can refer to [here](data.html#class-torchvision-sunner-data-imagefolder-source) for further usage
* ``BCHW2BHWC`` - The numeric value of this constant is ``0``. This constant represent to transfer the rank of tensor from ``BCHW`` to ``BHWC``. This constant may be used in [here](transforms.html#torchvision-sunner-transforms-transpose-source)
* ``BHWC2BCHW`` - The numeric value of this constant is ``1``. This constant represent to transfer the rank of tensor from ``BHWC`` to ``BCHW``. This constant may be used in [here](transforms.html#torchvision-sunner-transforms-transpose-source)

### Categorical constant

These constant may be used in [here](categorical.html#torchvision-sunner-transforms-categorical-categoricaltranspose-source)

* ``ONEHOT2INDEX`` - This constant represent the symbol to transfer the tensor from the one-hot form into index form
* ``INDEX2ONEHOT`` - This constant represent the symbol to transfer the tensor from the index form into one-hot form
* ``ONEHOT2COLOR`` - This constant represent the symbol to transfer the tensor from the one-hot form into RGB colorful form
* ``COLOR2ONEHOT`` - This constant represent the symbol to transfer the tensor from the RGB colorful form into one-hot form
* ``INDEX2COLOR `` - This constant represent the symbol to transfer the tensor from the index form into RGB colorful form
* ``COLOR2INDEX `` - This constant represent the symbol to transfer the tensor from the RGB colorful form into index form