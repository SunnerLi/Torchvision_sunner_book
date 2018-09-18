## torchvision_sunner.transforms.categorical

Categorical API is the advance topic in this package. If you want to realize the whole usage, please refer to the tutorial [here](example3.html). This pages only shows the parameters. You only need to import the same module and you can start:
```python
import torchvision_sunner.transforms as sunnertransforms
```

### torchvision_sunner.transforms.categorical.getCategoricalMapping [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/categorical.py#L70)]

This function will obtain the pallete object. Here is the usage:
```python
pallete = sunnertransforms.getCategoricalMapping(loader, path = 'pallete.json')
```

#### Parameters
* **loader (torch.utils.data.DataLoader) -** The data loader. You should create the loader first which only contains the RGB colorful label domain.
* **path (str) -**The path of the pallete record file you want to store.

#### Return
The list of pallete. In the usual usage, we only need the single pallete. Thus use ``pallete[0]`` to get the pallete object.

### torchvision_sunner.transforms.categorical.CategoricalTranspose [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/transforms/categorical.py#L116)]

Transfer the tensor into particular representation. Here is the usage:
```python
# Get the pallete object first
pallete = sunnertransforms.getCategoricalMapping(loader, path = 'pallete.json')

# Create transfer operator
goto_op = sunnertransforms.CategoricalTranspose(pallete = pallete, direction = sunnertransforms.COLOR2ONEHOT)
back_op = sunnertransforms.CategoricalTranspose(pallete = pallete, direction = sunnertransforms.ONEHOT2COLOR)

# Transfer!
label_index_format = goto_op(label)
label_color_format = back_op(label_index_format)
```

* **Notice :** You should transfer the tensor into rank format ``BCHW`` first.
* **Notice :** You should normalize the tensor into the range of [-1, 1]

#### Parameters
* **pallete (OrderDict) -** The pallete object. 
* **direction (int) -** The direction you want to transfer. We provide for six direction:
    * ``sunnertransforms.ONEHOT2INDEX``: Transfer the tensor from one-hot format into index format
    * ``sunnertransforms.INDEX2ONEHOT``: Transfer the tensor from index format into one-hot format
    * ``sunnertransforms.ONEHOT2COLOR``: Transfer the tensor from one-hot format into RGB colorful format
    * ``sunnertransforms.COLOR2ONEHOT``: Transfer the tensor from RGB colorful format into one-hot format
    * ``sunnertransforms.INDEX2COLOR``: Transfer the tensor from index format into RGB colorful format
    * ``sunnertransforms.COLOR2INDEX``: Transfer the tensor from RGB colorful format into index format
* **index_default (int) -** The index of default while the color is not found in pallete

#### Return
The tensor with corresponding format