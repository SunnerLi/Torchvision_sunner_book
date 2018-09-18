# Example 3: Deal with Categorical Data

The full program can be found [here](https://github.com/SunnerLi/Torchvision_sunner/blob/master/example/simple_categorical_example.py). Several techniques are list below:

* We can create multiple ``data.Dataset`` object with different augmentation sequence. For example, for the usual image:
```python
img_dataset = sunnerData.ImageDataset(
    root = [
        ['./Datasets/Ear-Pen/train/img'], 
    ],
    transform = transforms.Compose([
        sunnertransforms.Resize((260, 195)),
        sunnertransforms.ToTensor(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize(),
    ]), save_file = False
)
```
However, we can add ``CategoricalTranspose`` for the label domain:
```python
tag_dataset = sunnerData.ImageDataset(
    root = [
        ['./Dataset/Ear-Pen/train/tag']
    ],
    transform = transforms.Compose([
        sunnertransforms.Resize((260, 195)),
        sunnertransforms.ToTensor(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize(),

        # Add the new augmentation method
        sunnertransforms.CategoricalTranspose(
            pallete = pallete, 
            direction = sunnertransforms.COLOR2INDEX, 
            index_default = 0
        )
    ])
)
```

* The ``MultiLoader`` can combine different datasets as single loader:
```python
loader = sunnerData.MultiLoader(
    datasets = [img_dataset, tag_dataset], 
    batch_size = 1, 
    shuffle = False, 
    num_workers = 2
)
```