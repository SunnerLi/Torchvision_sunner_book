# Example 4: Tackle with CityScapes

The full program can be found [here](https://github.com/SunnerLi/Torchvision_sunner/blob/master/example/simple_cityscapes_example.py). The powerful of the ``torchvision_sunner`` is that the package can build the pallete automatically. This help you can construct the pallete for various dataset. Before you need to start your training, you should generate the pallete. First, you should create the loader which only contains labels:
```python
loader = sunnerData.DataLoader(sunnerData.ImageDataset(
    root = [
        tag_folder
    ],
    transform = transforms.Compose([
        sunnertransforms.ToTensor(),
    ])), batch_size = 2, shuffle = False, num_workers = 2
)
```

Then you can generate the pallete by this:
```python
pallete = sunnertransforms.getCategoricalMapping(loader, path = 'cityscapes_pallete.json')[0]
```

So once you want to train the segmentation model, you can use this pallete to construct the augmentation operator!
```python
goto_op = sunnertransforms.CategoricalTranspose(
    pallete = pallete, 
    direction = sunnertransforms.COLOR2ONEHOT
)
```