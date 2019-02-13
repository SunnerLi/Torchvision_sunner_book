# Example 4: Tackle with CityScapes

The full program can be found [here](https://github.com/SunnerLi/Torchvision_sunner/blob/master/example/simple_cityscapes_example.py). The powerful of the ``torchvision_sunner`` is that the package can build the pallete automatically. This help you can construct the pallete for various dataset. Before you need to start your training, you should generate the pallete. First, you should create the loader which only contains labels. 

For the pallete, the intensity of color is an integer vector whose length is 3. However, we update the definition of `ToTensor` in version `19.3.15`. As the result, you should add `UnNormalize` and `Transpose` to change back to original format. Here is the example:

```python
loader = sunnerData.DataLoader(sunnerData.ImageDataset(
    root = [
        tag_folder
    ],
    transforms = transforms.Compose([
        sunnertransforms.ToTensor(),
        sunnertransforms.ToFloat(),                                         # To make sure UnNormalize can work normally
        sunnertransforms.UnNormalize(mean=[0, 0, 0], std=[255, 255, 255]),  # Remember to transfer back to [0~255] before generate pallete 
        sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC)              # Remember to transfer back to BHWC before generate pallete 
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

For the transform operators of image, you can set as usual:

```python
loader = sunnerData.DataLoader(sunnerData.ImageDataset(
    root = [
        img_folder,
        tag_folder
    ],
    transforms = transforms.Compose([
        sunnertransforms.Resize((512, 1024)),
        sunnertransforms.ToTensor(),
        sunnertransforms.ToFloat(),
        sunnertransforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),  # [0, 1] => [-1, 1]
    ])), batch_size = 32, shuffle = False, num_workers = 2
)
```