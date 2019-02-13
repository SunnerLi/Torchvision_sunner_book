# Example 1: Deal with Images

The full program can be found [here](https://github.com/SunnerLi/Torchvision_sunner/blob/master/example/simple_image_example.py). Several techniques are list below:

* We use ``ImageDataset`` and ``data.DataLoader`` to load the image, just like this:
```python
loader = sunnerData.DataLoader(sunnerData.ImageDataset(
    root = [
        ['./Datasets/waiting_for_you_dataset/wait'], 
        ['./Datasets/waiting_for_you_dataset/real_world']
    ],
    transforms = transforms.Compose([
        sunnertransforms.Resize((160, 320)),
        sunnertransforms.ToTensor(),
        sunnertransforms.ToFloat(),
        sunnertransforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ])), batch_size=32, shuffle=False, num_workers = 2
)
```

* The upper wrapper ``IterationLoader`` can help you control the iteration of training more easily!
```python
loader = sunnerData.IterationLoader(loader, max_iter = 1)
```