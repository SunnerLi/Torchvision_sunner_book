# Example 1: Deal with Images

The full program can be found [here](???). Several techniques are list below:

* We use ``ImageDataset`` and ``data.DataLoader`` to load the image, just like this:
```python
loader = sunnerData.DataLoader(sunnerData.ImageDataset(
    root = [
        ['./Datasets/waiting_for_you_dataset/wait'], 
        ['./Datasets/waiting_for_you_dataset/real_world']
    ],
    transform = transforms.Compose([
        sunnertransforms.Resize((160, 320)),
        sunnertransforms.ToTensor(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize(),
    ])), batch_size=32, shuffle=False, num_workers = 2
)
```

* The upper wrapper ``IterationLoader`` can help you control the iteration of training more easily!
```python
loader = sunnerData.IterationLoader(loader, max_iter = 1)
```