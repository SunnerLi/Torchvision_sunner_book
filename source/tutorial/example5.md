# Example 5: Train-test-split

The full program can be found [here](???). This example will demonstrate how to split some data as testing data, and remain the rest one as training data. You can achieve this goal by setting ``split_ratio`` as non-zero and setting ``save_file`` flag as ``True``. Here is the example:
```python
train_dataset = sunnerData.ImageDataset(
    root = [['image_folder']], 
    transform = None, 
    split_ratio = 0.1, 
    save_file = True
)
```

After that, the ``.split.pkl`` file is created. You can load ``.split.pkl`` to construct another ``Dataset`` instance with your own custom augmentation operations:
```python
test_dataset = sunnerData.ImageDataset(
    file_name = '.split.pkl',
    transform = transforms.Compose([
        sunnertransforms.Resize((160, 320)),
        sunnertransforms.ToTensor(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize(),
    ])
)
```

At last, just create the ``DataLoader`` instance with this ``Dataset`` instance. All rest work is the same!
```python
loader = sunnerData.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers = 2
)
```

To simplified the usage of this package, we don't provide the extra parameters which can let you change the name of split record file. That is, the name can only be ``.split.pkl``. You should use other python module (e.g. ``os`` module) to revise the name if you want to change. 