## Let's Get Start!

### Import 
Before we use the API of ``Torchvision_sunner``, we should import the library. The design of ``Torchvision_sunner`` is very similar to ``Torchvision``. There are two main module in ``Torchvision``: ``data`` and ``transforms``, and also contain in ``Torchvision_sunner``. As the result, we also need to import them firstly.
```python
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms
```

In the ``Torchvision_sunner``, we also provide the compatable interface with ``torchvision``. Thus you can still use some ``torchvision`` API. For example, ``torchvision.transforms.Compose``. However, this toolkit also provide the support to the video. So we don't recommand to mixtually use them. 

### Create the loader

We can create the ``DataLoader`` object Just like the following. For example, you can place the dataset into the current folder. You can also assign some preprocessing function like ``ToTensor`` or ``Normalize``. It's very simple to load the data with different domain at once. 

```python
loader = sunnerData.DataLoader(sunnerData.ImageDataset(
    root = [
        ['./Dataset/waiting_for_you_dataset/wait'], 
        ['./Dataset/waiting_for_you_dataset/real_world']
    ],
    transform = transforms.Compose([
        sunnertransforms.Resize((160, 320)),
        sunnertransforms.ToTensor(),
        sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
        sunnertransforms.Normalize(),
    ])), batch_size=32, shuffle=False, num_workers = 2
)
```

You should notice about the format of ``root`` parameters. The detail can be found in API pages.