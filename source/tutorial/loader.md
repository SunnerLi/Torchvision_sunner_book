# torchvision_sunner.data.loader

In the ``torchvision_sunner.data`` module, we defined ``ImageLoader`` class previously, and used it to become the data loader object. In the version of ``18.9.15``, this mechanism is also remained. But you can also use the traditional ``data.DataLoader`` to load the data. Before we start to address the function of each API in loader category, let's import the module first:
```python
import torchvision_sunner.data as sunnerData
import torch.utils.data as data
```

You can create the ``ImageLoader`` as usual:
```python
loader = sunnerData.ImageLoader(
    dataset = sunnerData.ImageDataset(
        root = [['image_folder']],
        transforms = transforms.Compose([
            sunnertransforms.ToTensor(),
        ])
    ), batch_size=32, shuffle=False, num_workers = 2
)
```

Also, you can use ``DataLoader`` directly:
```python
loader = data.DataLoader(
    dataset = sunnerData.ImageDataset(
        root = [['image_folder']],
        transforms = transforms.Compose([
            sunnertransforms.ToTensor(),
        ])
    ), batch_size=32, shuffle=False, num_workers = 2
)
```

## torchvision_sunner.data.ImageLoader [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/data/loader.py#L14)]

The wrapper of the ``DataLoader``. This function provides few parameters to construct the data loader. If you want to assign the parameter of loader for more detail, we recommand you to use the torchvision original dataloader

### Parameter
* **dataset (data.Dataset) -** The dataset which you want to load
* **batch_size (int) -** The number of image you want to load in single batch
* **shuffle (bool) -** To shuffle the image or not
* **num_workers (int) -** The number of process you want to create to load the data

## torchvision_sunner.data.MultiLoader [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/data/loader.py#L34)]

This class can deal with multiple dataset object. The usage of ``MultiLoader`` is as same as other data loader. For example:
```python
dataset1 = sunnerData.ImageDataset(
    root = [['image_folder1']], 
    transforms = transforms_compose_op1
)
dataset2 = sunnerData.ImageDataset(
    root = [['image_folder2']], 
    transforms = transforms_compose_op2
)
loader = sunnerData.MultiLoader([dataset1, dataset2], num_workers = 2)
```

### Parameter
* **datasets (The list of data.Dataset) -** The datasets which you want to load
* **batch_size (int) -** The number of image you want to load in single batch
* **shuffle (bool) -** To shuffle the image or not
* **num_workers (int) -** The number of process you want to create to load the data

## torchvision_sunner.data.IterationLoader [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/data/loader.py#L90)]

This class can wrap the loader into iteration view. For the usual usage, we might go through the whole data for multiple rounds. In this case, we call ``epoch`` for each round. However, we might not need much training step conversely. On the other hand, we can load the data for rigid number of ``iteration`` and save the time. Here is the usage to assign loading the batch data for 30000 times: 

```python
loader = sunnerData.MultiLoader([dataset1, dataset2], num_workers = 2)
loader = sunnerData.IterationLoader(loader, 30000)
```

### Parameter
* **loader (data.DataLoader) -** The loader you want to wrap
* **max_iter (int) -** The number of iteration you want to assign