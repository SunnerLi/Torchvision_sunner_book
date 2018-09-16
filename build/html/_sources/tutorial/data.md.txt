# torchvision_sunner.data

In the ``torchvision_sunner.data`` module, we provide for two different dataset: ``ImageFolder`` and ``VideoFolder``. You can load the images in different domain right away by using ``ImageFolder``. On the other hand, you can also load the video sequences in different domain right away by using ``VideoFolder``. 

## class torchvision_sunner.data.ImageFolder [[source](https://github.com/SunnerLi/Torchvision_sunner2/blob/master/torchvision_sunner/data/image_dataset.py#L34)]

This is the fundemental class in ``torchvision_sunner``, and it's inherit from ``torch.utils.data.Dataset``. You can just create the instance by the following way. We assume the two different image folders are ``image_folder1`` and ``image_folder2``. 
```python
dataset = sunnerData.ImageDataset(
    root = [
        ['image_folder1'], 
        ['image_folder2']
    ],
    file_name = '.remain.pkl',
    transform = transforms.Compose([
        sunnertransforms.ToTensor(),
    ])
)
```

This class also support for paired data! If the number of image in each folder is the same, then the corresponding data will be gotten. For example, you want to train for ``pix2pix`` which needs pair data, and you can write as:
```
dataset = sunnerData.ImageDataset(
    root = [
        ['draft_folder'],   # This folder contain 2000 draft images
        ['bag_folder']      # This folder contain 2000 bag images
    ],
    file_name = '.remain.pkl',
    transform = transforms.Compose([
        sunnertransforms.ToTensor(),
    ])
)
```
The corresponding draft image and bag image will be raise. 

```python
for img_draft, img_bag in loader:
    # Do something computing
```
You can use the loader just the same as usual ``torch.utils.data.Dataset`` instance. In the default setting, the type of the return object is ``np.ndarray``, and the rank format is ``BHWC``. You should use ``ToTensor()`` augmentation function to change into ``torch.Tensor`` type, and use ``Transpose(torchvision_sunner.transforms.BHWC2BCHW)`` to change the rank format into ``BCHW``. 


### Paeameters
* **root (list of list) -** The list of different image domain. You **must use list of list** to represent the different domain. Each element in the inner list can be the path of **image** or the path of **folder**. Also, you can mutually to combine the single domain. We list some example:
```python
[[image1.jpg], [image2.jpg]]                                # This is valid
[[image1.jpg], [image_folder]]                              # This is valid
[[image1.jpg, image2.jpg], [image_folder1, image_folder2]]  # This is valid
[[image1.jpg, image_folder]]                                # This is also valid
[image1.jpg, image2.jpg]                                    # This is invalid!
[image1.jpg, image_folder]                                  # This is invalid!
``` 
* **file_name (str) -** The name of record file. In the design of ``torchvision_sunner`` toolkit, we will store some information into the single ``.pkl`` file. This mechanism can avoid the program to tranversal the storage next time but read the information from the file only. This parameter will assign the path of record file. In the default, the name of record file is ``.remain.pkl``. 

* **sample_method (int) -** The method to deal with data unbalance. Since the number of images may not the same in different domains. We provide for **under sampling** or **over sampling** two choice. You can assign as ``torchvision_sunner.UNDER_SAMPLING`` or ``torchvision_sunner.OVER_SAMPLING``, respectively. 

* **transform (torchvision.transforms.Compose) -** The series of augmentation. You can use the augmentation methods which are defined in ``torchvision_sunner.transforms``. Check the [page](transforms.html) for the detail

* **split_ratio (float) -** The ratio to do the train-test split. The range of this ratio is between 0 to 1. The split process will not be conducted if you set the ratio as ``0.0``. The default is ``0.0``. 

* **save_file (Bool) -** The flag to control if storing the record file. Since there is some cases that the user don't have priority to do the IO, or the environment doesn't have the storage to store the record file, we provide this mechanism. Anything will not be saved if you set this flag as ``False``. The default is ``False``. 

## class torchvision_sunner.data.VideoFolder [[source](https://github.com/SunnerLi/Torchvision_sunner2/blob/master/torchvision_sunner/data/video_dataset.py#L35)]

This class is the video version of ``ImageFolder``, and it's also inherit from ``torch.utils.data.Dataset``. You can just create the instance by the following way. We assume the two different image folders are ``video_folder1`` and ``video_folder2``. 
```python
dataset = sunnerData.ImageDataset(
    root = [
        ['video_folder1'], 
        ['video_folder2']
    ],
    file_name = '.remain.pkl',
    transform = transforms.Compose([
        sunnertransforms.ToTensor(),
    ])
)
```

For the function of the ``VideoFolder``. We will use ``ffmpeg`` package to split each video, and save into a temporal folder. During loading the data, we only send a small sequence of video to supply the program. For example, the single video migh have 9876 frames. But in single batch, we only provide 10 frames. This mechanism can make the program reduce the burden in avoid to load the whole 9876 frames into RAM at the same time. 

* **Notice:** The single record file only store the information of the list of different domain, and also store one kinds of folder information. If you want to use multiple dataset in your program, you should assign different name of ``file_path``, or the program cannot load the correct information. 
* **Notice:** The rank format of ``ImageFolder`` return is ``BHWC``, but the rank format of ``VideoFolder`` return is ``BTHWC``. 

### Paeameters

The parameters are very similar to the ``ImageFolder``. 

* **root (list of list) -** The list of different image domain. You **must use list of list** to represent the different domain. Each element in the inner list can be the path of **video** or the path of **folder**. Also, you can mutually to combine the single domain. We list some example:
```python
[[video1.mp4], [video2.mp4]]                                # This is valid
[[video1.mp4], [video_folder]]                              # This is valid
[[video1.mp4, video2.mp4], [video_folder1, video_folder2]]  # This is valid
[[video1.mp4, video_folder]]                                # This is also valid
[video1.mp4, video2.mp4]                                    # This is invalid!
[video1.mp4, video_folder]                                  # This is invalid!
``` 

* **file_name (str) -** The name of record file. In the design of ``torchvision_sunner`` toolkit, we will store some information into the single ``.pkl`` file. This mechanism can avoid the program to tranversal the storage next time but read the information from the file only. This parameter will assign the path of record file. In the default, the name of record file is ``.remain.pkl``. 

* **T (int) -** The maximun length of small video sequence, and it's the second dimension in the tensor's rank. The default of T is 10.    

* **sample_method (int) -** The method to deal with data unbalance. Since the number of videos may not the same in different domains. We provide for **under sampling** or **over sampling** two choice. You can assign as ``torchvision_sunner.UNDER_SAMPLING`` or ``torchvision_sunner.OVER_SAMPLING``, respectively. 

* **transform (torchvision.transforms.Compose) -** The series of augmentation. You can use the augmentation methods which are defined in ``torchvision_sunner.transforms``. Check the [page](transforms.html) for the detail

* **split_ratio (float) -** The ratio to do the train-test split. The range of this ratio is between 0 to 1. The split process will not be conducted if you set the ratio as ``0.0``. The default is ``0.0``. 

* **decode_root (str) -** The path to store the ffmpeg decode result. In the default, the frames will be stored in ``.decode``

* **save_file (bool) -** The flag to control if storing the record file. Since there is some cases that the user don't have priority to do the IO, or the environment doesn't have the storage to store the record file, we provide this mechanism. Anything will not be saved if you set this flag as ``False``. The default is ``False``. 