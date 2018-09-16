# Extend your data format

Before the instruction of each API, the most important feature should be clearified. Even though you can call the API which ``Torchvision_sunner`` provides, the only file you need to **much** realize is ``Torchvision_sunner.read.py``. Also, **you might need to revise this script to handle for your own data!** This script define the two functions, and they help you to deal with other data format!

## torchvision_sunner.read.readContains [[source](https://github.com/SunnerLi/Torchvision_sunner2/blob/master/torchvision_sunner/read.py#L14)]

```python
def readContain(folder_name):
    # Check the common type in the folder
    common_type = Counter()
    for name in os.listdir(folder_name):
        common_type[name.split('.')[-1]] += 1
    common_type = common_type.most_common()[0][0]

    # Deal with the type
    if common_type == 'jpg':
        name_list = glob(os.path.join(folder_name, '*.jpg'))
    elif common_type == 'png':
        name_list = glob(os.path.join(folder_name, '*.png'))
    elif common_type == 'mp4':
        name_list = glob(os.path.join(folder_name, '*.mp4'))
    else:
        raise Exception("Unknown type {}, You should customize in read.py".format(common_type))
    return name_list

```
The above quote shows the original definition of the function. This function will load the containing in the particular folder. However, this function only deal with ``JPG``, ``PNG`` and ``MP4`` file format. **If you want to process with extra data format, you should revise by yourself!** Just add the other ``elif`` statement like the previous lines.

## torchvision_sunner.read.readItem [[source](https://github.com/SunnerLi/Torchvision_sunner2/blob/master/torchvision_sunner/read.py#L42)]

```python
def readItem(item_name):
    file_type = item_name.split('.')[-1]
    if file_type == "png" or file_type == 'jpg':
        file_obj = np.asarray(Image.open(item_name))
        
        if len(file_obj.shape) == 3:
            # Ignore the 4th dim (RGB only)
            file_obj = file_obj[:, :, :3]
        elif len(file_obj.shape) == 2:
            # Make the rank of gray-scale image as 3
            file_obj = np.expand_dims(file_obj, axis = -1)
    return file_obj
``` 
The above quote shows the original definition of the function. This function will load the particular file. However, this function only handle for ``JPG`` and ``PNG`` only. **If you want to custom for your own data format (e.g. the ``.pth`` or ``.npy`` file format), you should revise by yourself!** Just add the other ``elif`` statement like the previous lines.
