# Extend your data format

Before the instruction of each API, the most important feature should be clearified. Even though you can call the API which ``Torchvision_sunner`` provides, the only file you need to **much** realize is ``Torchvision_sunner.read.py``. Also, **you might need to revise this script to handle for your own data!** This script define the two functions, and they help you to deal with other data format!

## torchvision_sunner.read.readContains [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/read.py#L14)]

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

## torchvision_sunner.read.readItem [[source](https://github.com/SunnerLi/Torchvision_sunner/blob/master/torchvision_sunner/read.py#L42)]

```python
def readItem(item_name):
    file_type = item_name.split('.')[-1]
    if file_type == "png" or file_type == 'jpg':
        # Read the image by PIL 
        file_obj = Image.open(item_name)

        # Drop the alpha channel
        if np.asarray(file_obj).shape[-1] == 4:
            file_obj = Image.fromarray(np.asarray(file_obj)[:, :, :3], mode='RGB')

        # Convert the image format if the image is gray-scale
        if len(np.asarray(file_obj).shape) == 2:
            file_obj = file_obj.convert('L')
    return file_obj
``` 
The above quote shows the original definition of the function. This function will load the particular file. However, this function only handle for ``JPG`` and ``PNG`` only. **If you want to custom for your own data format (e.g. the ``.pth`` or ``.npy`` file format), you should revise by yourself!** Just add the other ``elif`` statement like the previous lines.

---

## The other example

We are sorry that the master branch cannot include whole input data format since the user should install lots of related dummy package. However, we provide some link for you to refer if you want to use for the specific data format. The following shows the link:

* DICOM format [[link](https://gist.github.com/SunnerLi/b90bc7bda7531045eedbf14ee4addfe3)] - used in medical scenario. 