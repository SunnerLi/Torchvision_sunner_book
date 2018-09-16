# Example 2: Deal with Video

The full program can be found [here](???). Several techniques are list below:

* Just like ``ImageFolder`` usage, we can load the multiple video with ``VideoFolder`` like this way:
```python
loader = sunnerData.DataLoader(
    sunnerData.VideoDataset(
        root = [
            ['./Dataset/flower/A'], 
            ['./Dataset/flower/B']
        ], transform = transform_compose_obj, T = 10
    ), batch_size=2, shuffle=False, num_workers = 2
)
```