import numpy as np

class MockDataset:
    def __init__(self, sample_size=100, seed: int=0):
        self._sample_size = sample_size
        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, i: int):
        raise NotImplementedError()

    def __len__(self):
        return self._sample_size


class RandomImageClassificationDataset(MockDataset):
    def __init__(self, height: int=100, width: int=100, num_classes: int=10, multilabel: int=False, num_channels=3, channels_first=True, **kwargs):
        self._height = height
        self._width = width
        self._num_classes = num_classes
        self._multilabel = multilabel
        self._num_channels = 3
        self._channels_first = channels_first
        if channels_first:
            self._img_shape = (num_channels, height, width)
        else:
            self._img_shape = (height, width, num_channels)
        
        super().__init__(**kwargs)

    def __getitem__(self, i: int):
        img = np.random.rand(*self._img_shape)
        if self._multilabel:
            label = np.random.randint(low=0, high=2, size=self._num_classes)
        else:
            label = np.random.randint(low=0, high=self._num_classes)
        return img, label