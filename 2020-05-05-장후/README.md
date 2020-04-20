메모해두려고 만들었습니다. 폴더 제멋대로 만들어서 미안해요.

<br>

Draw CAM with GAP <br>
https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/

<br>

Yolo v3 build model with keras, testing implementation <br>
[yolo v3 testing implementation](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)

<br>

keras example code <br>
https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/

<br>

yolo v2 step-by-step <br>
https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb

<br>

### Image Augmentation

image augmentation for detection problem <br>
https://blog.paperspace.com/data-augmentation-for-bounding-boxes/

<br>

### Weight Initialization

<br>

What initialization works better for Convolution Filter with ReLU Activation function <br>
https://medium.com/@tylernisonoff/weight-initialization-for-cnns-a-deep-dive-into-he-initialization-50b03f37f53d

<br>

Deepest - initialization <br>
https://deepestdocs.readthedocs.io/en/latest/002_deep_learning_part_1/0025/

<br>

### Keras Real Code

<br>

[VGG16](https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py) <br>
[get_submodules_from_kwargs](https://github.com/keras-team/keras-applications/blob/bc89834ed36935ab4a4994446e34ff81c0d8e1b7/keras_applications/__init__.py#L13)

<br>

```Python3
    #line 82
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
```

<br>

```Python3
def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils
```

<br>

```Python3
    # VGG16
    # Create model.
    model = models.Model(inputs, x, name='vgg16')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model
```
<br>

아무리 찾아도, 저 .get_file 을 찾을수가 없었음.

<br>

[model.Model Class](https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/engine/training.py#L28) <br>
[model.load_weight](https://github.com/keras-team/keras/blob/1cf5218edb23e575a827ca4d849f1d52d21b4bb0/keras/engine/network.py#L1188) <br>


```Python3
    #line 1188
    def load_weights(self, filepath, by_name=False,
                     skip_mismatch=False, reshape=False):
        """Loads all layer weights from a HDF5 save file.
        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.
        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.
        # Arguments
            filepath: String, path to the weights file to load.
            by_name: Boolean, whether to load weights by name
                or by topological order.
            skip_mismatch: Boolean, whether to skip loading of layers
                where there is a mismatch in the number of weights,
                or a mismatch in the shape of the weight
                (only valid when `by_name`=True).
            reshape: Reshape weights to fit the layer when the correct number
                of weight arrays is present but their shape does not match.
        # Raises
            ImportError: If h5py is not available.
        """
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            if by_name:
                saving.load_weights_from_hdf5_group_by_name(
                    f, self.layers, skip_mismatch=skip_mismatch,
                    reshape=reshape)
            else:
                saving.load_weights_from_hdf5_group(
                    f, self.layers, reshape=reshape)
            if hasattr(f, 'close'):
                f.close()
            elif hasattr(f.file, 'close'):
                f.file.close()    

```


[hdf5_format.py, def save_weights_to_hdf5_group(f,layers)](https://github.com/tensorflow/tensorflow/blob/db821b3c2b5a999da6915ff079e9709329a722fb/tensorflow/python/keras/saving/hdf5_format.py) <br>

