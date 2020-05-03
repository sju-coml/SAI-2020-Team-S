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
    
    # Arguments :
    '''
    weights: one of `None` (random initialization),
    'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
    '''
    
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
```Python3
        else:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)

```

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

``` Python3
from tensorflow.python.keras import backend as K

# def load_weights_from_hdf5_group(f, layers):
  """Implements topological (order-based) weight loading.
  Arguments:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.
  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file.
  """
  
  layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
  filtered_layer_names = []
  for name in layer_names:
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    if weight_names:
      filtered_layer_names.append(name)
  layer_names = filtered_layer_names
  if len(layer_names) != len(filtered_layers):
    raise ValueError('You are trying to load a weight file '
                     'containing ' + str(len(layer_names)) +
                     ' layers into a model with ' + str(len(filtered_layers)) +
                     ' layers.')

  # We batch weight value assignments in a single backend call
  # which provides a speedup in TensorFlow.
  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
    layer = filtered_layers[k]
    symbolic_weights = _legacy_weights(layer)
    weight_values = preprocess_weights_for_loading(
        layer, weight_values, original_keras_version, original_backend)
    if len(weight_values) != len(symbolic_weights):
      raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                       '" in the current model) was found to '
                       'correspond to layer ' + name + ' in the save file. '
                       'However the new layer ' + layer.name + ' expects ' +
                       str(len(symbolic_weights)) +
                       ' weights, but the saved weights have ' +
                       str(len(weight_values)) + ' elements.')
    weight_value_tuples += zip(symbolic_weights, weight_values)
  K.batch_set_value(weight_value_tuples)
```

- 진짜 오류가 오지게 나서 Keras 코드를 싹 까뒤집었는데도 문제 발견이 안돼서 왜 그런가 잘 생각을 해 봄.
- 그런데, 문제는 tensorflow.keras 로 호출하지 않고 colab 환경에서 그냥 keras 를 import 해서 생기는 문제였음.
- 그 다음에 발생한 문제는 단지 vgg-16 이 너무 오래된 모델이라 그런지, h5 file 규격이 조금 다른 듯 함.


according to https://github.com/neokt/car-damage-detective/issues/6 <br>

> @yuyifan1991 No, i was not able to find out the solution taking the "nb_layers" route. i ended up using a different approach to pop out the last layer of vgg16 and then inserting my own classifier. To get a layer of a pretrained network as an input to your own model, use something like : ```model.get_layer(layername).output function``` Hope that helps. Thanks!

> nb_layers is not going to work. VGG!6 model has 16 layers in total, 13 convolution and 3 dense. The last dense layer in default trained VGG16 model has 1000 categories. If you want to use the VGG16 model just for feature extraction, use the 13 convolution layers and maybe one dense layer. Pop out the last two layer or just the topmost layer. Once you extract the features, you can use any classifier(SVM or Logistic) to make predictions. There are multiple ways of removing the top layers from VGG16. Search google. Hope that helps.

<br>

[KOR, DSSchool : about h5 file](https://datascienceschool.net/view-notebook/f1c286a1d5164975a9909bb7a341bf4c/) <br>
[HDF File Viewer](https://www.hdfgroup.org/downloads/hdfview/?https%3A%2F%2Fwww.hdfgroup.org%2Fdownloads%2Fhdfview%2F) <br>

<br>

- 그래서, h5 file 안에 데이터가 어떻게 담겨 있나 열어볼 필요가 있었음
- HDF File viewer 을 사용해서 내부 데이터를 살펴봄.
- 다양한 경로로 다양한 h5 file 을 받아 봤는데 모두 똑같이 생겼음

<br>

변경된 
