메모해두려고 만들었습니다. 폴더 제멋대로 만들어서 미안해요.

<br>

## 참고한 자료 / 참고할 자료

<br>

ENG - Draw CAM with GAP <br>
https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/

<br>

ENG - Yolo v3 build model with keras, testing implementation <br>
[yolo v3 testing implementation](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)

<br>

ENG - yolo v2 step-by-step <br>
https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb

<br>

KOR - Keras difference between backends <br>
https://github.com/keras-team/keras-docs-ko/blob/master/sources/backend.md

<br>

### Image Augmentation

ENG - Building powerful image classification models using very little data <br>
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html<br>

<br>

KOR - Building powerful image classification models using very little data <br>
https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/

<br>

image augmentation for detection problem <br>
https://blog.paperspace.com/data-augmentation-for-bounding-boxes/

<br>

### Weight Initialization

<br>

ENG - What initialization works better for Convolution Filter with ReLU Activation function <br>
https://medium.com/@tylernisonoff/weight-initialization-for-cnns-a-deep-dive-into-he-initialization-50b03f37f53d

<br>

KOR - Deepest - initialization <br>
https://deepestdocs.readthedocs.io/en/latest/002_deep_learning_part_1/0025/

<br>

## Keras Real Code

### CAM Colab Code 만들기

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
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='vgg16')

    
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
def load_attributes_from_hdf5_group(group, name):
  """Loads attributes of the specified name from the HDF5 group.
  This method deals with an inherent problem
  of HDF5 file which is not able to store
  data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
  Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to load.
  Returns:
      data: Attributes data.
  """
  if name in group.attrs:
    data = [n.decode('utf8') for n in group.attrs[name]]
  else:
    data = []
    chunk_id = 0
    while '%s%d' % (name, chunk_id) in group.attrs:
      data.extend(
          [n.decode('utf8') for n in group.attrs['%s%d' % (name, chunk_id)]])
      chunk_id += 1
  return data
```

<br>

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


### 노가다 끝에 얻은 조각 코드


```Python3
    with h5py.File(VGG_weight_path, 'r') as hw:
      for index, layer_name in enumerate(hw.attrs["layer_names"]):
          layer_obj = hw[layer_name]
          weights = [layer_obj[weight_name] for weight_name in layer_obj.attrs['weight_names']]
          print(model.layers[index])
          model.layers[index].set_weights(weights)
          if model.layers[index].name == "convolution2d_13":
            break
      print('Model loaded.')
```

- 특정 레이어까지 업로드하는 핵심 코드.
- 이 때, ```model.layers[index]``` 에서 model 은, ```model = models.Model(inputs, x, name='modelname')```에 의해 완성된 model 이어야 함.
- 그런데 이러한 방식으로 load weight 를 한다면, 문제는 무엇이냐면 input layer 이나 flatten layer 과 같이, h5 file 의 layer 과 1:1 대응이 안 되면 터진다는 것. h5 file 에는 input layer 이 명세되어있지 않으므로 input layer 을 model 짤때 넣어놨다면 터짐. 또한, h5 file 에는 conv2d 만 명세되어 있는데, 코드에서 conv2D 따로 zerropad layer 따로 따로 짰다면 터짐.
- 여튼 핵심은 이게 맞음.


<br>

### Channel 에 맞추어 수정하기

- openCV imread : channel last
- tensorflow : channel last
- theano(default for local) : [code at here](https://github.com/tdeboissiere/VGG16CAM-keras/blob/master/VGGCAM-keras.py) channel first

<br>

[Numpy.transpose axis 이해하기](https://superelement.tistory.com/18) <br>

<br>

### 터지는 현상들

- Keras 의 imagedatagenerator 안에는 hand-craft preprocessing 을 할 수 있는 parameter 이 있음.

```
  # With Data Augmentation
    train_datagenerator = keras.preprocessing.image.ImageDataGenerator(
                                                 #featurewise_center=False, 
                                                 #samplewise_center=False, 
                                                 #featurewise_std_normalization=False, 
                                                 #samplewise_std_normalization=False, 
                                                 #zca_whitening=False, 
                                                 #zca_epsilon=1e-06, 
                                                 #rotation_range=0, 
                                                 ##width_shift_range=0.0, 
                                                 #height_shift_range=0.0,
                                                 #brightness_range=None, 
                                                 #shear_range=0.0, 
                                                 #zoom_range=0.0, 
                                                 #channel_shift_range=0.0, 
                                                 #fill_mode='nearest', 
                                                 #cval=0.0, 
                                                 #horizontal_flip=False, 
                                                 #vertical_flip=False, 
                                                 #rescale=None,
                                                 data_format='channels_last',
                                                 dtype='float32',
                                                 preprocessing_function=preprocessing_function,
                                                 #validation_split=0.0, 
                                                 #interpolation_order=1, 
                                                 )
```

여기에 들어가는 preprocessing function 

```python3
def preprocessing_function(im):
  # N.B. The data should be compatible with the VGG16 model style:
  im[:,:,0] -= 103.939
  im[:,:,1] -= 116.779
  im[:,:,2] -= 123.68
  return im
```

이 preprocessing function 은 numpy 3D array 를 input 으로 받는데, 여기서 return 값을 지정을 해주지 않으면 모든 값이 NAN 으로 찍히는 마법을 구경할 수 있음. 처음에는 nan error 이 나길래 tensorboard 가 터지는 건줄알았는데 그게 아니라 애초에 모든 input 값이 nan 이었던 것임.


[Batch Dot, Tensorflow.Backend.batch_dot](https://www.tensorflow.org/api_docs/python/tf/keras/backend/batch_dot)
