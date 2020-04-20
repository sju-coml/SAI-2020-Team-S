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

<br>


```Python3
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

[model.Model Class](https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/engine/training.py#L28) <br>
[hdf5_format.py, def save_weights_to_hdf5_group(f,layers)](https://github.com/tensorflow/tensorflow/blob/db821b3c2b5a999da6915ff079e9709329a722fb/tensorflow/python/keras/saving/hdf5_format.py) <br>

