from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

import numpy as np

# get copy of ResNet50
model = ResNet50(weights='imagenet')
#model = Sequential()

#for layer in res_model.layers:
#    model.add(layer)

print("before pop")
model.summary()

# 27 for joint angles, 3 for camera rotation.
posenet_final_layer = Dense(30, activation='relu')(model.layers[-2].output)

model2 = Model(model.input, posenet_final_layer)

print("after pop")
model2.build(None)
model2.summary()


# check the weights are same;

# shallow copy
resnet50_layers = model.layers
posenet_layers = model2.layers

# set weights
for i in range(len(posenet_layers) - 1):
    posenet_layers[i].set_weights(resnet50_layers[i].get_weights())

# check equality
for i in range(len(posenet_layers)):
    for mat1, mat2 in zip(resnet50_layers[i].get_weights(), posenet_layers[i].get_weights()):
        if (not np.equal(mat1.shape, mat2.shape).all()):
            print("Shapes are different at layer #", i)
            print(mat1.shape)
            print(mat2.shape)
            continue

        if (not np.equal(mat1, mat2).all()):
            print("Weights are different at layer #", i)
            continue


img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)
