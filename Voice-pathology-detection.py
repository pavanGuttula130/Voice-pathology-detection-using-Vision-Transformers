# %%
import opensmile
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import librosa
import tensorflow as tf

# %% [markdown]
# # Machine Learning

# %%
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# %%
class_labels = np.array(["Healthy","Laryngitis","Reinke_Odem"])

# %%
Data_path_train = r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\MutliClass_SplitedData\Train'
Data_path_test = r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\MutliClass_SplitedData\Test'
all_audio = []
all_sampling_rates = []
for i in os.listdir(Data_path_train):
    audio_file_path = os.path.join(Data_path_train,i)
    signal,sample_rate = librosa.load(audio_file_path,sr=8000)
    all_audio.append(signal)
    all_sampling_rates.append(sample_rate)
    
for i in os.listdir(Data_path_test):
    audio_file_path = os.path.join(Data_path_test,i)
    signal,sample_rate = librosa.load(audio_file_path,sr=8000)
    all_audio.append(signal)
    all_sampling_rates.append(sample_rate)

# %%
max_len = len(max(all_audio,key=len))

# %%
def padding(arr,length):
    len_ = length - len(arr)
    new_arr = [float(0.0000)]*len_ + arr
    return new_arr

def extract_mfcc(signal,sample_rate):
    mfcc_features = librosa.feature.mfcc(y=np.array(signal,dtype='float32'), sr=sample_rate,n_mfcc=13)
    return mfcc_features

def preprocessing(test_path,csv_path,max_len,unique_labels):
    jitter = []
    shimmer = []
    f0 = []

    for i in tqdm(os.listdir(test_path)):
        audio_file = os.path.join(test_path,i)
        features = smile.process_file(audio_file)
        reshaped_features = np.reshape(features, (1, -1))
        df = pd.DataFrame(reshaped_features, columns=smile.feature_names)
        jitter.append(df['jitterLocal_sma3nz_amean'][0])
        shimmer.append(df['shimmerLocaldB_sma3nz_amean'][0])
        f0.append(df['F0semitoneFrom27.5Hz_sma3nz_amean'][0])

    test_audio = []
    test_sampling_rates = []
    for i in os.listdir(test_path):
        audio_file_path = os.path.join(test_path,i)
        signal,sample_rate = librosa.load(audio_file_path,sr=8000)
        test_audio.append(signal)
        test_sampling_rates.append(sample_rate)

    padded_audio = [padding(i.tolist(),max_len) for i in test_audio]
    
    mfcc_features = [extract_mfcc(signal=signal,sample_rate=8000) for signal in padded_audio]
    
        
    o_features = np.array([jitter[0],shimmer[0],f0[0]])
    test_features = np.hstack((o_features,mfcc_features[0].ravel()))
    for i in range(1,len(mfcc_features)):
        o_features = np.array([jitter[i],shimmer[i],f0[i]])
        stacked = np.hstack((o_features,mfcc_features[i].ravel()))
        test_features = np.vstack((test_features,stacked))
        
    csv = pd.read_csv(csv_path)
        
    list_ID = list(csv['ID'])
    list_Disease = list(csv['Disease'])
    
    wav_files = os.listdir(test_path)
    
    test_labels=[]
    for i in wav_files:
        for j in list_ID:
            if i==j:
                index=(list_ID.index(j))
                test_labels.append(list_Disease[index])
        
    class_labels = test_labels            
    label_map = {label: index + 1 for index, label in enumerate(unique_labels)}
    labels = [label_map[label] for label in class_labels]
    test_labels=np.array(labels)-1
    
    return test_features,test_labels

# %%
X_train_ml,y_train_ml = preprocessing(r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\MutliClass_SplitedData\Train',r'E:\Semester-6\Speech-Processing\Project\My_trail\MultiClass\MultiClass_AugmentedData.csv',max_len,class_labels)
X_test_ml,y_test_ml = preprocessing(r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\MutliClass_SplitedData\Test',r'E:\Semester-6\Speech-Processing\Project\My_trail\MultiClass\MultiClass_AugmentedData.csv',max_len,class_labels)

# %%
y_train_ml

# %%
import xgboost as xgb
from sklearn.metrics import accuracy_score

model = xgb.XGBClassifier()
model.fit(X_train_ml,y_train_ml)

y_pred_xgb = model.predict(X_test_ml)

accuracy_xgb = accuracy_score(y_test_ml, y_pred_xgb)
print("Accuracy:", accuracy_xgb)

# %%
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm.fit(X_train_ml, y_train_ml)

y_pred_lgbm = lgbm.predict(X_test_ml)

accuracy_lgbm = accuracy_score(y_test_ml, y_pred_lgbm)
print("Accuracy:", accuracy_lgbm)

# %%
from sklearn.ensemble import ExtraTreesClassifier

extra_trees = ExtraTreesClassifier()
extra_trees.fit(X_train_ml, y_train_ml)

y_pred = extra_trees.predict(X_test_ml)

accuracy_extra_tree = accuracy_score(y_test_ml, y_pred)
print("Accuracy:", accuracy_extra_tree)

# %% [markdown]
# # Deep Learning

# %%
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# %%
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# %%
csv_path = r'E:\Semester-6\Speech-Processing\Project\Trail_5\MultiClass\MultiClass_AugmentedData.csv'

# %%
def load_image_data(path,csv_path):
    
    y=pd.read_csv(csv_path)
    list_ID=list(y['ID'])
    list_Disease=list(y['Disease'])
    
    
    X = []
    y = []
    for i in tqdm(os.listdir(path)):
        img = os.path.join(path,i)
        X.append(cv2.imread(img).tolist())
        
        i=i.replace('.png','')
        for j in list_ID:
            if i==j:
                index=(list_ID.index(j))
                y.append(list_Disease[index])
             
    print("Converting to numpy")
                
    X = np.array(X)
    y = np.array(y)
    
    return X,y
                

# %%
train_path = r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\mel_train_sir'
test_path = r'E:\Semester-6\Speech-Processing\Project\Trail_5\Phase - II\mel_test_sir'

X_train,y_train = load_image_data(train_path,csv_path)

# %%
X_test,y_test = load_image_data(test_path,csv_path)

# %%
y_test

# %%
y_test_ml

# %%
num_classes = 3
input_shape = (372,499, 3)
X_test.shape

# %%
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 50
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
# transformer_layers = 8
transformer_layers = 16
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

# %%
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal",seed=42),
        layers.RandomRotation(factor=0.02,seed=42),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2,
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(X_train)

# %%
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# %%
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = X_train[np.random.choice(range(X_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

# %%
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# %%
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# %%
num_classes

# %%
def label_encoding(y,unique_labels):
    
    class_labels = y
    label_map = {label: index + 1 for index, label in enumerate(unique_labels)}
    numeric_labels = [label_map[label] for label in class_labels]
    numeric_labels=np.array(numeric_labels)-1
    numeric_labels = keras.utils.to_categorical(numeric_labels,num_classes=3)
    
    return numeric_labels

# %%
y_train = label_encoding(y_train,class_labels)
y_test = label_encoding(y_test,class_labels)

# %%
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1
        #callbacks=[checkpoint_callback],
    )
    return history,model

vit_classifier = create_vit_classifier()
history,model = run_experiment(vit_classifier)

# %%
accuracy_dl = model.evaluate(X_test,y_test)

# %%
y_pred_vit = model.predict(X_test)

# %%
y_pred_vit = np.argmax(y_pred_vit,axis=1)
y_pred_vit

# %%
y_test = np.argmax(y_test,axis=1)
y_test

# %%
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Assuming you have y_pred and y_test as numpy arrays or lists

# Calculate precision
precision = precision_score(y_test, y_pred_vit, average='weighted')

# Calculate recall
recall = recall_score(y_test, y_pred_vit, average='weighted')

# Calculate F1 score
f1 = f1_score(y_test, y_pred_vit, average='weighted')

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred_vit)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)

# %%
y_pred

# %%
y_pred_lgbm

# %%
y_pred_xgb

# %%
import numpy as np

# Calculate the weights based on model accuracies
weight_xgb = accuracy_xgb / (accuracy_dl[1] + accuracy_xgb + accuracy_lgbm + accuracy_extra_tree)
weight_lgbm = accuracy_lgbm / (accuracy_dl[1] + accuracy_xgb + accuracy_lgbm + accuracy_extra_tree)
weight_extra_trees = accuracy_extra_tree / (accuracy_dl[1] + accuracy_xgb + accuracy_lgbm + accuracy_extra_tree)
weight_dl = accuracy_dl[1] / (accuracy_dl[1] + accuracy_xgb + accuracy_lgbm + accuracy_extra_tree)

# Determine the shape of y_test
test_shape = np.shape(y_test)

# Perform weighted probabilistic voting
ensemble_probs = (weight_xgb * np.eye(3)[y_pred_xgb.tolist()]) + (weight_lgbm * np.eye(3)[y_pred_lgbm.tolist()]) + (weight_dl * np.eye(3)[y_pred_vit.tolist()]) + (weight_extra_trees * np.eye(3)[y_pred.tolist()])
# Determine the final ensemble predictions based on the highest average probability
ensemble_predictions = np.argmax(ensemble_probs, axis=1)

print("Final ensemble predictions:", ensemble_predictions)


# %%
accuracy_score(y_test,ensemble_predictions)

# %%
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Assuming you have y_pred and y_test as numpy arrays or lists

# Calculate precision
precision = precision_score(y_test, np.array(ensemble_predictions), average='weighted')

# Calculate recall
recall = recall_score(y_test, np.array(ensemble_predictions), average='weighted')

# Calculate F1 score
f1 = f1_score(y_test, np.array(ensemble_predictions), average='weighted')

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test, np.array(ensemble_predictions))

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)
