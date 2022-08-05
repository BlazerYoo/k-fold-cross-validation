"""
Built for tfds slicing API https://www.tensorflow.org/datasets/splits
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import matplotlib.pyplot as plt



"""
Constants
"""
DATASET = "mnist"
IMG_HEIGHT = 28
IMG_WIDTH = 28
SEED = 42
MODEL_NAME = "/path/to/exported/model"

K = 10
CRS_VLD_BEGIN = 0
CRS_VLD_END = 90
TEST_BEGIN = 90
TEST_END = 100
BATCH = 16
EPOCHS = 100000



"""
Import and load dataset first time
"""
import {REPLACE_WITH_DATASET_NAME}
ds, info = tfds.load(DATASET, as_supervised=True, with_info=True)



"""
Image preprocessing
"""
def preprocess(img, label):
    image = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]) / 255
    return tf.image.rgb_to_grayscale(image), label



"""
Configure dataset for performance and shuffle consistency
"""
# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

# Ensure shuffle consistency
tf.random.set_seed(SEED)
read_config = tfds.ReadConfig(shuffle_seed=SEED)



"""
Reset model weights to ensure independence between each fold
"""
# Import exported model
model = tf.keras.models.load_model(MODEL_NAME)

# Clone and compare original and randomized weights
original_weights = model.get_weights()
print("Original weights", original_weights[0])
print("========================================================")
print("========================================================")
print("========================================================")
model_cloned = tf.keras.models.clone_model(model)
new_weights = model_cloned.get_weights()
print("New weights", new_weights[0])

# Compile cloned model that has randomized weights
model_cloned.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=model.optimizer.learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Reset model to cloned model
model = model_cloned
model_weights = model.get_weights()

# Function to reset model weights
reset_model = lambda model : model.set_weights(model_weights)



"""
Training and validation
"""
# Early stop training for convergence
convergence_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-20,
    patience=3,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
)

# Record accuracy histories for each fold
acc_histories = []

# Function to train and validate
def cross_validate(dataset, train_range, val_range):
    train_ds = tfds.load(name=dataset,
                         split=train_range,
                         shuffle_files=True,
                         as_supervised=True,
                         read_config=read_config)
    train_ds = train_ds.map(preprocess).batch(BATCH)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = tfds.load(name=dataset,
                       split=val_range,
                       shuffle_files=True,
                       as_supervised=True,
                       read_config=read_config)
    val_ds = val_ds.map(preprocess).batch(BATCH)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    acc_history = model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[convergence_callback],
        validation_data=val_ds,
    )
    acc_histories.append(acc_history)

    # Pickle in case of RAM outage on Colab
    with open('acc_histories.pkl', 'wb') as f:
        pickle.dump(acc_histories, f)
    reset_model(model)



"""
Perform k-fold cross validation
"""
for fold in range(K):
    print(f"FOLD {fold + 1}", end=": ")

    val_end = CRS_VLD_END - fold*(100/K-1)
    val_start = val_end - 100/K + 1

    val_range = f'train[{val_start}%:{val_end}%]'

    # Validate ending
    if val_end == CRS_VLD_END:
        train_start = CRS_VLD_BEGIN
        train_end = val_start
        train_range = f'train[{train_start}%:{train_end}%]'
    # Validate beginning
    elif val_start == CRS_VLD_BEGIN:
        train_start = val_end
        train_end = CRS_VLD_END
        train_range = f'train[{train_start}%:{train_end}%]'
    # Middle
    else:
        train_start = CRS_VLD_BEGIN
        train_mid1 = val_start
        train_mid2 = val_end
        train_end = CRS_VLD_END
        train_range = f'train[{train_start}%:{train_mid1}%]+train[{train_mid2}%:{train_end}%]'

    cross_validate(DATASET, train_range, val_range)



"""
Plot training and valdiation curves
"""
# Load from pickle in case of RAM outage for Colab
with open('acc_histories.pkl', 'rb') as f:
    acc = pickle.load(f)

# list all data in history
for acc in acc_histories:
    print(acc.history.keys())

fold_num = 1
for acc in acc_histories:
    # summarize history for accuracy
    plt.plot(acc.history['accuracy'])
    plt.plot(acc.history['val_accuracy'])
    title = 'Fold ' + str(fold_num) + ': model accuracy'
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.show()
    # summarize history for loss
    plt.plot(acc.history['loss'])
    plt.plot(acc.history['val_loss'])
    title = 'Fold ' + str(fold_num) + ': model loss'
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()
    plt.show()
    fold_num += 1