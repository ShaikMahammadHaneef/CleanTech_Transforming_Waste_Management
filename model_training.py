from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
train_dir = 'dataset/train'
val_dir   = 'dataset/test'
save_path = 'healthy_vs_rotten.h5'

# Data generators
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir,   target_size=(224,224), batch_size=32, class_mode='categorical')

# Build model
base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False

model = Sequential([
    base,
    Flatten(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
mc = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss')
es = EarlyStopping(patience=3, restore_best_weights=True)

# Train
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[mc, es])

print(f"Model saved to {save_path}")
