import tensorflow as tf
from tensorflow.keras import layers, models

# Example: U-Net model for segmentation
def unet_model(input_size=(128,128,3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2,2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D((2,2))(c3)
    concat1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(concat1)

    u2 = layers.UpSampling2D((2,2))(c4)
    concat2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(concat2)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
