import tensorflow as tf
import numpy as np

NUM_EPOCHS = 3

if __name__ == "__main__":
    X = np.array([[0, 0], [3, 1], [0, 1], [0, 3], [2, 2]])
    y = np.array([0, 1, 0, 1, 1])

    kernel_init = tf.constant_initializer([[1], [1]])
    bias_init = tf.constant_initializer([1])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                1,
                activation="relu",
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
            )
        ]
    )

    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for epoch in range(NUM_EPOCHS):
        with tf.GradientTape() as tape:
            preds = model(X, training=True)
            loss = loss_fn(y, preds)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        w, b = model.layers[0].get_weights()
        print(f"Epoch {epoch+1}")
        print("Loss:", float(loss))
        print("Gradients:", [g.numpy() for g in grads])
        print("Weights:", w)
        print("Bias:", b)
