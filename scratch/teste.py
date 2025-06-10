import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras
from tensorflow.keras.models import clone_model
import tempfile
import os
import numpy as np
def create_model_api(dataset_name, num_classes_override=None, seed=None, config=None): # Adicionado config
    #if config is None: config = FL_STATE.get('config', get_default_args()) # Fallback

    if dataset_name == 'mnist' or dataset_name == 'emnist_digits':
        input_shape = (28, 28, 1)
        num_classes = 10
    elif dataset_name == 'emnist_char':
        input_shape = (28, 28, 1)
        num_classes = 62
    elif dataset_name == 'cifar10':
        input_shape = (32, 32, 3)
        num_classes = 10
    #else:
        #app.logger.error(f"Dataset desconhecido para criação de modelo: {dataset_name}")
        #raise ValueError(f"Dataset desconhecido para criação de modelo: {dataset_name}")

    if num_classes_override is not None:
        num_classes = num_classes_override

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if dataset_name == 'cifar10':
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
    #app.logger.info(f"Model created for dataset '{dataset_name}' with input shape {input_shape} and {num_classes} classes.")
    return model
def evaluate_model(interpreter, test_images, test_labels):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for i, test_image in enumerate(test_images):
        if i % 1000 == 0:
            print(f'Evaluated on {i} results so far.')

        # Pre-processing: add batch dimension and channel dimension (for grayscale images)
        test_image = np.expand_dims(test_image, axis=-1)  # Add the channel dimension (height, width, 1)
        test_image = np.expand_dims(test_image, axis=0)   # Add the batch dimension (1, height, width, 1)
        test_image = test_image.astype(np.float32)        # Ensure it's the correct dtype

        # Verifique as dimensões de test_image antes de passá-la para o modelo
        #print(f"Shape of test_image before inference: {test_image.shape}")

        interpreter.set_tensor(input_index, test_image)

        # Run inference
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)  # Get the output tensor
        digit = np.argmax(output()[0])  # Find the predicted class (digit)
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()  # Calculate accuracy
    return accuracy

def structurally_prune_model(original_model, dataset_name, conv_prune_ratio=0.3, dense_prune_ratio=0.5):
    """Prunes entire filters from Conv2D layers and units from Dense layers"""
    
    # Clone the model to preserve original
    model = clone_model(original_model)
    model.set_weights(original_model.get_weights())
    
    # Get layer-wise pruning ratios
    prune_ratios = {
        'conv2d': conv_prune_ratio,
        'dense': dense_prune_ratio
    }
    
    # Special handling for CIFAR10's second conv block
    if dataset_name == 'cifar10':
        prune_ratios['conv2d_1'] = conv_prune_ratio
    
    # Create new model with pruned architecture
    new_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Calculate remaining filters
            layer_name = layer.name.lower()
            ratio = prune_ratios.get(layer_name, prune_ratios['conv2d'])
            remaining_filters = int(layer.filters * (1 - ratio))
            
            # Add pruned conv layer
            new_layer = tf.keras.layers.Conv2D(
                remaining_filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=layer.activation,
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
            new_layers.append(new_layer)
            
        elif isinstance(layer, tf.keras.layers.Dense) and i != len(model.layers)-1:
            # Don't prune output layer
            remaining_units = int(layer.units * (1 - prune_ratios['dense']))
            
            new_layer = tf.keras.layers.Dense(
                remaining_units,
                activation=layer.activation,
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
            new_layers.append(new_layer)
            
        else:
            # Copy non-prunable layers (MaxPool, Flatten, Output)
            new_layers.append(layer.__class__.from_config(layer.get_config()))
    
    # Build new model
    pruned_model = tf.keras.Sequential(new_layers)
    pruned_model.build(input_shape=original_model.input_shape)
    
    return pruned_model

def prune_model_layers(original_model, dataset_name, remove_second_conv=False):
    """Remove entire layers from the model"""
    
    new_layers = []
    
    for i, layer in enumerate(original_model.layers):
        # Skip second conv block if pruning CIFAR10 model
        if dataset_name == 'cifar10' and remove_second_conv:
            if isinstance(layer, tf.keras.layers.Conv2D) and i == 2:
                continue
            if isinstance(layer, tf.keras.layers.MaxPooling2D) and i == 3:
                continue
                
        new_layers.append(layer.__class__.from_config(layer.get_config()))
    
    pruned_model = tf.keras.Sequential(new_layers)
    pruned_model.build(input_shape=original_model.input_shape)
    
    return pruned_model

if __name__ == "__main__":
    m = create_model_api("mnist",10)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    pruned_model = structurally_prune_model(
    m, 
    dataset_name='mnist',
    conv_prune_ratio=0.4,  # Remove 40% of conv filters
    dense_prune_ratio=0.5  # Remove 50% of dense units
)
    # for new_layer, original_layer in zip(pruned_model.layers, m.layers):
    #     if hasattr(new_layer, 'kernel'):
    #         original_weights = original_layer.get_weights()
    #         # For conv layers, keep first N filters
    #         if isinstance(original_layer, tf.keras.layers.Conv2D):
    #             n_filters = new_layer.kernel.shape[-1]
    #             new_weights = [w[..., :n_filters] for w in original_weights]
    #         # For dense layers, keep first N units
    #         elif isinstance(original_layer, tf.keras.layers.Dense):
    #             n_units = new_layer.kernel.shape[-1]
    #             new_weights = [w[:, :n_units] for w in original_weights]
    #         else:
    #             new_weights = original_weights
    #         new_layer.set_weights(new_weights)
            
    pruned_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
    pruned_model.fit(
  train_images,
  train_labels,
  epochs=3,
  validation_split=0.1,
)
    _, pruning = tempfile.mkstemp('.tflite')
    float_converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    float_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    float_tflite_model = float_converter.convert()
    with open(pruning, 'wb') as f:
        f.write(float_tflite_model)
    
    print("prunned model in Mb:", os.path.getsize(pruning)/float(2**20))
    ## pruning não estruturado
    
#     # 3. Apply pruning AFTER compiling but BEFORE fitting 
#     prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# # Define pruning parameters
#     pruning_params = {
#     'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
#         initial_sparsity=0.50,
#         final_sparsity=0.90,
#         begin_step=0,
#         end_step=1000  # Should match your expected number of steps
#     )
# }

# # Apply pruning to the model
#     model = prune_low_magnitude(m, **pruning_params)
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # 4. Add the pruning callback
#     callbacks = [
#     tfmot.sparsity.keras.UpdatePruningStep()
# ]
#     model.fit(train_images, train_labels,
#           epochs=3,
#           callbacks=callbacks)
#     final_model = tfmot.sparsity.keras.strip_pruning(model)
#     _, pruning = tempfile.mkstemp('.tflite')
#     float_converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
#     #float_converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     float_tflite_model = float_converter.convert()
#     with open(pruning, 'wb') as f:
#         f.write(float_tflite_model)
    
#     print("prunned model in Mb:", os.path.getsize(pruning)/float(2**20))
#     print("hentai")
    
    m.fit(
  train_images,
  train_labels,
  epochs=6,
  validation_split=0.1,
)
    
    print(m.summary())
    #quantized_model = quantize_model(m, keep_quantized=True)
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(m)
    q_aware_model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    train_images_subset = train_images[0:1000] # out of 60000
    train_labels_subset = train_labels[0:1000]

    #q_aware_model.fit(train_images_subset, train_labels_subset,
                  #batch_size=500, epochs=1, validation_split=0.1)
    
    _, baseline_model_accuracy = m.evaluate(
    test_images, test_labels, verbose=0)

    _, q_aware_model_accuracy = q_aware_model.evaluate(
   test_images, test_labels, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)
    print('Quant test accuracy:', q_aware_model_accuracy)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_tflite_model = converter.convert()

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_tflite_model = converter.convert()
    
    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()

    test_accuracy = evaluate_model(interpreter, test_images, test_labels)

    print('Quant TFLite test_accuracy:', test_accuracy)
    print('Quant TF test accuracy:', q_aware_model_accuracy)
    
    float_converter = tf.lite.TFLiteConverter.from_keras_model(m)
    float_tflite_model = float_converter.convert()
    
    _, float_file = tempfile.mkstemp('.tflite')
    _, quant_file = tempfile.mkstemp('.tflite')
    _, dequant_file = tempfile.mkstemp('.tflite')
    with open(float_file, 'wb') as f:
        f.write(float_tflite_model)

# Criar um modelo quantizado (com base em uma quantização, se necessário)
# Se você tiver um modelo quantizado, faça a conversão e salve como mostrado abaixo:
# Exemplo de quantização do modelo

    quant_converter = tf.lite.TFLiteConverter.from_keras_model(m)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_tflite_model = quant_converter.convert()
    with open(quant_file, 'wb') as f:
        f.write(quant_tflite_model)

    print("Float model in Mb:", os.path.getsize(float_file)/float(2**20))
    print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))
    #print(quantized_tflite_model.summary())