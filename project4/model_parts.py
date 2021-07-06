import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, activations, models, losses
from tensorflow.keras.layers import (
    Dense, Input, Dropout, Convolution1D, BatchNormalization, GlobalMaxPool1D,
    Bidirectional, LSTM, Add, ReLU, Flatten, Layer, MaxPool1D,
    GlobalAveragePooling1D)

from utils import f1_metric, bac_metric


##############################################################
# COMMON FUNCTIONALITY
##############################################################

def prepare(model, config):

    if config.schedule_lr:
        lr = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config.learning_rate,
            decay_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay_rate,
            staircase=True)
    else:
        lr = config.learning_rate

    opt = optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt, loss=config.loss_function,
        # run_eagerly=True,
        # metrics=['acc', f1_metric('weighted'),
        #          f1_metric('macro'), bac_metric()]
        metrics=['acc']
    )

    model.summary()

    return model


def modify_model(model, config):

    if config.mode == 'frozen':
        for layer in model.layers:
            layer.trainable = False

    inputs = model.input
    features = model.get_layer(
        'final_dropout' if config.mode == 'frozen' else 'dense_2').output

    if config.mode == 'frozen':

        t = BatchNormalization()(features)
        t = Dense(config.ndim, activation=activations.relu, name='dense_1')(t)

        t = BatchNormalization()(t)
        t = Dense(config.ndim, activation=activations.relu, name='dense_2')(t)

    else:
        t = features

    outputs = Dense(
        2, activation=activations.softmax, name='dense_final_ptbdb')(t)

    model = models.Model(inputs, outputs)

    model = prepare(model, config)

    return model


##############################################################
# PARTS
##############################################################

def residual_block(x, downsample, filters, kernel_size=3):

    y = Convolution1D(
        kernel_size=kernel_size,
        strides=(1 if not downsample else 2),
        filters=filters,
        padding='same')(x)

    y = BatchNormalization()(y)
    y = activations.swish(y)

    y = Convolution1D(
        kernel_size=kernel_size,
        strides=1,
        filters=filters,
        padding='same')(y)

    y = BatchNormalization()(y)

    if downsample:
        x = Convolution1D(
            kernel_size=1,
            strides=2,
            filters=filters,
            padding='same')(x)
        x = BatchNormalization()(x)

    out = Add()([x, y])

    out = BatchNormalization()(out)
    out = activations.swish(out)

    return out


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W, self.b = None, None

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='normal')
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros')
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


##############################################################
# GETTERS
##############################################################


def get_baseline_cnn_model(dataset):

    nclass = 5 if dataset == 'mit' else 1
    final_activation_fn = activations.softmax if dataset == 'mit' \
        else activations.sigmoid
    loss_fn = losses.sparse_categorical_crossentropy if dataset == 'mit' \
        else losses.binary_crossentropy

    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=final_activation_fn, name=f"dense_3_{dataset}")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(
        optimizer=opt, loss=loss_fn, metrics=['acc'])
    model.summary()
    return model


def get_basic_cnn_model(config, nclass=5, prepare_model=True):

    inputs = Input(shape=(187, 1))

    t = inputs
    for l in range(len(config.ndims)):
        t = Convolution1D(
            config.ndims[l],
            kernel_size=config.kernel_sizes[l],
            activation=activations.swish,
            padding='valid')(t)
        t = Convolution1D(
            config.ndims[l],
            kernel_size=config.kernel_sizes[l],
            activation=activations.swish,
            padding='valid')(t)
        t = MaxPool1D(pool_size=2)(t)
        t = Dropout(rate=config.dropout)(t)

    t = GlobalMaxPool1D()(t)
    t = Dropout(rate=config.dropout)(t)

    t = Dense(
        config.dense_size, activation=activations.swish, name="dense_1")(t)
    t = Dense(
        config.dense_size, activation=activations.swish, name="dense_2")(t)
    t = Dense(nclass, activation=activations.softmax, name="dense_3")(t)

    model = models.Model(inputs=inputs, outputs=t)

    if prepare_model:
        model = prepare(model, config)

    return model


def get_basic_rnn_model(config, nclass=5, prepare_model=True):
    inputs = Input(shape=(187, 1))

    t = BatchNormalization()(inputs)
    t = LSTM(config.ndim, dropout=config.rnn_dropout, name='rec_1')(t)

    t = BatchNormalization()(t)
    t = Dense(config.dense_size, activation=activations.relu, name='dense_1')(t)

    t = BatchNormalization()(t)
    t = Dense(config.dense_size, activation=activations.relu, name='dense_2')(t)

    t = BatchNormalization()(t)
    outputs = Dense(
        nclass, activation=activations.softmax, name='final_dense')(t)

    model = models.Model(inputs, outputs)

    if prepare_model:
        model = prepare(model, config)

    return model


def get_res_model(config, nclass=5, prepare_model=True):
    inputs = Input(shape=(187, 1))

    t = BatchNormalization()(inputs)
    t = Convolution1D(
        kernel_size=5,
        strides=1,
        filters=config.num_filters[0],
        activation='relu',
        padding='same')(t)
    t = BatchNormalization()(t)

    for i in range(len(config.num_blocks_list)):
        num_blocks = config.num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(
                t, downsample=(i != 0 and j == 0),
                filters=config.num_filters[i],
                kernel_size=5 if i == 0 else 3)

    # t = AveragePooling1D(pool_size=3)(t)
    # t = Flatten()(t)

    t = GlobalMaxPool1D()(t)

    t = Dropout(rate=0.2, name='final_dropout')(t)

    t = BatchNormalization()(t)
    t = Dense(config.ndim, activation=activations.relu, name='dense_1')(t)

    t = BatchNormalization()(t)
    t = Dense(config.ndim, activation=activations.relu, name='dense_2')(t)

    outputs = Dense(
        nclass, activation=activations.softmax, name='dense_final_mitbih')(t)

    model = models.Model(inputs, outputs)

    if prepare_model:
        model = prepare(model, config)

    return model


def get_rnn_model(config, nclass=5, prepare_model=True):
    inputs = Input(shape=(187, 1))

    t = BatchNormalization()(inputs)
    t = Convolution1D(
        kernel_size=5,
        strides=1,
        filters=config.ndim[0],
        activation='relu',
        padding='valid',
        name='conv_1')(t)

    t = BatchNormalization()(t)

    t = Bidirectional(
        LSTM(config.ndim[1],
             return_sequences=True,
             dropout=config.rnn_dropout), name='rec_1')(t)

    t = Bidirectional(
        LSTM(config.ndim[2],
             return_sequences=False,
             dropout=config.rnn_dropout), name='rec_2')(t)

    t = BatchNormalization()(t)
    t = Dense(config.ndim[-1], activation=activations.relu, name='dense_1')(t)

    t = BatchNormalization()(t)
    t = Dense(config.ndim[-1], activation=activations.relu, name='dense_2')(t)

    t = BatchNormalization()(t)
    outputs = Dense(
        nclass, activation=activations.softmax, name='final_dense_mitbih')(t)

    model = models.Model(inputs, outputs)

    if prepare_model:
        model = prepare(model, config)

    return model


def get_attention_model(config, nclass=5, prepare_model=True):
    inputs = Input(shape=(187, 1))

    t = BatchNormalization()(inputs)
    t = Convolution1D(
        kernel_size=config.kernel_sizes[0],
        strides=1,
        filters=config.num_filters[0],
        activation='relu',
        padding='same')(t)
    t = BatchNormalization()(t)

    for ii in range(1, len(config.num_filters)):
        t = residual_block(
            t, downsample=True,
            filters=config.num_filters[ii],
            kernel_size=config.kernel_sizes[ii],)

    t = BatchNormalization()(t)

    t = Bidirectional(LSTM(config.ndim, dropout=0.3, return_sequences=True))(t)

    ts = []
    for ii in range(config.nheads):
        ts.append(AttentionLayer(name=f'attention_layer_{ii}')(t))

    ts.append(GlobalAveragePooling1D()(t))

    t = Add()(ts)

    t = BatchNormalization()(t)
    t = Dense(config.ndim, activation=activations.relu, name='dense_1')(t)

    # t = Dropout(rate=0.1)(t)
    t = BatchNormalization()(t)
    t = Dense(config.ndim, activation=activations.relu, name='dense_2')(t)

    t = BatchNormalization()(t)
    outputs = Dense(
        nclass, activation=activations.softmax, name='final_dense_mitbih')(t)

    model = models.Model(inputs, outputs)

    if prepare_model:
        model = prepare(model, config)

    return model



##############################################################
# RUN VARIOUS EXPERIMENTS/TRAINING PROCEDURES
##############################################################


def run_add_cnn_experiment(model, train, test, skip_feats = None,
                           weight_file_path='./models/cnn_res_dilated_mitbih.h5', 
                           batch_size = 32, num_epochs = 50):
    
    optimizer = optimizers.Adam()  
    
    min_lr           = 0.00005 
    reduce_lr_factor = 0.5     
    patience         = 5
    cooldown         = 5 
    
    monitor = "val_acc" 
    mode = "max"  
    
    checkpoint = ModelCheckpoint(
        weight_file_path,
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        mode=mode
    )
    early = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=8,  
        verbose=1
    )
    redonplat = ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=reduce_lr_factor,
        patience=patience,
        cooldown=cooldown,
        min_lr=min_lr,
        verbose=2
    )

    callbacks_list = [checkpoint, early, redonplat, PlotLossesKeras()] 
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', 
        metrics=["acc"] 
    )

    if skip_feats is not None:
        X_train = [train[0], skip_feats[0]]
        X_test = [test[0], skip_feats[1]]
    else:
        X_train = train[0]
        X_test = test[0]
    
    
    history = model.fit(
        x=X_train, 
        y=train[1],
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=callbacks_list,
        shuffle = True
    )

    model.load_weights(weight_file_path)
    _, accuracy = model.evaluate(X_test, test[1])
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history