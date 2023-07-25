from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention, RNN, LSTM, GRU, Dropout, AlphaDropout, GaussianDropout, \
    AdditiveAttention, \
    ActivityRegularization, Bidirectional, TimeDistributed, LayerNormalization, BatchNormalization, GaussianNoise, \
    InputLayer, Dense, Layer, Concatenate, SimpleRNNCell, Add, SimpleRNN, Input
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.activations import selu, sigmoid
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def deep_learning_model():
    # Input Layer
    input_layer = Input(shape=(64, 161))

    # First Level Layers
    gru_layer_1 = GRU(units=128, activation=selu, dropout=0.2, return_sequences=True,
                      kernel_initializer="glorot_uniform", activity_regularizer=l2(0.1))(input_layer)

    lstm_layer_1 = LSTM(units=128, activation=selu, dropout=0.2, return_sequences=True,
                        kernel_initializer="glorot_uniform", activity_regularizer=l2(0.1))(input_layer)

    bilstm_layer_1 = Bidirectional(LSTM(units=128, activation=selu, dropout=0.2, return_sequences=True,
                                        kernel_initializer="glorot_uniform", activity_regularizer=l2(0.1)))(input_layer)
    lstm_peep_layer_1 = RNN(cell=PeepholeLSTMCell(units=128, activation=selu, dropout=0.2,
                                                  return_sequences=True, kernel_initializer="glorot_uniform",
                                                  activity_regularizer=l2(0.1)))(input_layer)

    # Concatenate the First Lot of Layers
    concatenated_layer_1 = Concatenate()([gru_layer_1, lstm_layer_1, bilstm_layer_1, lstm_peep_layer_1])

    # Bottleneck Chained Layers to Capture Most Important Information
    gru_layer_2 = GRU(units=64, activation=selu, dropout=0.1, return_sequences=False,
                      kernel_initializer="glorot_uniform", activity_regularizer=l2(0.1))(concatenated_layer_1)
    lstm_layer_2 = LSTM(units=64, activation=selu, dropout=0.1, return_sequences=False,
                        kernel_initializer="glorot_uniform", activity_regularizer=l2(0.1))(concatenated_layer_1)
    bilstm_layer_2 = Bidirectional(
        LSTM(units=64, activation=selu, dropout=0.1, return_sequences=False, kernel_initializer="glorot_uniform",
             activity_regularizer=l2(0.1)))(concatenated_layer_1)
    lstm_peep_layer_2 = RNN(cell=PeepholeLSTMCell(units=64, activation=selu, dropout=0.1, return_sequences=False,
                                                  kernel_initializer="glorot_uniform", activity_regularizer=l2(0.1)))(
        concatenated_layer_1)

    # Concatenate the Bottleneck Layers
    concatenated_layer_2 = Concatenate()([gru_layer_2, lstm_layer_2, bilstm_layer_2, lstm_peep_layer_2])

    # Last Level of Cells
    gru_layer_3 = GRU(units=128, activation=selu, dropout=0.2, return_sequences=False,
                      kernel_initializer="glorot_uniform", activity_regularizer=l2(0.1))(concatenated_layer_2)
    lstm_layer_3 = LSTM(units=128, activation=selu, dropout=0.2, return_sequences=False,
                        kernel_initializer="glorot_uniform", activity_regularizer=l2(0.1))(concatenated_layer_2)
    bilstm_layer_3 = Bidirectional(LSTM(units=128, activation=selu, dropout=0.2,
                                        return_sequences=False, kernel_initializer="glorot_uniform",
                                        activity_regularizer=l2(0.1)))(concatenated_layer_2)
    lstm_peep_layer_3 = RNN(cell=PeepholeLSTMCell(units=128, activation=selu, dropout=0.2,
                                                  return_sequences=False, kernel_initializer="glorot_uniform",
                                                  activity_regularizer=l2(0.1)))(concatenated_layer_2)

    # Concatenate the Last Layer
    concatenated_layer_3 = Concatenate()([gru_layer_3, lstm_layer_3, bilstm_layer_3, lstm_peep_layer_3])

    # Skip Connection between First level and last level so last layer gets information from
    # the first layer and the bottleneck layer
    skip_connection = Add()([concatenated_layer_1, concatenated_layer_3])

    # Final Part of Model
    output_layer = Dense(units=1, activation='sigmoid')(skip_connection)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Nadam(), loss="binary_crossentropy", metrics=["accuracy"])

    return model


m = deep_learning_model()
# m.fit(X_tf_train, y_tf_train, batch_size=32, epochs=500, verbose=1, class_weight="balanced", use_multiprocessing=True)
