from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense

class basicModel:
    # When creating neural network number of observations == input neurons number and number of neurons on final layer == actions

    # Traiing Model network - Value Network
    def __init__(self,observations):
        self.no_observations = observations
      
    model = Sequential()
    #Add layers
    model.add(Dense(4, input_shape=(1,self.no_observations)))
    model.add(Activation('relu'))

    #  Input shape must be provided defined in first layer to compile properly but can define at any layer
    model.add(Dense(16, input_shape=(1,self.no_observations)))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(16))
    model.add(Activation('relu'))

    # output or last layer gives number of actions
    model.add(Dense(2))
    model.add(Activation('linear'))