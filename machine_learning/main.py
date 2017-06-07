
import json
import math

from keras.models import Sequential
from keras.layers import Dense

filepath = "/home/ocius/Downloads/data_1496864215997.json"
fp = open(filepath)
raw_data = json.load(fp)
print("Training on sample set of size " + str(len(raw_data)))

max_wheel_angle = 25
max_speed = 40
long_range_sensors = ['front_center', 'rear_center']
long_range = 16
short_range = 8

key_right = 39
key_up = 38
key_left = 37
key_down = 40

tanh_max = 0.9

print(json.dumps(raw_data[0], indent=4))

def normalizeDataPoint(r):
    relative_wheel_angle = r["wheel_angle"] / max_wheel_angle
    relative_velocity = math.sqrt(r["velocity"]["x"] ** 2 + r["velocity"]["y"] ** 2) / max_speed
    normalized = [relative_velocity, relative_wheel_angle]
    for sensorName in sorted(r["sensors"].keys()):
        sensor_range = long_range if sensorName in long_range_sensors else short_range
        sensor_value = r["sensors"][sensorName]
        normalized.append(float(sensor_value / sensor_range if sensor_value > 0 else 1))
    return normalized

def extractOutput(r):
    normalized = []
    if key_right in r["keys"] and r["keys"][key_right]:
        normalized.append(tanh_max)
    else:
        normalized.append(-tanh_max)

    if key_up in r["keys"] and r["keys"][key_up]:
        normalized.append(tanh_max)
    else:
        normalized.append(-tanh_max)

    if key_left in r["keys"] and r["keys"][key_left]:
        normalized.append(tanh_max)
    else:
        normalized.append(-tanh_max)

    if key_down in r["keys"] and r["keys"][key_down]:
        normalized.append(tanh_max)
    else:
        normalized.append(-tanh_max)

    return normalized


normalized_data = [normalizeDataPoint(r) for r in raw_data]
normalized_output = [extractOutput(r) for r in raw_data]

training_set_size = int(0.75 * len(normalized_data))

training_set_x = normalized_data[:training_set_size]
training_set_y = normalized_output[:training_set_size]
test_set_x = normalized_data[training_set_size:]
test_set_y = normalized_output[training_set_size:]

print(normalized_data[0])
print(normalized_output[1])

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(10,)))
# model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(4, activation='tanh'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

batch_size = 512
epochs = 20

history = model.fit(training_set_x, training_set_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(test_set_x, test_set_y))
score = model.evaluate(test_set_x, test_set_y, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
