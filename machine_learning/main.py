
import json
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout

filepath = "/home/ocius/Downloads/data_1496923624387.json"
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

prob_key_max = 0.95
prob_key_min = 0.05

def normalizeDataPoint(r):
    """
    relative_wheel_angle = (r["wheel_angle"] + max_wheel_angle) / (2 * max_wheel_angle) # 0: all left, 0.5: straigt, 1: right
    relative_velocity = math.sqrt(r["velocity"]["x"] ** 2 + r["velocity"]["y"] ** 2) / max_speed # 0: stop, 1: full speed
    normalized = [relative_velocity, relative_wheel_angle]
    """
    normalized = []
    for sensorName in sorted(r["sensors"].keys()):
        sensor_range = long_range if sensorName in long_range_sensors else short_range
        sensor_value = r["sensors"][sensorName]
        normalized.append(float(sensor_value / sensor_range if sensor_value > 0 else 1)) # 0: very close 1: very far / no obstacle
    return normalized

def extractOutput(r):
    normalized = []
    if key_right in r["pressed_keys"]:
        normalized.append(prob_key_max)
    else:
        normalized.append(prob_key_min)

    if key_up in r["pressed_keys"]:
        normalized.append(prob_key_max)
    else:
        normalized.append(prob_key_min)

    if key_left in r["pressed_keys"]:
        normalized.append(prob_key_max)
    else:
        normalized.append(prob_key_min)

    if key_down in r["pressed_keys"]:
        normalized.append(prob_key_max)
    else:
        normalized.append(prob_key_min)

    return normalized


normalized_data = [normalizeDataPoint(r) for r in raw_data]
normalized_output = [extractOutput(r) for r in raw_data]

training_set_size = int(0.75 * len(normalized_data))

training_set_x = normalized_data[:training_set_size]
training_set_y = normalized_output[:training_set_size]
test_set_x = normalized_data[training_set_size:]
test_set_y = normalized_output[training_set_size:]
"""
for r, x, y in zip(raw_data, training_set_x, training_set_y):
    print(r)
    input()
    print(x)
    input()
    print(y)
    input()

print(normalized_data[0])
print(normalized_output[1])

"""

model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(8,)))
model.add(Dense(24, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mean_squared_error'])

batch_size = 512
epochs = 100

history = model.fit(training_set_x, training_set_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(test_set_x, test_set_y))
score = model.evaluate(test_set_x, test_set_y, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights('model.hdf5')
with open('model.json', 'w') as f:
    f.write(model.to_json())
