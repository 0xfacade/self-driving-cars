var gamejs = require('gamejs');
var box2d = require('./Box2dWeb-2.1.a.3');
var vectors = require('gamejs/utils/vectors');
var math = require('gamejs/utils/math');

var STEER_NONE = 0;
var STEER_RIGHT = 1;
var STEER_LEFT = 2;

var ACC_NONE = 0;
var ACC_ACCELERATE = 1;
var ACC_BRAKE = 2;

var WIDTH_PX = window.innerWidth * 0.7;   //screen width in pixels
var HEIGHT_PX = window.innerHeight; //screen height in pixels
var SCALE = 15;  //how many pixels in a meter
var WIDTH_M = WIDTH_PX / SCALE; //world width in meters. for this example, world is as large as the screen
var HEIGHT_M = HEIGHT_PX / SCALE; //world height in meters
var KEYS_DOWN = {}; //keep track of what keys are held down by the player
var b2world;

const RIGHT = 39;
const UP = 38;
const LEFT = 37;
const DOWN = 40;

const ACTION_KEYCODE = {0: [RIGHT], 1: [RIGHT, UP], 2: [UP], 3: [UP, LEFT], 4: [LEFT], 5: [LEFT, DOWN], 6: [DOWN], 7: [DOWN, RIGHT]};


const CAR_LENGTH = 4;
const CAR_WIDTH = 2;
const SENSORS = {
    right: {
        x: CAR_WIDTH / 2,
        y: 0,
        angle: 0,
        range: 8
    },
    front_right: {
        x: CAR_WIDTH / 2,
        y: -CAR_LENGTH / 2,
        angle: 315,
        range: 8
    },
    front_center: {
        x: 0,
        y: -CAR_LENGTH / 2,
        angle: 270,
        range: 16
    },
    front_left: {
        x: -CAR_WIDTH / 2,
        y: -CAR_LENGTH / 2,
        angle: 225,
        range: 8
    },
    left: {
        x: -CAR_WIDTH / 2,
        y: 0,
        angle: 180,
        range: 8
    },
    rear_left: {
        x: -CAR_WIDTH / 2,
        y: CAR_LENGTH / 2,
        angle: 135,
        range: 8
    },
    rear_center: {
        x: 0,
        y: CAR_LENGTH / 2,
        angle: 90,
        range: 16
    },
    rear_right: {
        x: CAR_WIDTH / 2,
        y: CAR_LENGTH / 2,
        angle: 45,
        range: 8
    }
};
const SENSOR_NAMES_SORTED = Object.keys(SENSORS).filter(function(s) {return SENSORS.hasOwnProperty(s)}).sort();
const SENSOR_NO_OBSTACLES = -1;

// neural network code
var num_inputs = 8;
var num_actions = 8;
var temporal_window = 1; // amount of temporal memory. 0 = agent lives in-the-moment :)
var network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

// the value function network computes a value of taking any of the possible actions
// given an input state. Here we specify one explicitly the hard way
// but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
// to just insert simple relu hidden layers.
var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
layer_defs.push({type:'regression', num_neurons:num_actions});

// options for the Temporal Difference learner that trains the above net
// by backpropping the temporal difference learning rule.
var tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 30000;
opt.start_learn_threshold = 1000;
opt.gamma = 0.7;
opt.learning_steps_total = 200000;
opt.learning_steps_burnin = 3000;
opt.epsilon_min = 0.05;
opt.epsilon_test_time = 0.05;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

const brain = new deepqlearn.Brain(num_inputs, num_actions, opt); // woohoo
window.brain = brain;

function getBrainASJson() {
    return brain.value_net.toJSON();
}

function loadBrainFromJson(json) {
    brain.value_net.fromJSON(json);
}

//initialize font to draw text with
var font = new gamejs.font.Font('16px Sans-serif');

//key bindings
var BINDINGS = {
    accelerate: gamejs.event.K_UP,
    brake: gamejs.event.K_DOWN,
    steer_left: gamejs.event.K_LEFT,
    steer_right: gamejs.event.K_RIGHT
};


var BoxProp = function (pars) {
    /*
     static rectangle shaped prop

     pars:
     size - array [width, height]
     position - array [x, y], in world meters, of center
     */
    this.size = pars.size;

    //initialize body
    var bdef = new box2d.b2BodyDef();
    bdef.position = new box2d.b2Vec2(pars.position[0], pars.position[1]);
    bdef.angle = 0;
    bdef.fixedRotation = true;
    this.body = b2world.CreateBody(bdef);

    //initialize shape
    var fixdef = new box2d.b2FixtureDef;
    fixdef.shape = new box2d.b2PolygonShape();
    fixdef.shape.SetAsBox(this.size[0] / 2, this.size[1] / 2);
    fixdef.restitution = 0.4; //positively bouncy!
    this.body.CreateFixture(fixdef);
    return this;
};

function Wheel(pars) {
    /*
     wheel object

     pars:

     car - car this wheel belongs to
     x - horizontal position in meters relative to car's center
     y - vertical position in meters relative to car's center
     width - width in meters
     length - length in meters
     revolving - does this wheel revolve when steering?
     powered - is this wheel powered?
     */

    this.position = [pars.x, pars.y];
    this.car = pars.car;
    this.revolving = pars.revolving;
    this.powered = pars.powered;

    //initialize body
    var def = new box2d.b2BodyDef();
    def.type = box2d.b2Body.b2_dynamicBody;
    def.position = this.car.body.GetWorldPoint(new box2d.b2Vec2(this.position[0], this.position[1]));
    def.angle = this.car.body.GetAngle();
    this.body = b2world.CreateBody(def);

    //initialize shape
    var fixdef = new box2d.b2FixtureDef;
    fixdef.density = 1;
    fixdef.isSensor = true; //wheel does not participate in collision calculations: resulting complications are unnecessary
    fixdef.shape = new box2d.b2PolygonShape();
    fixdef.shape.SetAsBox(pars.width / 2, pars.length / 2);
    this.body.CreateFixture(fixdef);

    //create joint to connect wheel to body
    if (this.revolving) {
        var jointdef = new box2d.b2RevoluteJointDef();
        jointdef.Initialize(this.car.body, this.body, this.body.GetWorldCenter());
        jointdef.enableMotor = false; //we'll be controlling the wheel's angle manually
    } else {
        var jointdef = new box2d.b2PrismaticJointDef();
        jointdef.Initialize(this.car.body, this.body, this.body.GetWorldCenter(), new box2d.b2Vec2(1, 0));
        jointdef.enableLimit = true;
        jointdef.lowerTranslation = jointdef.upperTranslation = 0;
    }
    b2world.CreateJoint(jointdef);


}

Wheel.prototype.setAngle = function (angle) {
    /*
     angle - wheel angle relative to car, in degrees
     */
    this.body.SetAngle(this.car.body.GetAngle() + math.radians(angle));
};

Wheel.prototype.getLocalVelocity = function () {
    /*returns get velocity vector relative to car
     */
    var res = this.car.body.GetLocalVector(this.car.body.GetLinearVelocityFromLocalPoint(new box2d.b2Vec2(this.position[0], this.position[1])));
    return [res.x, res.y];
};

Wheel.prototype.getDirectionVector = function () {
    /*
     returns a world unit vector pointing in the direction this wheel is moving
     */
    return vectors.rotate((this.getLocalVelocity()[1] > 0) ? [0, 1] : [0, -1], this.body.GetAngle());
};


Wheel.prototype.getKillVelocityVector = function () {
    /*
     substracts sideways velocity from this wheel's velocity vector and returns the remaining front-facing velocity vector
     */
    var velocity = this.body.GetLinearVelocity();
    var sideways_axis = this.getDirectionVector();
    var dotprod = vectors.dot([velocity.x, velocity.y], sideways_axis);
    return [sideways_axis[0] * dotprod, sideways_axis[1] * dotprod];
};

Wheel.prototype.killSidewaysVelocity = function () {
    /*
     removes all sideways velocity from this wheels velocity
     */
    var kv = this.getKillVelocityVector();
    this.body.SetLinearVelocity(new box2d.b2Vec2(kv[0], kv[1]));

};


function Car(pars) {
    /*
     pars is an object with possible attributes:

     width - width of the car in meters
     length - length of the car in meters
     position - starting position of the car, array [x, y] in meters
     angle - starting angle of the car, degrees
     max_steer_angle - maximum angle the wheels turn when steering, degrees
     max_speed       - maximum speed of the car, km/h
     power - engine force, in newtons, that is applied to EACH powered wheel
     wheels - wheel definitions: [{x, y, rotatable, powered}}, ...] where
     x is wheel position in meters relative to car body center
     y is wheel position in meters relative to car body center
     revolving - boolean, does this turn rotate when steering?
     powered - is force applied to this wheel when accelerating/braking?
     */

    //state of car controls
    this.steer = STEER_NONE;
    this.accelerate = ACC_NONE;

    this.max_steer_angle = pars.max_steer_angle;
    this.max_speed = pars.max_speed;
    this.power = pars.power;
    this.wheel_angle = 0;//keep track of current wheel angle relative to car.
                         //when steering left/right, angle will be decreased/increased gradually over 200ms to prevent jerkyness.

    //initialize body
    var def = new box2d.b2BodyDef();
    def.type = box2d.b2Body.b2_dynamicBody;
    def.position = new box2d.b2Vec2(pars.position[0], pars.position[1]);
    def.angle = math.radians(pars.angle);
    def.linearDamping = 0.15;  //gradually reduces velocity, makes the car reduce speed slowly if neither accelerator nor brake is pressed
    def.bullet = true; //dedicates more time to collision detection - car travelling at high speeds at low framerates otherwise might teleport through obstacles.
    def.angularDamping = 0.3;
    this.body = b2world.CreateBody(def);

    //initialize shape
    var fixdef = new box2d.b2FixtureDef();
    fixdef.density = 1.0;
    fixdef.friction = 0.3; //friction when rubbing agaisnt other shapes
    fixdef.restitution = 0.4;  //amount of force feedback when hitting something. >0 makes the car bounce off, it's fun!
    fixdef.shape = new box2d.b2PolygonShape;
    fixdef.shape.SetAsBox(pars.width / 2, pars.length / 2);
    this.body.CreateFixture(fixdef);

    //initialize wheels
    this.wheels = [];
    var wheeldef, i;
    for (i = 0; i < pars.wheels.length; i++) {
        wheeldef = pars.wheels[i];
        wheeldef.car = this;
        this.wheels.push(new Wheel(wheeldef));
    }


}

Car.prototype.getPoweredWheels = function () {
    //return array of powered wheels
    var retv = [];
    for (var i = 0; i < this.wheels.length; i++) {
        if (this.wheels[i].powered) {
            retv.push(this.wheels[i]);
        }
    }
    return retv;
};

Car.prototype.getLocalVelocity = function () {
    /*
     returns car's velocity vector relative to the car
     */
    var retv = this.body.GetLocalVector(this.body.GetLinearVelocityFromLocalPoint(new box2d.b2Vec2(0, 0)));
    return [retv.x, retv.y];
};

Car.prototype.getRevolvingWheels = function () {
    //return array of wheels that turn when steering
    var retv = [];
    for (var i = 0; i < this.wheels.length; i++) {
        if (this.wheels[i].revolving) {
            retv.push(this.wheels[i]);
        }
    }
    return retv;
};

Car.prototype.getSpeedKMH = function () {
    var velocity = this.body.GetLinearVelocity();
    var len = vectors.len([velocity.x, velocity.y]);
    return (len / 1000) * 3600;
};

Car.prototype.setSpeed = function (speed) {
    /*
     speed - speed in kilometers per hour
     */
    var velocity = this.body.GetLinearVelocity();
    velocity = vectors.unit([velocity.x, velocity.y]);
    velocity = new box2d.b2Vec2(velocity[0] * ((speed * 1000.0) / 3600.0),
        velocity[1] * ((speed * 1000.0) / 3600.0));
    this.body.SetLinearVelocity(velocity);

};

Car.prototype.update = function (msDuration) {

    //1. KILL SIDEWAYS VELOCITY

    //kill sideways velocity for all wheels
    var i;
    for (i = 0; i < this.wheels.length; i++) {
        this.wheels[i].killSidewaysVelocity();
    }

    //2. SET WHEEL ANGLE

    //calculate the change in wheel's angle for this update, assuming the wheel will reach is maximum angle from zero in 200 ms
    var incr = (this.max_steer_angle / 200) * msDuration;

    if (this.steer == STEER_RIGHT) {
        this.wheel_angle = Math.min(Math.max(this.wheel_angle, 0) + incr, this.max_steer_angle) //increment angle without going over max steer
    } else if (this.steer == STEER_LEFT) {
        this.wheel_angle = Math.max(Math.min(this.wheel_angle, 0) - incr, -this.max_steer_angle) //decrement angle without going over max steer
    } else {
        this.wheel_angle = 0;
    }

    //update revolving wheels
    var wheels = this.getRevolvingWheels();
    for (i = 0; i < wheels.length; i++) {
        wheels[i].setAngle(this.wheel_angle);
    }

    //3. APPLY FORCE TO WHEELS
    var base_vect; //vector pointing in the direction force will be applied to a wheel ; relative to the wheel.

    //if accelerator is pressed down and speed limit has not been reached, go forwards
    if ((this.accelerate == ACC_ACCELERATE) && (this.getSpeedKMH() < this.max_speed)) {
        base_vect = [0, -1];
    }
    else if (this.accelerate == ACC_BRAKE) {
        //braking, but still moving forwards - increased force
        if (this.getLocalVelocity()[1] < 0) base_vect = [0, 1.3];
        //going in reverse - less force
        else base_vect = [0, 0.7];
    }
    else base_vect = [0, 0];

    //multiply by engine power, which gives us a force vector relative to the wheel
    var fvect = [this.power * base_vect[0], this.power * base_vect[1]];

    //apply force to each wheel
    wheels = this.getPoweredWheels();
    for (i = 0; i < wheels.length; i++) {
        var position = wheels[i].body.GetWorldCenter();
        wheels[i].body.ApplyForce(wheels[i].body.GetWorldVector(new box2d.b2Vec2(fvect[0], fvect[1])), position);
    }

    //if going very slow, stop - to prevent endless sliding
    if ((this.getSpeedKMH() < 4) && (this.accelerate == ACC_NONE)) {
        this.setSpeed(0);
    }

};


/*
 *initialize car and props, start game loop
 */
function main() {

    //initialize display
    var display = gamejs.display.setMode([WIDTH_PX, HEIGHT_PX]);

    //SET UP B2WORLD
    b2world = new box2d.b2World(new box2d.b2Vec2(0, 0), false);

    //set up box2d debug draw to draw the bodies for us.
    //in a real game, car will propably be drawn as a sprite rotated by the car's angle
    var debugDraw = new box2d.b2DebugDraw();
    debugDraw.SetSprite(display._canvas.getContext("2d"));
    debugDraw.SetDrawScale(SCALE);
    debugDraw.SetFillAlpha(0.5);
    debugDraw.SetLineThickness(1.0);
    debugDraw.SetFlags(box2d.b2DebugDraw.e_shapeBit);
    b2world.SetDebugDraw(debugDraw);

    //initialize car
    var car = new Car({
        'width': CAR_WIDTH,
        'length': CAR_LENGTH,
        'position': [WIDTH_M * 0.75, HEIGHT_M * 0.75],
        'angle': 0,
        'power': 30,
        'max_steer_angle': 25,
        'max_speed': 40,
        'wheels': [{'x': -1, 'y': -1.2, 'width': 0.4, 'length': 0.8, 'revolving': true, 'powered': true}, //top left
            {'x': 1, 'y': -1.2, 'width': 0.4, 'length': 0.8, 'revolving': true, 'powered': true}, //top right
            {'x': -1, 'y': 1.2, 'width': 0.4, 'length': 0.8, 'revolving': false, 'powered': false}, //back left
            {'x': 1, 'y': 1.2, 'width': 0.4, 'length': 0.8, 'revolving': false, 'powered': false}]
    }); //back right

    //initialize some props to bounce against
    var props = [];

    //outer walls
    props.push(new BoxProp({'size': [WIDTH_M, 1], 'position': [WIDTH_M / 2, 0.5]}));
    props.push(new BoxProp({'size': [1, HEIGHT_M - 2], 'position': [0.5, HEIGHT_M / 2]}));
    props.push(new BoxProp({'size': [WIDTH_M, 1], 'position': [WIDTH_M / 2, HEIGHT_M - 0.5]}));
    props.push(new BoxProp({'size': [1, HEIGHT_M - 2], 'position': [WIDTH_M - 0.5, HEIGHT_M / 2]}));


    /*
    //pen in the center
    var center = [WIDTH_M / 2, HEIGHT_M / 2];
    props.push(new BoxProp({'size': [1, 6], 'position': [center[0] - 3, center[1]]}));
    props.push(new BoxProp({'size': [1, 6], 'position': [center[0] + 3, center[1]]}));
    props.push(new BoxProp({'size': [5, 1], 'position': [center[0], center[1] + 2.5]}));
    */

    var rayCastResults = {};
    function resetRayCastResults() {
        for(var sensorName in SENSORS) {
            if (!SENSORS.hasOwnProperty(sensorName)) continue;
            rayCastResults[sensorName] = SENSORS[sensorName].range;
        }
    }
    resetRayCastResults();

    function rayCastCallback(sensor, fixture, point, normal, fraction) {
        rayCastResults[sensor] = fraction * SENSORS[sensor].range;
        return fraction;
    }

    var currentPolygon = [];
    var currentMousePosition = new box2d.b2Vec2(0, 0);
    function drawPolygon(event) {
        const worldX = event.clientX / SCALE;
        const worldY = event.clientY / SCALE;
        currentPolygon.push(new box2d.b2Vec2(worldX, worldY));
        const distance = currentPolygon[0].Copy();
        distance.Subtract(currentPolygon[currentPolygon.length - 1]);
        if (currentPolygon.length > 3 && distance.Length() < 1) {
            finishPolygon();
        }
    }
    function updateMouseLocation(event) {
        currentMousePosition.x = event.clientX / SCALE;
        currentMousePosition.y = event.clientY / SCALE;
    }

    display._canvas.onclick = drawPolygon.bind(this);
    display._canvas.onmousemove = updateMouseLocation.bind(this);
    function finishPolygon() {
        var polygon = currentPolygon.slice(0, currentPolygon.length - 1).map(function(v) {
            return [v.x, v.y];
        });
        decomp.makeCCW(polygon);
        var convexPolygons = decomp.quickDecomp(polygon);
        for(var k = 0; k < convexPolygons.length; k++) {
            var convertedPolygon = convexPolygons[k].map(function(v){
                return new box2d.b2Vec2(v[0], v[1]);
            });
            //initialize body
            var bdef = new box2d.b2BodyDef();
            bdef.angle = 0;
            bdef.fixedRotation = true;
            var body = b2world.CreateBody(bdef);

            //initialize shape
            var fixdef = new box2d.b2FixtureDef();
            fixdef.shape = new box2d.b2PolygonShape();
            fixdef.shape.SetAsArray(convertedPolygon, convertedPolygon.length);
            fixdef.restitution = 0.4; //positively bouncy!
            body.CreateFixture(fixdef);
        }
        currentPolygon = [];
    }

    function sendKeyEvent(name, keycode) {
        var keyboardEvent = document.createEvent("KeyboardEvent");
        var initMethod = typeof keyboardEvent.initKeyboardEvent !== 'undefined' ? "initKeyboardEvent" : "initKeyEvent";
        keyboardEvent[initMethod](
                            name, // event type : keydown, keyup, keypress
                            true, // bubbles
                            true, // cancelable
                            window, // viewArg: should be window
                            false, // ctrlKeyArg
                            false, // altKeyArg
                            false, // shiftKeyArg
                            false, // metaKeyArg
                            keycode, // keyCodeArg : unsigned long the virtual key code, else 0
                            0 // charCodeArgs : unsigned long the Unicode character associated with the depressed key, else 0
        );
        document.dispatchEvent(keyboardEvent);
    }

    var autoSteeringOn = false;
    var framesSinceLastReward = 0;
    var framesSinceLastDecision = 0;
    $('#toggleAutopilot').click(function(){
        autoSteeringOn = !autoSteeringOn;
        $('#autopilotIndicator').text(autoSteeringOn ? 'on' : 'off');
    });

    var isRecordingData = false;
    var recordedData = [];
    var lastPredictionPressedKeys = [];

    var averageSpeed = 0;
    var framesReversing = 0;
    var framesGoingStraight = 0;
    var framesCrashing = 0;
    var lastCarPosition = car.body.GetWorldCenter();

    function tick(msDuration) {
        //GAME LOOP

        //set car controls according to player input
        if (KEYS_DOWN[BINDINGS.accelerate]) {
            car.accelerate = ACC_ACCELERATE;
        } else if (KEYS_DOWN[BINDINGS.brake]) {
            car.accelerate = ACC_BRAKE;
        } else {
            car.accelerate = ACC_NONE;
        }

        if (KEYS_DOWN[BINDINGS.steer_right]) {
            car.steer = STEER_RIGHT;
        } else if (KEYS_DOWN[BINDINGS.steer_left]) {
            car.steer = STEER_LEFT;
        } else {
            car.steer = STEER_NONE;
        }

        //update car
        car.update(msDuration);

        //update physics world
        b2world.Step(msDuration / 1000, 10, 8);

        //clear applied forces, so they don't stack from each update
        b2world.ClearForces();

        //fill background
        gamejs.draw.rect(display, '#FFFFFF', new gamejs.Rect([0, 0], [WIDTH_PX, HEIGHT_PX]), 0)

        //let box2d draw it's bodies
        b2world.DrawDebugData();

        resetRayCastResults();
        var carAngle = car.body.GetAngle();
        for(sensorName in SENSORS) {
            if (!SENSORS.hasOwnProperty(sensorName)) continue;
            const sensor = SENSORS[sensorName];
            const sensorAngle = sensor.angle / 180 * Math.PI;
            const rayStart = car.body.GetWorldPoint(new box2d.b2Vec2(sensor.x, sensor.y));
            // b2Body.GetAngle(): angle between y axis and body orientation, clockwise (!)
            const ray = new box2d.b2Vec2(Math.cos(carAngle + sensorAngle), Math.sin(carAngle + sensorAngle));
            ray.Multiply(sensor.range);
            const rayEnd = rayStart.Copy();
            rayEnd.Add(ray);

            b2world.RayCast(rayCastCallback.bind(this, sensorName), rayStart, rayEnd);
            debugDraw.DrawSegment(rayStart, rayEnd, [0, 0, 0]);
        }

        //fps and car speed display
        display.blit(font.render('FPS: ' + parseInt((1000) / msDuration)), [25, 25]);
        display.blit(font.render('Speed: ' + parseInt(Math.floor(car.getSpeedKMH())) + ' km/h'), [25, 55]);
        var i = 0;
        for(sensorName in SENSORS) {
            if (!SENSORS.hasOwnProperty(sensorName)) continue;
            var text = rayCastResults[sensorName] == SENSOR_NO_OBSTACLES
                ? '-'
                : (Math.round(rayCastResults[sensorName] * 100) / 100) + 'm';
            display.blit(font.render(sensorName + ': '  + text), [25, 85 + 30 * i]);
            i++;
        }

        for(var j = 1; j < currentPolygon.length; j++) {
            debugDraw.DrawSegment(currentPolygon[j-1], currentPolygon[j], [0, 0, 0]);
        }
        if (currentPolygon.length >= 1) {
            debugDraw.DrawSegment(currentPolygon[currentPolygon.length - 1], currentMousePosition, [0, 0, 0]);
        }

        if (isRecordingData) {
            const telemetry = {
                wheel_angle: car.wheel_angle,
                velocity: car.body.GetLinearVelocity(),
                sensors: rayCastResults,
                pressed_keys: Object.keys(KEYS_DOWN).filter(function(k) {
                    return KEYS_DOWN.hasOwnProperty(k) && KEYS_DOWN[k]
                }).map(function(k) {
                    return parseInt(k);
                }).filter(function(k){
                    return k >= 37 && k <= 40;
                })
            };
            recordedData.push(telemetry);
        }


        const rewardRate = 40;
        const decisionRate = 10;
        if (autoSteeringOn) {

            if (framesSinceLastDecision >= decisionRate) {
                modelInput = [];
                SENSOR_NAMES_SORTED.forEach(function (sensorName) {
                    modelInput.push(rayCastResults[sensorName] / SENSORS[sensorName].range);
                });

                const action = brain.forward(modelInput); // action is an int in [0, num_actions)
                const newKeys = ACTION_KEYCODE[action];

                for (var l = 0; l < lastPredictionPressedKeys.length; l++) {
                    if (newKeys.indexOf(lastPredictionPressedKeys[l]) != -1) {
                        continue;
                    }
                    sendKeyEvent('keyup', lastPredictionPressedKeys[l]);
                }
                lastPredictionPressedKeys = [];
                lastPredictionPressedKeys = lastPredictionPressedKeys.concat(newKeys);
                framesSinceLastDecision = 0;
            }
            framesSinceLastDecision++;

            for (var l = 0; l < lastPredictionPressedKeys.length; l++) {
                sendKeyEvent('keydown', lastPredictionPressedKeys[l]);
            }

            // Update the statistics which will be used by the reward function
            var currentVelocity = car.body.GetLinearVelocity();
            var currentSpeed = currentVelocity.Length();
            averageSpeed += currentSpeed;
            // Check if we are going forwards or backwards.
            if (currentSpeed > 0) {
                const straight = car.body.GetWorldPoint(new box2d.b2Vec2(0, -CAR_LENGTH / 2));
                straight.Subtract(car.body.GetWorldPoint(new box2d.b2Vec2(0, 0)));
                var angleToStraight = Math.acos((straight.x * currentVelocity.x + straight.y * currentVelocity.y)
                        / (straight.Length() * currentSpeed)) / Math.PI * 180;
                if (angleToStraight >= 90) {
                    // We are going backwards.
                    framesReversing++;
                } else {
                    framesGoingStraight++;
                }
            }

            // check for crashes
            var closestObstacleDistance = 100000;
            SENSOR_NAMES_SORTED.forEach(function(sensorName){
                if(rayCastResults[sensorName] < closestObstacleDistance) {
                    closestObstacleDistance = rayCastResults[sensorName];
                }
            });
            if (closestObstacleDistance <= 0.1) {
                framesCrashing++;
            }

            if (framesSinceLastReward >= rewardRate){
                averageSpeed = averageSpeed / framesSinceLastReward;
                const traveled = car.body.GetWorldCenter().Copy();
                traveled.Subtract(lastCarPosition);
                const distanceTraveled = traveled.Length();
                lastCarPosition = car.body.GetWorldCenter().Copy();

                var reward = (-8 * framesCrashing / framesSinceLastReward)
                    + (1 * framesGoingStraight / framesSinceLastReward * averageSpeed / 2);
                console.log(reward);
                brain.backward(reward);

                framesSinceLastReward = 0;
                framesGoingStraight = 0;
                framesCrashing = 0;
                framesReversing = 0;
                averageSpeed = 0;
            }
            framesSinceLastReward++;
        }


        return;
    };



    function startDownloadData() {
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(recordedData));
        var dlAnchorElem = document.getElementById('downloadAnchor');
        dlAnchorElem.setAttribute("href", dataStr);
        dlAnchorElem.setAttribute("download", "data_" + Date.now() + ".json");
        dlAnchorElem.click();
    }

    function handleEvent(event) {
        if (event.type === gamejs.event.KEY_DOWN) {
            KEYS_DOWN[event.key] = true;
        } else if (event.type === gamejs.event.KEY_UP) {
            if (event.key === gamejs.event.K_SPACE) {
                if(isRecordingData) {
                    isRecordingData = false;
                    startDownloadData();
                    recordedData = [];
                } else {
                    isRecordingData = true;
                }
            }
            KEYS_DOWN[event.key] = false;
        }
    }

    gamejs.onTick(tick, this);
    gamejs.onEvent(handleEvent.bind(this));

}

gamejs.ready(main);

