export const RADIUS = 150, ROAD_WIDTH = 80, SPEED = 2, MAX_TURN = 0.06;
export function wrap(a) { return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; }
export function observe(car) { return {offset: Math.hypot(car.x, car.y) - RADIUS, angle: wrap(car.heading - Math.atan2(car.y, car.x) - Math.PI / 2)}; }
export function predict(model, inputs) {
  return model.weights.reduce((values, layer, l) => layer.map((row, j) => {
    const z = row.reduce((sum, weight, i) => sum + weight * values[i], model.biases[l][j]);
    return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  }), inputs);
}
export function makeCar(offset = 20, angle = 0.15) { return {x: RADIUS + offset, y: 0, heading: Math.PI / 2 + angle, steps: 0, offRoad: false}; }
export function step(car, model) {
  const {offset, angle} = observe(car);
  const output = predict(model, [offset / (ROAD_WIDTH / 2), angle / 0.6])[0];
  car.heading = wrap(car.heading + (2 * output - 1) * MAX_TURN);
  car.x += SPEED * Math.cos(car.heading); car.y += SPEED * Math.sin(car.heading); car.steps++;
  car.offRoad = Math.abs(observe(car).offset) > ROAD_WIDTH / 2;
}
