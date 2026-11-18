"""Train a tiny network to steer a car around a circular simulated road."""

import argparse
import math
import random

from neural_network import NeuralNetwork

RADIUS = 150.0
ROAD_WIDTH = 80.0
SPEED = 2.0
MAX_TURN = 0.06  # radians per step


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def observations(x, y, heading):
    """Return signed distance from the lane center and angle from its tangent."""
    radial_angle = math.atan2(y, x)
    return math.hypot(x, y) - RADIUS, wrap(heading - (radial_angle + math.pi / 2))


def inputs_for(offset, angle):
    # Fixed scales keep the network's inputs near the range seen during training.
    return [offset / (ROAD_WIDTH / 2), angle / 0.6]


def teacher(offset, angle):
    """A hand-written controller supplies steering labels for supervised learning."""
    turn = SPEED / RADIUS + 0.0012 * offset - 0.09 * angle
    return max(-MAX_TURN, min(MAX_TURN, turn))


def train_driver(seed=7):
    rng = random.Random(seed)
    samples = []
    for _ in range(350):
        offset = rng.uniform(-ROAD_WIDTH / 2, ROAD_WIDTH / 2)
        angle = rng.uniform(-0.6, 0.6)
        target = (teacher(offset, angle) / MAX_TURN + 1) / 2
        samples.append((inputs_for(offset, angle), [target]))
    network = NeuralNetwork([2, 8, 1], seed=seed)
    losses = network.train(samples, epochs=180, learning_rate=0.5, seed=seed)
    return network, losses


class Car:
    def __init__(self, offset=20.0, angle=0.15):
        self.x = RADIUS + offset
        self.y = 0.0
        self.heading = math.pi / 2 + angle
        self.steps = 0
        self.off_road = False

    def step(self, network):
        offset, angle = observations(self.x, self.y, self.heading)
        output = network.predict(inputs_for(offset, angle))[0]
        turn = (2 * output - 1) * MAX_TURN
        self.heading = wrap(self.heading + turn)
        self.x += SPEED * math.cos(self.heading)
        self.y += SPEED * math.sin(self.heading)
        self.steps += 1
        self.off_road = abs(observations(self.x, self.y, self.heading)[0]) > ROAD_WIDTH / 2
        return turn


def run_visual(network, losses):
    import tkinter as tk

    root = tk.Tk()
    root.title("Neural network lane follower")
    canvas = tk.Canvas(root, width=520, height=560, bg="#143b2a", highlightthickness=0)
    canvas.pack()
    center_x, center_y = 260, 260
    car = Car()
    paused = False

    def toggle(_event=None):
        nonlocal paused
        paused = not paused

    def reset(_event=None):
        nonlocal car
        car = Car()

    root.bind("<space>", toggle)
    root.bind("r", reset)
    root.bind("R", reset)

    def draw():
        if not paused and not car.off_road:
            car.step(network)
        canvas.delete("all")
        for radius, color in [(RADIUS + ROAD_WIDTH / 2, "#444b52"),
                              (RADIUS - ROAD_WIDTH / 2, "#143b2a")]:
            canvas.create_oval(center_x - radius, center_y - radius, center_x + radius,
                               center_y + radius, fill=color, outline="")
        canvas.create_oval(center_x - RADIUS, center_y - RADIUS, center_x + RADIUS,
                           center_y + RADIUS, outline="#f6cf60", dash=(8, 8), width=2)
        # The screen's y-axis points down, as does the simulated coordinate system.
        forward = (math.cos(car.heading), math.sin(car.heading))
        side = (-forward[1], forward[0])
        points = []
        for along, across in [(12, 0), (-9, 7), (-9, -7)]:
            points.extend([center_x + car.x + along * forward[0] + across * side[0],
                           center_y + car.y + along * forward[1] + across * side[1]])
        canvas.create_polygon(points, fill="#63d4ff", outline="white", width=2)
        offset, angle = observations(car.x, car.y, car.heading)
        status = "OFF ROAD — press R to restart" if car.off_road else ("PAUSED" if paused else "DRIVING")
        canvas.create_text(260, 488, text=f"{status}  |  Steps: {car.steps}  |  Offset: {offset:+.1f}",
                           fill="white", font=("Helvetica", 13, "bold"))
        canvas.create_text(260, 514, text=f"Heading error: {angle:+.2f} rad  |  Loss: {losses[0]:.4f} → {losses[-1]:.4f}",
                           fill="#c4d8cc", font=("Helvetica", 11))
        canvas.create_text(260, 540, text="Space: pause / resume     R: restart", fill="#c4d8cc")
        root.after(20, draw)

    draw()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run 1,000 simulation steps without a window")
    args = parser.parse_args()
    print("Training the neural network...", flush=True)
    network, losses = train_driver()
    print(f"Training loss: {losses[0]:.5f} -> {losses[-1]:.5f}", flush=True)
    if args.headless:
        car = Car()
        while car.steps < 1000 and not car.off_road:
            car.step(network)
        print(f"Steps: {car.steps}; off road: {car.off_road}; offset: {observations(car.x, car.y, car.heading)[0]:+.2f}")
    else:
        try:
            run_visual(network, losses)
        except ImportError as exc:
            raise SystemExit("The visual demo requires tkinter. Install Python with Tk support, or use --headless.") from exc


if __name__ == "__main__":
    main()
