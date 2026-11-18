import math
import unittest

from lane_following import (Car, MAX_TURN, RADIUS, inputs_for, observations,
                            teacher, train_driver)


class LaneFollowingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network, cls.losses = train_driver()

    def test_observations_on_centerline(self):
        offset, angle = observations(RADIUS, 0, math.pi / 2)
        self.assertAlmostEqual(offset, 0)
        self.assertAlmostEqual(angle, 0)

    def test_network_approximates_teacher_on_unseen_states(self):
        self.assertLess(self.losses[-1], self.losses[0])
        for offset, angle in [(-23, -0.17), (5, 0.12), (21, -0.11)]:
            prediction = (2 * self.network.predict(inputs_for(offset, angle))[0] - 1) * MAX_TURN
            self.assertLess(abs(prediction - teacher(offset, angle)), 0.012)

    def test_driver_stays_on_road_from_multiple_starts(self):
        for offset, angle in [(-25, -0.2), (20, 0.15), (25, -0.2)]:
            with self.subTest(offset=offset, angle=angle):
                car = Car(offset, angle)
                for _ in range(1000):
                    car.step(self.network)
                    if car.off_road:
                        break
                self.assertFalse(car.off_road)
                self.assertLess(abs(observations(car.x, car.y, car.heading)[0]), 10)

    def test_off_road_is_detected(self):
        class StraightDriver:
            def predict(self, _inputs):
                return [0.5]

        car = Car(offset=39, angle=0)
        for _ in range(100):
            car.step(StraightDriver())
            if car.off_road:
                break
        self.assertTrue(car.off_road)


if __name__ == "__main__":
    unittest.main()
