import random
import unittest
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from point_sim import FlyExperiment
from serve_sim import make_request_handler


class FlyExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        roles = [
            {"root_id": "steer-left", "side": "left", "model_role": "steering"},
            {"root_id": "steer-right", "side": "right", "model_role": "steering"},
            {"root_id": "forward", "side": "left", "model_role": "forward"},
        ]
        edges = [
            {
                "post_id": "steer-left",
                "pre_side": "left",
                "synapse_count": 5,
            },
            {
                "post_id": "steer-right",
                "pre_side": "right",
                "synapse_count": 7,
            },
            {
                "post_id": "forward",
                "pre_side": "unknown",
                "synapse_count": 3,
            },
        ]
        circuit_data = {
            "roles": roles,
            "visual_edges": edges,
            "stats": {
                "aggregated_edges": len(edges),
                "total_synapses": 15,
                "presynaptic_neurons": 3,
                "metadata_coverage": 3,
                "source": "test.csv",
            },
        }
        self.experiment = FlyExperiment(
            image_size=(40, 8),
            mask_size=(10, 2),
            walkable_bits=bytes([0xFF, 0xFF, 0xFF]),
            circuit_data=circuit_data,
        )

    def test_select_point_rejects_out_of_bounds_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "통로"):
            self.experiment.select_point(40, 4)

    def test_point_selection_and_reset_update_public_state(self) -> None:
        self.experiment.select_point(2, 2)
        self.experiment.select_point(38, 6)

        state = self.experiment.public_state()
        self.assertEqual(state["start"], {"x": 2.0, "y": 2.0})
        self.assertEqual(state["destination"], {"x": 38.0, "y": 6.0})
        self.assertEqual(len(state["path"]), 1)

        self.experiment.reset_points()
        state = self.experiment.public_state()
        self.assertIsNone(state["start"])
        self.assertIsNone(state["destination"])
        self.assertEqual(state["path"], [])

    def test_python_learning_and_run_reach_goal_on_open_map(self) -> None:
        random.seed(42)
        self.experiment.select_point(2, 2)
        self.experiment.select_point(38, 2)

        self.experiment.train()
        self.assertEqual(self.experiment.training_episodes, 20)
        self.assertGreater(self.experiment.successful_training_episodes, 0)

        self.experiment.start_run()
        for _ in range(2500):
            if not self.experiment.is_running:
                break
            self.experiment.step()

        self.assertFalse(self.experiment.is_running)
        self.assertIn("실험 성공", self.experiment.message)
        self.assertTrue(self.experiment._is_at_goal(
            self.experiment.point, self.experiment.destination
        ))

    def test_http_api_accepts_point_and_returns_python_state(self) -> None:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_request_handler(self.experiment),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            request = Request(
                f"{url}/api/point",
                data=json.dumps({"x": 2, "y": 2}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                state = json.load(response)
            self.assertEqual(state["start"], {"x": 2.0, "y": 2.0})
            self.assertIsNone(state["destination"])

            with urlopen(f"{url}/api/state") as response:
                current_state = json.load(response)
            self.assertEqual(current_state["message"], state["message"])

            def post(
                path: str, payload: dict[str, object] | None = None
            ) -> dict[str, object]:
                request = Request(
                    f"{url}{path}",
                    data=json.dumps(payload or {}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request) as response:
                    return json.load(response)

            post("/api/point", {"x": 38, "y": 2})
            random.seed(42)
            trained = post("/api/train")
            self.assertGreater(trained["successful_training_episodes"], 0)
            self.assertTrue(post("/api/run")["is_running"])
            step_state = post("/api/step")
            self.assertNotIn("path", step_state)
            with urlopen(f"{url}/api/state") as response:
                current_state = json.load(response)
            self.assertEqual(len(current_state["path"]), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
