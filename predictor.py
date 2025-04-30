import abc
import multiprocessing as mp
import typing

import skyjo as sj
import skynet

PredictionId: typing.TypeAlias = int


class Predictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, state: sj.Skyjo) -> PredictionId:
        pass


class SkyNetPredictor(Predictor):
    def __init__(
        self, input_queue: mp.Queue, output_queue: mp.Queue, batch_size: int = 64
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        self.prediction_map = {}
        self.count = 0

    def send_to_prediction(self, state: sj.Skyjo) -> PredictionId:
        pred_id = self.count
        self.count += 1
        self.prediction_map[pred_id] = state
        spatial_numpy = sj.get_table(state)
        nonspatial_numpy = sj.get_game(state)
        self.input_queue.put((pred_id, spatial_numpy, nonspatial_numpy))
        return pred_id

    def recieve_ready_outputs(
        self,
    ) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        if self.output_queue.empty():
            return []

        outputs = []
        while not self.output_queue.empty():
            output = self.output_queue.get_nowait()
            outputs.append(output)
        return outputs
