"""Implementations of predictors that run model inference.

This abstracts the model execution and provides a consistent interface for
callers to request and receive model predictions. This includes a multi-processed
version that runs predictions in a dedicated process and communicates via
mp.Queue.
"""

from __future__ import annotations

import abc
import logging
import multiprocessing as mp
import time
import traceback
import typing

import numpy as np
import torch

import model_factory
import skyjo as sj
import skynet

# MARK: Types

PredictionId: typing.TypeAlias = int
PredictorInput: typing.TypeAlias = tuple[
    int,  # unique (per input queue) prediction id
    np.ndarray[tuple[int], np.uint8],  # spatial input
    np.ndarray[tuple[int], np.uint8],  # non-spatial input
]

PredictorOutput: typing.TypeAlias = tuple[
    int,  # unique (per input queue) prediction id
    np.ndarray[tuple[int], np.float32],  # value output
    np.ndarray[tuple[int], np.float32],  # points output
    np.ndarray[tuple[int], np.float32],  # policy output
]
QueueId: typing.TypeAlias = int
ModelUpdate: typing.TypeAlias = bool


# MARK: Predictor Process


class PredictorProcess(mp.Process):
    """Predictor process that runs model inference in a dedicated process."""

    def __init__(
        self,
        model_factory: model_factory.SkyNetModelFactory,
        model_update_queue: mp.Queue[ModelUpdate],
        input_queues: dict[QueueId, mp.Queue[PredictorInput]],
        output_queues: dict[QueueId, mp.Queue[PredictorOutput]],
        batch_size: int = 1024,
        device: torch.device = torch.device("cpu"),
        max_wait_seconds: float = 0.1,
        debug: bool = False,
    ):
        super().__init__()
        self.model_factory = model_factory
        self.model_update_queue = model_update_queue
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.batch_size = batch_size
        self.device = device
        self.max_wait_seconds = max_wait_seconds
        self.debug = debug

    def run(self):
        try:
            assert set(self.input_queues.keys()) == set(self.output_queues.keys()), (
                "Input and output queues must have the same id keys"
            )
            level = logging.DEBUG if self.debug else logging.INFO
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                filename="logs/multiprocessed_train/predictor.log",
                filemode="a",
            )
            model = self.model_factory.get_latest_model()
            model.set_device(self.device)
            model.eval()
            logging.info(
                f"Loaded model from {self.model_factory._get_latest_model_path()}"
            )
            start_time = time.time()
            input_data_queue = []
            while True:
                # Update model if available
                while not self.model_update_queue.empty():
                    model = self.model_factory.get_latest_model()
                    model.set_device(self.device)
                    model.eval()
                    logging.info(
                        f"New model loaded from {self.model_factory._get_latest_model_path()}"
                    )
                    self.model_update_queue.get()

                # Gather input from input queues
                for queue_id, input_queue in self.input_queues.items():
                    while not input_queue.empty():
                        pred_id, spatial_inputs, non_spatial_inputs = input_queue.get()
                        input_data_queue.append(
                            (queue_id, pred_id, spatial_inputs, non_spatial_inputs)
                        )

                if len(input_data_queue) >= self.batch_size or (
                    len(input_data_queue) > 0
                    and time.time() - start_time >= self.max_wait_seconds
                ):
                    if self.debug:
                        logging.info(
                            f"Running predictions, input_data_queue size: {len(input_data_queue)}"
                        )
                    to_run = input_data_queue[: self.batch_size]
                    input_data_queue = input_data_queue[self.batch_size :]
                    queue_ids, pred_ids, spatial_inputs, non_spatial_inputs = zip(
                        *to_run
                    )
                    spatial_input_tensor = torch.tensor(
                        np.array(spatial_inputs),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    non_spatial_input_tensor = torch.tensor(
                        np.array(non_spatial_inputs),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    # Model Inference
                    with torch.no_grad():
                        model_output = model(
                            spatial_input_tensor, non_spatial_input_tensor
                        )
                        numpy_output = skynet.output_to_numpy(model_output)

                    # Send outputs back to clients
                    for idx, (queue_id, pred_id) in enumerate(zip(queue_ids, pred_ids)):
                        value_numpy, points_numpy, policy_numpy = (
                            skynet.get_single_model_output(numpy_output, idx)
                        )
                        self.output_queues[queue_id].put(
                            (
                                pred_id,
                                value_numpy,
                                points_numpy,
                                policy_numpy,
                            )
                        )
                    start_time = time.time()
        except Exception as e:
            logging.error(f"Predictor process failed: {traceback.format_exc()}")
            raise e


# MARK: PredictorClient


class PredictorClient(abc.ABC):
    @abc.abstractmethod
    def put(self, state: sj.Skyjo) -> PredictionId:
        pass

    @abc.abstractmethod
    def get(self) -> tuple[PredictionId, skynet.SkyNetPrediction]:
        pass

    @abc.abstractmethod
    def get_all_nowait(
        self,
    ) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        pass


class MultiProcessPredictorClient(PredictorClient):
    """Predictor client that communicates with dedicated predictor process."""

    def __init__(
        self,
        input_queue: mp.Queue[PredictorInput],
        output_queue: mp.Queue[PredictorOutput],
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.count = 0

    def _get_unique_prediction_id(self) -> PredictionId:
        id = self.count
        self.count += 1
        return id

    def put(self, state: sj.Skyjo) -> PredictionId:
        prediction_id = self._get_unique_prediction_id()
        spatial_numpy = sj.get_spatial_input(state)
        nonspatial_numpy = sj.get_non_spatial_input(state)
        self.input_queue.put((prediction_id, spatial_numpy, nonspatial_numpy))
        return prediction_id

    def get_all_nowait(
        self,
    ) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        if self.output_queue.empty():
            return []

        outputs = []
        while not self.output_queue.empty():
            result = self.get_nowait()
            assert result is not None, (
                "result was None even though output queue isn't empty"
            )
            outputs.append(result)
        return outputs

    def get_all(self) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        if self.output_queue.empty():
            return []

        outputs = [self.get()]
        while not self.output_queue.empty():
            result = self.get_nowait()
            assert result is not None, (
                "result was None even though output queue isn't empty"
            )
            outputs.append(result)
        return outputs

    def get(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction]:
        pred_id, value_numpy, points_numpy, policy_numpy = self.output_queue.get()
        return pred_id, skynet.SkyNetPrediction(
            value_output=value_numpy,
            points_output=points_numpy,
            policy_output=policy_numpy,
        )

    def get_nowait(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction] | None:
        if self.output_queue.empty():
            return None
        pred_id, value_numpy, points_numpy, policy_numpy = (
            self.output_queue.get_nowait()
        )
        return pred_id, skynet.SkyNetPrediction(
            value_output=value_numpy,
            points_output=points_numpy,
            policy_output=policy_numpy,
        )


class NaivePredictorClient(PredictorClient):
    """Predictor client that actually just lazily runs the model inference."""

    def __init__(
        self,
        model: skynet.SkyNet,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.count = 0
        self.input_queue = []
        self.output_queue = []

    def _get_unique_prediction_id(self) -> PredictionId:
        id = self.count
        self.count += 1
        return id

    def _run_model(self) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        assert len(self.input_queue) > 0, "No predictions are available to run"
        prediction_ids, spatial_inputs, nonspatial_inputs = zip(*self.input_queue)
        spatial_input_tensor = torch.tensor(
            np.array(spatial_inputs),
            dtype=torch.float32,
            device=self.device,
        )
        non_spatial_input_tensor = torch.tensor(
            np.array(nonspatial_inputs),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            model_output = self.model(spatial_input_tensor, non_spatial_input_tensor)
            numpy_output = skynet.output_to_numpy(model_output)
        self.input_queue = []
        return [
            (
                prediction_id,
                skynet.SkyNetPrediction(
                    value_output=value_numpy,
                    points_output=points_numpy,
                    policy_output=policy_numpy,
                ),
            )
            for prediction_id, value_numpy, points_numpy, policy_numpy in zip(
                prediction_ids, *numpy_output
            )
        ]

    def put(self, state: sj.Skyjo) -> PredictionId:
        prediction_id = self._get_unique_prediction_id()
        spatial_numpy = sj.get_spatial_input(state)
        nonspatial_numpy = sj.get_non_spatial_input(state)
        self.input_queue.append((prediction_id, spatial_numpy, nonspatial_numpy))
        return prediction_id

    def get_all_nowait(
        self,
    ) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        if len(self.output_queue) == 0:
            if len(self.input_queue) == 0:
                return []
            self.output_queue = self._run_model()
        outputs = self.output_queue
        self.output_queue = []
        return outputs

    def get_all(
        self,
    ) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        return self.get_all_nowait()

    def get(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction]:
        if len(self.output_queue) == 0:
            self.output_queue = self._run_model()
        return self.output_queue.pop(0)

    def get_nowait(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction] | None:
        if len(self.output_queue) == 0:
            if len(self.input_queue) == 0:
                return None
            self.output_queue = self._run_model()
        return self.output_queue.pop(0)
