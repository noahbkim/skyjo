"""Implementations of predictors that run model inference.

This abstracts the model execution and provides a consistent interface for
callers to request and receive model predictions. This includes a multi-processed
version that runs predictions in a dedicated process and communicates via
mp.Queue.
"""

from __future__ import annotations

import abc
import datetime
import logging
import pathlib
import time
import traceback
import typing

import numpy as np
import torch
import torch.multiprocessing as mp

import model_factory
import skyjo as sj
import skynet

# MARK: Types

PredictionId: typing.TypeAlias = int
QueueId: typing.TypeAlias = int
ModelUpdate: typing.TypeAlias = bool


# MARK: Queues


class PredictorInputQueue:
    def __init__(self, queue_id: QueueId, max_batch_size: int):
        self.queue_id = queue_id
        self.spatial_input_queue = mp.Queue()
        self.free_spatial_input_queue = mp.Queue()
        self.non_spatial_input_queue = mp.Queue()
        self.free_non_spatial_input_queue = mp.Queue()
        self.prediction_ids_queue = mp.Queue()
        self.free_prediction_ids_queue = mp.Queue()
        self.batch_size_queue = mp.Queue()
        self.max_batch_size = max_batch_size

    def empty(self) -> bool:
        return (
            self.spatial_input_queue.empty()
            and self.non_spatial_input_queue.empty()
            and self.prediction_ids_queue.empty()
            and self.batch_size_queue.empty()
        )

    def has_free(self) -> bool:
        return (
            not self.free_prediction_ids_queue.empty()
            or not self.free_spatial_input_queue.empty()
            or not self.free_non_spatial_input_queue.empty()
        )

    def get_free(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.free_prediction_ids_queue.get(),
            self.free_spatial_input_queue.get(),
            self.free_non_spatial_input_queue.get(),
        )

    def put_free(
        self,
        prediction_ids_tensor: torch.Tensor,
        spatial_input_tensor: torch.Tensor,
        non_spatial_input_tensor: torch.Tensor,
    ) -> None:
        self.free_spatial_input_queue.put(spatial_input_tensor)
        self.free_non_spatial_input_queue.put(non_spatial_input_tensor)
        self.free_prediction_ids_queue.put(prediction_ids_tensor)

    def put(
        self,
        prediction_ids_tensor: torch.Tensor,
        spatial_input_tensor: torch.Tensor,
        non_spatial_input_tensor: torch.Tensor,
        batch_size: int,
    ) -> None:
        assert batch_size <= self.max_batch_size, (
            f"Batch size: {batch_size} is greater than max batch size allowed: {self.max_batch_size}"
        )
        assert (
            prediction_ids_tensor.shape[0]
            == spatial_input_tensor.shape[0]
            == non_spatial_input_tensor.shape[0]
            == self.max_batch_size
        ), (
            "All input actual tensors must have the same batch size, got:",
            f"prediction id tensor: {prediction_ids_tensor.shape[0]}",
            f"spatial input tensor: {spatial_input_tensor.shape[0]}",
            f"non spatial input tensor: {non_spatial_input_tensor.shape[0]}",
            f"max batch size: {self.max_batch_size}",
        )
        self.spatial_input_queue.put(spatial_input_tensor)
        self.non_spatial_input_queue.put(non_spatial_input_tensor)
        self.prediction_ids_queue.put(prediction_ids_tensor)
        self.batch_size_queue.put(batch_size)

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        spatial_input_tensor = self.spatial_input_queue.get()
        non_spatial_input_tensor = self.non_spatial_input_queue.get()
        prediction_ids_tensor = self.prediction_ids_queue.get()
        batch_size = self.batch_size_queue.get()
        return (
            prediction_ids_tensor,
            spatial_input_tensor,
            non_spatial_input_tensor,
            batch_size,
        )


class PredictorOutputQueue:
    def __init__(self, queue_id: QueueId, max_batch_size: int):
        self.queue_id = queue_id
        self.prediction_ids_queue = mp.Queue()
        self.free_prediction_ids_queue = mp.Queue()
        self.value_output_queue = mp.Queue()
        self.free_value_output_queue = mp.Queue()
        self.points_output_queue = mp.Queue()
        self.free_points_output_queue = mp.Queue()
        self.policy_output_queue = mp.Queue()
        self.free_policy_output_queue = mp.Queue()
        self.batch_size_queue = mp.Queue()

    def empty(self) -> bool:
        return (
            self.prediction_ids_queue.empty()
            and self.value_output_queue.empty()
            and self.points_output_queue.empty()
            and self.policy_output_queue.empty()
        )

    def has_free(self) -> bool:
        return (
            not self.free_prediction_ids_queue.empty()
            or not self.free_spatial_input_queue.empty()
            or not self.free_non_spatial_input_queue.empty()
        )

    def get_free(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.free_prediction_ids_queue.get(),
            self.free_value_output_queue.get(),
            self.free_points_output_queue.get(),
            self.free_policy_output_queue.get(),
        )

    def put_free(
        self,
        prediction_ids_tensor: torch.Tensor,
        value_output_tensor: torch.Tensor,
        points_output_tensor: torch.Tensor,
        policy_output_tensor: torch.Tensor,
    ) -> None:
        self.free_prediction_ids_queue.put(prediction_ids_tensor)
        self.free_value_output_queue.put(value_output_tensor)
        self.free_points_output_queue.put(points_output_tensor)
        self.free_policy_output_queue.put(policy_output_tensor)

    def put(
        self,
        prediction_ids_tensor: torch.Tensor,
        value_output_tensor: torch.Tensor,
        points_output_tensor: torch.Tensor,
        policy_output_tensor: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.prediction_ids_queue.put(prediction_ids_tensor)
        self.value_output_queue.put(value_output_tensor)
        self.points_output_queue.put(points_output_tensor)
        self.policy_output_queue.put(policy_output_tensor)
        self.batch_size_queue.put(batch_size)

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        prediction_ids_tensor = self.prediction_ids_queue.get()
        value_output_tensor = self.value_output_queue.get()
        points_output_tensor = self.points_output_queue.get()
        policy_output_tensor = self.policy_output_queue.get()
        batch_size = self.batch_size_queue.get()
        return (
            prediction_ids_tensor,
            value_output_tensor,
            points_output_tensor,
            policy_output_tensor,
            batch_size,
        )


class UnifiedPredictorInputQueue:
    """Queue that holds all of the inputs to be processed by the predictor."""

    def __init__(self, device: torch.device):
        self._queue = []
        self.input_count = 0
        self.device = device

    def put(
        self,
        queue_id: QueueId,
        prediction_ids_tensor: torch.Tensor,
        spatial_input_tensor: torch.Tensor,
        non_spatial_input_tensor: torch.Tensor,
        batch_size: int,
    ):
        assert (
            len(prediction_ids_tensor)
            == len(spatial_input_tensor)
            == len(non_spatial_input_tensor)
            == batch_size
        ), (
            "All input tensors must have the same batch size",
            f"prediction_ids_tensor: {len(prediction_ids_tensor)}",
            f"spatial_input_tensor: {len(spatial_input_tensor)}",
            f"non_spatial_input_tensor: {len(non_spatial_input_tensor)}",
            f"batch_size: {batch_size}",
        )
        self._queue.append(
            (
                queue_id,
                prediction_ids_tensor,
                spatial_input_tensor,
                non_spatial_input_tensor,
                batch_size,
            )
        )
        self.input_count += batch_size

    def get(self) -> tuple[QueueId, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        (
            queue_id,
            prediction_ids_tensor,
            spatial_input_tensor,
            non_spatial_input_tensor,
            batch_size,
        ) = self._queue.pop(0)
        self.input_count -= batch_size
        return (
            queue_id,
            prediction_ids_tensor,
            spatial_input_tensor,
            non_spatial_input_tensor,
            batch_size,
        )

    def peek(self) -> tuple[QueueId, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self._queue[0]

    def get_batch(
        self,
        max_batch_size: int,
    ) -> tuple[
        list[tuple[QueueId, torch.Tensor, torch.Tensor, torch.Tensor, int]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Gets a batch of inputs from the queue in ready to run state.

        Returns the inputs in a stacked combined tensor ready for immediate model inference.
        Additionally, returns the original input tensors so that they can be put back into
        the free input queues once inference is complete.

        Args:
            max_batch_size: The maximum batch size to create

        Returns:
            A tuple of:
                - A list of tuples of the original input tensors.
                - A tensor of batched prediction ids.
                - A tensor of batched spatial inputs.
                - A tensor of batched non-spatial inputs.
                - A list of batch sizes for each input.
        """
        (
            queue_ids,
            spatial_tensors,
            non_spatial_tensors,
            prediction_id_tensors,
            batch_sizes,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )
        (
            queue_id,
            prediction_ids_tensor,
            spatial_input_tensor,
            non_spatial_input_tensor,
            batch_size,
        ) = self.peek()
        while sum(batch_sizes) + batch_size <= max_batch_size and self.input_count > 0:
            queue_ids.append(queue_id)
            spatial_tensors.append(spatial_input_tensor)
            non_spatial_tensors.append(non_spatial_input_tensor)
            prediction_id_tensors.append(prediction_ids_tensor)
            batch_sizes.append(batch_size)
            self.get()
            if self.input_count == 0:
                break
            (
                queue_id,
                prediction_ids_tensor,
                spatial_input_tensor,
                non_spatial_input_tensor,
                batch_size,
            ) = self._queue[0]
        spatial_input_tensor = torch.cat(spatial_tensors).to(self.device)
        non_spatial_input_tensor = torch.cat(non_spatial_tensors).to(self.device)
        prediction_id_tensor = torch.cat(prediction_id_tensors).to(self.device)
        return (
            queue_ids,
            prediction_id_tensor,
            spatial_input_tensor,
            non_spatial_input_tensor,
            batch_sizes,
        )


# MARK: Predictor Process


class PredictorProcess(mp.Process):
    """Predictor process that runs model inference in a dedicated process."""

    def __init__(
        self,
        factory: model_factory.SkyNetModelFactory,
        model_update_queue: mp.Queue[ModelUpdate],
        input_queues: dict[QueueId, PredictorInputQueue],
        output_queues: dict[QueueId, PredictorOutputQueue],
        min_batch_size: int = 512,
        max_batch_size: int = 1024,
        device: torch.device = torch.device("cpu"),
        max_wait_seconds: float = 0.1,
        debug: bool = False,
        free_output_queue_free: int = 10,
    ):
        super().__init__()
        self.model_factory = factory
        self.model_update_queue = model_update_queue
        self.input_queues = input_queues
        self.output_queues = output_queues
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.device = device
        self.max_wait_seconds = max_wait_seconds
        self.debug = debug
        self.free_output_queue_free = free_output_queue_free

        assert set(self.input_queues.keys()) == set(self.output_queues.keys()), (
            "Input and output queues must have the same id keys"
        )

    def _populate_free_input_queues(self, model: skynet.SkyNet):
        """Populate the input queues with free shared tensors for use when returning results."""
        for queue_id, input_queue in self.input_queues.items():
            for _ in range(2):
                input_queue.put_free(
                    torch.zeros(
                        size=(input_queue.max_batch_size,),
                        dtype=torch.int64,
                    ).share_memory_(),  # prediction ids
                    torch.zeros(
                        size=(
                            input_queue.max_batch_size,
                            *model.spatial_input_shape,
                        ),
                        dtype=torch.float32,
                    ).share_memory_(),  # spatial input
                    torch.zeros(
                        size=(
                            input_queue.max_batch_size,
                            *model.non_spatial_input_shape,
                        ),
                        dtype=torch.float32,
                    ).share_memory_(),  # non spatial input
                )

    def _populate_free_output_queues(self, model: skynet.SkyNet):
        """Populate the output queues with free shared tensors for use when returning results."""
        for queue_id, output_queue in self.output_queues.items():
            for _ in range(2):
                output_queue.put_free(
                    torch.zeros(
                        size=(self.input_queues[queue_id].max_batch_size,),
                        dtype=torch.int64,
                    ).share_memory_(),  # prediction ids
                    torch.zeros(
                        size=(
                            self.input_queues[queue_id].max_batch_size,
                            *model.value_output_shape,
                        ),
                        dtype=torch.float32,
                    ).share_memory_(),  # value output
                    torch.zeros(
                        size=(
                            self.input_queues[queue_id].max_batch_size,
                            *model.points_output_shape,
                        ),
                        dtype=torch.float32,
                    ).share_memory_(),  # points output
                    torch.zeros(
                        size=(
                            self.input_queues[queue_id].max_batch_size,
                            *model.policy_output_shape,
                        ),
                        dtype=torch.float32,
                    ).share_memory_(),  # policy output
                )

    def _setup_logging(self):
        """Setup logging for the predictor process."""
        log_dir = pathlib.Path(
            f"logs/multiprocessed_train/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_dir / "predictor.log",
            filemode="w",
        )

    def _load_latest_model(self) -> skynet.SkyNet:
        """Load the latest model from the model factory."""
        model = self.model_factory.get_latest_model()
        model.set_device(self.device)
        model.eval()
        logging.info(f"Loaded model from {self.model_factory._get_latest_model_path()}")
        return model

    def _gather_available_inputs(self, unified_input_queue: UnifiedPredictorInputQueue):
        """Gather input from input queues and put it into the unified input queue."""
        for queue_id, input_queue in self.input_queues.items():
            while not input_queue.empty():
                (
                    prediction_ids_tensor,
                    spatial_input_tensor,
                    non_spatial_input_tensor,
                    batch_size,
                ) = input_queue.get()

                unified_input_queue.put(
                    queue_id,
                    prediction_ids_tensor[:batch_size].clone(),
                    spatial_input_tensor[:batch_size].clone(),
                    non_spatial_input_tensor[:batch_size].clone(),
                    batch_size,
                )
                # logging.info(
                #     f"Received input from queue {queue_id}, batch_size: {batch_size}, first prediction_id: {prediction_ids_tensor[0]}, last prediction_id: {prediction_ids_tensor[batch_size - 1]}"
                # )
                input_queue.put_free(
                    prediction_ids_tensor,
                    spatial_input_tensor,
                    non_spatial_input_tensor,
                )

    def run(self):
        try:
            self._setup_logging()

            # Load model
            model = self._load_latest_model()

            self._populate_free_input_queues(model)
            self._populate_free_output_queues(model)
            start_time = time.time()
            unified_input_queue = UnifiedPredictorInputQueue(device=self.device)

            model_run_count = 0
            processed_count_average = 0
            while True:
                # Update model if available
                while not self.model_update_queue.empty():
                    self.model_update_queue.get()
                    model = self._load_latest_model()

                self._gather_available_inputs(unified_input_queue)

                if unified_input_queue.input_count >= self.min_batch_size or (
                    unified_input_queue.input_count > 0
                    and time.time() - start_time >= self.max_wait_seconds
                ):
                    if self.debug:
                        logging.info(
                            f"Running predictions, input_data_queue size: {unified_input_queue.input_count}"
                        )

                    (
                        queue_ids,
                        prediction_ids,
                        spatial_input_tensor,
                        non_spatial_input_tensor,
                        batch_sizes,
                    ) = unified_input_queue.get_batch(self.max_batch_size)

                    # Model Inference
                    with torch.no_grad():
                        value_output, points_output, policy_output = model(
                            spatial_input_tensor, non_spatial_input_tensor
                        )

                    # Send outputs back to clients
                    processed_count = 0
                    for queue_id, batch_size in zip(queue_ids, batch_sizes):
                        # logging.info(
                        #     f"Processing batch batch_size: {batch_size}, first prediction_id: {prediction_ids[processed_count]}, last prediction_id: {prediction_ids[processed_count + batch_size - 1]}"
                        # )
                        # Get free output tensors and set to results
                        (
                            prediction_ids_tensor,
                            value_tensor,
                            points_tensor,
                            policy_tensor,
                        ) = self.output_queues[queue_id].get_free()
                        prediction_ids_tensor[:batch_size] = prediction_ids[
                            processed_count : processed_count + batch_size
                        ]
                        value_tensor[:batch_size] = value_output[
                            processed_count : processed_count + batch_size
                        ]
                        points_tensor[:batch_size] = points_output[
                            processed_count : processed_count + batch_size
                        ]
                        policy_tensor[:batch_size] = policy_output[
                            processed_count : processed_count + batch_size
                        ]

                        # Put results back into output queue
                        self.output_queues[queue_id].put(
                            prediction_ids_tensor,
                            value_tensor,
                            points_tensor,
                            policy_tensor,
                            batch_size,
                        )
                        processed_count += batch_size

                    processed_count_average = (
                        processed_count_average * model_run_count + processed_count
                    ) / (model_run_count + 1)
                    model_run_count += 1

                    if model_run_count % 1000 == 0:
                        logging.info(
                            f"Processed {processed_count_average} samples per model run"
                        )
                    start_time = time.time()
        except Exception as e:
            logging.error(f"Predictor process failed: {traceback.format_exc()}")
            raise e


# MARK: PredictorClient


class PredictorClient(abc.ABC):
    @abc.abstractmethod
    def put(self, state: sj.Skyjo) -> PredictionId:
        """Puts a state into the client's input queue."""
        pass

    @abc.abstractmethod
    def send(self) -> None:
        """Forces the client to send requests to the predictor process."""
        pass

    @abc.abstractmethod
    def get(self) -> tuple[PredictionId, skynet.SkyNetPrediction]:
        """Gets a prediction from the client's output queue."""
        pass

    @abc.abstractmethod
    def get_nowait(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction] | None:
        """Gets a prediction from the client's output queue without waiting."""
        pass

    @abc.abstractmethod
    def get_all(
        self,
    ) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        """Gets all predictions from the client's output queue."""
        pass

    @abc.abstractmethod
    def get_all_nowait(
        self,
    ) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        """Gets all predictions from the client's output queue without waiting."""
        pass


class MultiProcessPredictorClient(PredictorClient):
    """Predictor client that communicates with dedicated predictor process."""

    def __init__(
        self,
        input_queue: PredictorInputQueue,
        output_queue: PredictorOutputQueue,
    ):
        self.current_inputs = []
        self.current_output = None
        self.sample_count = 0
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.count = 0

    @property
    def output_ready(self) -> bool:
        return self.current_output is not None or not self.output_queue.empty()

    def _get_unique_prediction_id(self) -> PredictionId:
        id = self.count
        self.count += 1
        return id

    def _get_new_output(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            prediction_ids,
            value_output,
            points_output,
            policy_output,
            batch_size,
        ) = self.output_queue.get()
        new_output = (
            prediction_ids[:batch_size].clone(),
            value_output[:batch_size].clone(),
            points_output[:batch_size].clone(),
            policy_output[:batch_size].clone(),
        )
        self.output_queue.put_free(
            prediction_ids,
            value_output,
            points_output,
            policy_output,
        )
        # logging.info(
        #     f"Recieved new output, batch_size: {batch_size}, first prediction_id: {new_output[0][0]}, last prediction_id: {new_output[0][-1]}"
        # )
        return new_output

    def _get_current_batch(
        self,
    ) -> tuple[
        list[PredictionId],
        list[np.ndarray[tuple[int], np.float32]],
        list[np.ndarray[tuple[int], np.float32]],
        int,
    ]:
        batch_size = min(len(self.current_inputs), self.input_queue.max_batch_size)
        assert batch_size > 0, "Batch size must be greater than 0"
        batch = self.current_inputs[:batch_size]
        self.current_inputs = self.current_inputs[batch_size:]
        prediction_ids, spatial_inputs, nonspatial_inputs = zip(*batch)
        return (
            prediction_ids,
            spatial_inputs,
            nonspatial_inputs,
            batch_size,
        )

    def put(self, state: sj.Skyjo) -> PredictionId:
        prediction_id = self._get_unique_prediction_id()
        spatial_numpy = skynet.get_spatial_state_numpy(state)
        nonspatial_numpy = skynet.get_non_spatial_state_numpy(state)
        self.current_inputs.append((prediction_id, spatial_numpy, nonspatial_numpy))
        return prediction_id

    def send(self) -> None:
        prediction_ids, spatial_inputs, nonspatial_inputs, batch_size = (
            self._get_current_batch()
        )
        prediction_ids_tensor, spatial_input_tensor, nonspatial_input_tensor = (
            self.input_queue.get_free()
        )
        prediction_ids_tensor[:batch_size] = torch.tensor(
            np.array(prediction_ids),
            dtype=torch.int64,
        )
        spatial_input_tensor[:batch_size] = torch.tensor(
            np.array(spatial_inputs),
            dtype=torch.float32,
        )
        nonspatial_input_tensor[:batch_size] = torch.tensor(
            np.array(nonspatial_inputs),
            dtype=torch.float32,
        )
        self.input_queue.put(
            prediction_ids_tensor,
            spatial_input_tensor,
            nonspatial_input_tensor,
            batch_size,
        )
        # logging.info(
        #     f"sent batch, batch_size: {batch_size}, first prediction_id: {prediction_ids[0]}, last prediction_id: {prediction_ids[-1]}"
        # )

    def get(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction] | None:
        if self.current_output is None:
            self.current_output = self._get_new_output()
        to_return = (
            self.current_output[0][self.sample_count].item(),
            skynet.SkyNetPrediction(
                value_output=self.current_output[1][self.sample_count],
                points_output=self.current_output[2][self.sample_count],
                policy_output=self.current_output[3][self.sample_count],
            ),
        )
        self.sample_count += 1
        # We have returned all of the samples in the current batch
        if self.sample_count == self.current_output[0].shape[0]:
            self.current_output = None
            self.sample_count = 0
        return to_return

    def get_all(self) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        if self.current_output is None:
            self.current_output = self._get_new_output()
        outputs = []
        for sample_idx in range(self.sample_count, self.current_output[0].shape[0]):
            outputs.append(
                (
                    self.current_output[0][sample_idx].item(),
                    skynet.SkyNetPrediction(
                        value_output=self.current_output[1][sample_idx],
                        points_output=self.current_output[2][sample_idx],
                        policy_output=self.current_output[3][sample_idx],
                    ),
                )
            )
        self.current_output = None
        self.sample_count = 0
        return outputs

    def get_nowait(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction] | None:
        if self.current_output is None:
            return None
        return self.get()

    def get_all_nowait(self) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        if self.current_output is None:
            return []
        return self.get_all()


class NaivePredictorClient(PredictorClient):
    """Predictor client that actually just lazily runs the model inference."""

    def __init__(
        self,
        model: skynet.SkyNet,
        max_batch_size: int,
    ):
        self.model = model
        self.device = model.device
        self.count = 0
        self.input_queue = []
        self.output_queue = []
        self.current_inputs = []
        self.current_output = None
        self.sample_count = 0
        self.max_batch_size = max_batch_size

    @property
    def output_ready(self) -> bool:
        return self.current_output is not None or not self.output_queue.empty()

    def _get_unique_prediction_id(self) -> PredictionId:
        id = self.count
        self.count += 1
        return id

    def _get_new_output(
        self,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction_ids, value_output, points_output, policy_output = (
            self.output_queue.pop(0)
        )
        return (
            prediction_ids,
            value_output,
            points_output,
            policy_output,
        )

    def _get_current_batch(
        self,
    ) -> tuple[
        list[PredictionId],
        list[np.ndarray[tuple[int], np.float32]],
        list[np.ndarray[tuple[int], np.float32]],
        int,
    ]:
        batch_size = min(len(self.current_inputs), self.max_batch_size)
        assert batch_size > 0, "Batch size must be greater than 0"
        batch = self.current_inputs[:batch_size]
        self.current_inputs = self.current_inputs[batch_size:]
        prediction_ids, spatial_inputs, nonspatial_inputs = zip(*batch)
        return (
            prediction_ids,
            spatial_inputs,
            nonspatial_inputs,
            batch_size,
        )

    def put(self, state: sj.Skyjo) -> PredictionId:
        prediction_id = self._get_unique_prediction_id()
        spatial_numpy = skynet.get_spatial_state_numpy(state)
        nonspatial_numpy = skynet.get_non_spatial_state_numpy(state)
        self.current_inputs.append((prediction_id, spatial_numpy, nonspatial_numpy))
        return prediction_id

    def send(self) -> None:
        prediction_ids, spatial_inputs, nonspatial_inputs, batch_size = (
            self._get_current_batch()
        )

        spatial_input_tensor = torch.tensor(
            np.array(spatial_inputs),
            device=self.device,
            dtype=torch.float32,
        )
        nonspatial_input_tensor = torch.tensor(
            np.array(nonspatial_inputs),
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            value_output, points_output, policy_output = self.model(
                spatial_input_tensor, nonspatial_input_tensor
            )
            if self.device != torch.device("cpu"):
                value_output = value_output.cpu()
                points_output = points_output.cpu()
                policy_output = policy_output.cpu()
        self.output_queue.append(
            (
                prediction_ids[:batch_size],
                value_output,
                points_output,
                policy_output,
            )
        )

    def get(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction] | None:
        if self.current_output is None:
            self.current_output = self._get_new_output()
        to_return = (
            self.current_output[0][self.sample_count],
            skynet.SkyNetPrediction(
                value_output=self.current_output[1][self.sample_count],
                points_output=self.current_output[2][self.sample_count],
                policy_output=self.current_output[3][self.sample_count],
            ),
        )
        self.sample_count += 1
        # We have returned all of the samples in the current batch
        if self.sample_count == self.current_output[1].shape[0]:
            self.current_output = None
            self.sample_count = 0
        return to_return

    def get_all(self) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        if self.current_output is None:
            self.current_output = self._get_new_output()
        outputs = []
        for sample_idx in range(self.sample_count, self.current_output[1].shape[0]):
            outputs.append(
                (
                    self.current_output[0][sample_idx],
                    skynet.SkyNetPrediction(
                        value_output=self.current_output[1][sample_idx],
                        points_output=self.current_output[2][sample_idx],
                        policy_output=self.current_output[3][sample_idx],
                    ),
                )
            )
        self.current_output = None
        self.sample_count = 0
        return outputs

    def get_nowait(
        self,
    ) -> tuple[PredictionId, skynet.SkyNetPrediction] | None:
        if self.current_output is None:
            return None
        return self.get()

    def get_all_nowait(self) -> list[tuple[PredictionId, skynet.SkyNetPrediction]]:
        if self.current_output is None:
            return []
        return self.get_all()


if __name__ == "__main__":
    import time

    factory = model_factory.SkyNetModelFactory()
    input_queues = {0: PredictorInputQueue(0, 512)}
    output_queues = {0: PredictorOutputQueue(0, 512)}
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    debug = True
    predictor_process = PredictorProcess(
        factory=factory,
        model_update_queue=mp.Queue(),
        device=device,
        input_queues=input_queues,
        output_queues=output_queues,
        debug=debug,
    )
    predictor_process.start()
    predictor_client = MultiProcessPredictorClient(
        input_queue=input_queues[0],
        output_queue=output_queues[0],
    )
    state = sj.new(players=2)
    state = sj.start_round(state)

    predictor_client.put(state)
    predictor_client.put(sj.apply_action(state, sj.MASK_FLIP_SECOND_BELOW))
    time.sleep(1)
    assert not predictor_client.output_ready, (
        "Nothing has been sent yet, so output should not be ready"
    )
    print("Sending")
    predictor_client.send()
    time.sleep(2)
    assert predictor_client.output_ready, "Output should be ready after sending"
    print("Getting")
    prediction = predictor_client.get()
    print(prediction)
    assert prediction is not None, "Prediction should not be None"

    predictions = predictor_client.get_all()
    assert len(predictions) == 1, "Should have 1 remaining prediction"
    print(predictions[0])

    assert predictor_client.current_output is None, "Output should be empty"

    predictor_process.join()
