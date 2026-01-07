"""
Stream model state with optional persistence for memory-efficient simulation workflows.
"""

import logging
from pathlib import Path
import typing
from os import PathLike

import psutil
from typing_extensions import Self

from bores.errors import StreamError
from bores.states import ModelState, PickleStore, StateStore
from bores.types import NDimension


__all__ = ["StateStream", "StreamProgress"]

logger = logging.getLogger(__name__)


class StreamProgress(typing.TypedDict):
    """Progress statistics for state streaming."""

    num_yielded: int
    num_saved: int
    num_checkpoints: int
    batch_pending: int
    store_backend: typing.Optional[str]
    memory_mb: float


class StateStream(typing.Generic[NDimension]):
    """
    Memory-efficient stream for model state iteration with optional persistence.

    Wraps a state generator/iterator and optionally persists states to disk as they're yielded,
    immediately freeing memory. Supports batching for I/O efficiency.

    Usage Benefits:
        - Low memory overhead (states persisted immediately after yield)
        - Batch persistence for I/O efficiency
        - Optional validation before save
        - Progress tracking and logging
        - Auto-save on context exit (no lost data)
        - Replay from store (load previously saved states)
        - Selective persistence (save only states matching predicate)
        - Checkpointing for crash recovery
        - Memory monitoring with automatic flushing

    Important: The underlying iterable (typically a generator) can only be consumed once.
    After the first iteration, either:
        - Use `replay()` to load states from the store
        - Create a new stream with a fresh iterable
        - Let `__iter__` automatically use `replay()` if a store exists

    Example:
    ```python
    store = new_store(filepath="simulation.zarr", backend="zarr")
    stream = StateStream(
        states=run_simulation(...),
        store=store,
        batch_size=10,
    )
    with stream:
        for state in stream:
            process_state(state)  # State saved to disk after processing
        ...

    # Later: replay from disk (automatically or explicitly)
    for state in stream:  # Auto-replays if store exists
        analyze_state(state)

    # Or explicitly:
    for state in stream.replay():
        analyze_state(state)
    ```
    """

    def __init__(
        self,
        states: typing.Union[
            typing.Generator[ModelState[NDimension]],
            typing.Iterator[ModelState[NDimension]],
        ],
        store: typing.Optional[StateStore] = None,
        batch_size: int = 10,
        validate: bool = True,
        auto_save: bool = True,
        auto_replay: bool = False,
        lazy_load: bool = True,
        save_predicate: typing.Optional[
            typing.Callable[[ModelState[NDimension]], bool]
        ] = None,
        checkpoint_interval: typing.Optional[int] = None,
        checkpoint_dir: typing.Optional[PathLike] = None,
        max_memory_mb: typing.Optional[float] = None,
    ) -> None:
        """
        Initialize state stream.

        :param states: Generator or iterator of `ModelState` instances
        :param store: Optional `StateStore` for persistence. If None, states only yielded (no persistence)
        :param batch_size: Number of states to accumulate before flushing to disk (default: 10)
        :param validate: Validate states before persisting (default: True)
        :param auto_save: Automatically flush remaining states on context exit (default: True)
        :param auto_replay: If True, automatically replay from store when iterating after consumption.
            If False, raises `StreamError` instead (default: False, explicit replay required)
        :param lazy_load: When replaying from store, use lazy loading if supported (default: True)
        :param save_predicate: Optional function to filter which states to save.
            If provided, only states where save_predicate(state) returns True are saved.
            Example: lambda s: s.step % 10 == 0 (save every 10th state)
        :param checkpoint_interval: Optional interval for checkpointing. If provided,
            creates a checkpoint every N states for crash recovery. Example: 100
        :param checkpoint_dir: Directory to save checkpoints. Required if `checkpoint_interval` is set.
        :param max_memory_mb: Optional soft memory limit in MB. If current process memory
            exceeds this limit, batch is flushed immediately. Note: This is checked only
            when deciding whether to flush, so actual memory usage may exceed this limit
            between checks. Example: 1000.0
        """
        self.states = states
        self.store = store
        self.batch_size = batch_size
        self.validate = validate
        self.auto_save = auto_save
        self.auto_replay = auto_replay
        self.lazy_load = lazy_load
        self.save_predicate = save_predicate
        self.checkpoint_interval = checkpoint_interval
        self.max_memory_mb = max_memory_mb

        # Warn if store-dependent features are configured without a store
        if self.store is None:
            if self.validate:
                logger.warning(
                    "Validation is enabled but no store provided. States will be validated "
                    "but not persisted."
                )
            if self.auto_save:
                logger.debug(
                    "auto_save=True but no store provided. This setting has no effect."
                )
            if self.save_predicate is not None:
                logger.warning(
                    "save_predicate provided but no store configured. Predicate will be ignored."
                )
            if self.max_memory_mb is not None:
                logger.warning(
                    "max_memory_mb provided but no store configured. Memory-based flushing "
                    "will not occur without persistence."
                )

        self._batch: typing.List[ModelState[NDimension]] = []
        self._num_yielded: int = 0
        self._num_saved: int = 0
        self._num_checkpoints: int = 0
        self._started: bool = False
        self._consumed: bool = False
        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )

        # Validate checkpoint configuration
        if self.checkpoint_interval is not None:
            if self._checkpoint_dir is None:
                raise StreamError(
                    "`checkpoint_dir` must be provided when `checkpoint_interval` is set"
                )
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> typing.Iterator[ModelState[NDimension]]:
        """
        Iterate over states, optionally persisting as we go.

        Yields states one at a time, accumulating in a batch buffer.
        When batch is full, flush to store and clear buffer.

        Memory pattern:
            1. Yield state to user → User processes it
            2. Add to batch buffer (small memory cost)
            3. When batch full → Flush to disk, clear buffer
            4. Net effect: Only `batch_size` states stay in memory at once

        Note: If the underlying iterable is a generator, it can only be consumed once.
        After the first iteration:
            - If `auto_replay=True` and a store exists, automatically replays from store
            - If `auto_replay=False`, raises `StreamError` (use `replay()` explicitly)
            - If no store exists, raises `StreamError` (create fresh stream)

        :return: Iterator over `ModelState` instances
        :raises `StreamError`: If trying to iterate again after exhaustion (when auto_replay=False or no store)
        """
        # Check if we've already consumed the iterable
        if self._consumed:
            if self.auto_replay and self.store is not None:
                logger.debug(
                    "Stream already consumed. Auto-replaying from store. "
                    "Set auto_replay=False to disable this behavior."
                )
                yield from self.replay()
                return
            elif self.store is not None:
                raise StreamError(
                    "Cannot iterate again: the underlying iterable has been exhausted. "
                    "Use `replay()` to load from store or set auto_replay=True."
                )
            else:
                raise StreamError(
                    "Cannot iterate again: the underlying iterable has been exhausted. "
                    "Either provide a fresh iterable or use a store with replay capability."
                )

        if self.store is None:
            logger.info("No store provided, streaming without persistence")
            for state in self.states:
                self._num_yielded += 1
                yield state
            self._consumed = True
            return

        logger.debug(f"Streaming to {self.store} with batch_size={self.batch_size}")

        for state in self.states:
            self._num_yielded += 1

            yield state

            if self._should_save(state=state):
                self._batch.append(state)

                if self._should_flush():
                    self.flush()

                if self._should_checkpoint(state=state):
                    self._save_checkpoint(state=state)

        if self._batch and self.auto_save:
            logger.debug(f"Flushing final batch of {len(self._batch)} states")
            self.flush()

        # Mark the iterable as consumed
        self._consumed = True

        logger.debug(
            f"Completed stream: {self._num_yielded} yielded, {self._num_saved} saved"
        )

    def __enter__(self) -> Self:
        """
        Context manager entry. Prepare for streaming.

        :return: Self for context manager usage
        """
        self._started = True
        if self.store is not None:
            logger.info(f"Started stream session to {self.store}")
        else:
            logger.info("Started stream session (no persistence)")
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_val: typing.Optional[BaseException],
        exc_tb: typing.Optional[typing.Any],
    ) -> None:
        """
        Context manager exit - flush any remaining states.

        Ensures all states saved even if iteration interrupted.

        :param exc_type: Exception type if error occurred
        :param exc_val: Exception value if error occurred
        :param exc_tb: Exception traceback if error occurred
        """
        if self._batch and self.auto_save and self.store is not None:
            logger.warning(f"Flushing {len(self._batch)} unsaved states on exit")
            try:
                self.flush()
            except Exception as exc:
                logger.error(f"Failed to flush states on exit: {exc}")

        if exc_type is None:
            logger.info(
                f"Stream complete: {self._num_saved} states saved, "
                f"{self._num_checkpoints} checkpoints created."
            )
        else:
            logger.error(f"Stream interrupted at {self._num_saved} states: {exc_val}")

    def collect(self, *steps: int) -> typing.List[ModelState[NDimension]]:
        """
        Collect states from the stream into a list.

        Note: This loads all states (or filtered states) into memory at once, which defeats
        the memory-efficient purpose of streaming. Use sparingly for small datasets or when
        you specifically need all states in memory for processing.

        :param steps: Optional step numbers to filter. If provided, only states with
            matching step numbers are collected. Example: collect(0, 10, 20)
        :return: List of collected ModelState instances

        Example:
        ```python
        # Collect all states (memory-intensive!)
        all_states = stream.collect()

        # Collect specific steps only
        initial_and_final = stream.collect(0, 100, 200)

        # Better alternative for large datasets - iterate instead:
        for state in stream:
            if state.step in {0, 100, 200}:
                process(state)
        ```
        """
        if steps:
            logger.debug(f"Collecting states from stream (filtering steps: {steps})")
            # Convert to set for O(1) lookup and track remaining steps
            remaining_steps = set(steps)
        else:
            logger.warning(
                "Collecting entire stream into memory. This may consume significant memory "
                "for large simulations. Consider iterating instead if memory is a concern."
            )
            remaining_steps = None

        states = []
        for state in self:
            if remaining_steps is not None:
                if state.step in remaining_steps:
                    states.append(state)
                    remaining_steps.remove(state.step)
                    # Early exit if we've collected all requested steps
                    if not remaining_steps:
                        logger.debug(
                            f"Collected all {len(steps)} requested steps, stopping early"
                        )
                        break
            else:
                states.append(state)

        logger.debug(f"Collected {len(states)} states into memory")
        return states

    def consume(self) -> None:
        """
        Consume the entire stream without yielding states to the caller.

        This method iterates through all states, triggering any configured side effects
        (persistence, checkpointing, validation) without returning states. Useful when
        you only want the side effects (saving to store, creating checkpoints) without
        processing individual states.

        The stream's internal mechanisms (__iter__, batching, flushing, checkpointing)
        still occur normally - only the yielding to caller is skipped.

        Example:
        ```python
        # Just save all states to disk without processing them
        stream = StateStream(states=simulation(), store=store)
        stream.consume()  # States saved, nothing returned

        # Create checkpoints without holding states in memory
        stream = StateStream(
            states=simulation(),
            checkpoint_interval=100,
            checkpoint_dir=Path("checkpoints")
        )
        stream.consume()  # Checkpoints created, stream exhausted
        ```

        Note: After calling consume(), the stream is exhausted. Use replay() or set
        auto_replay=True to iterate again.
        """
        logger.debug("Consuming stream (no yield to caller)")
        for _ in self:
            pass  # Iterate through, triggering side effects but not yielding
        logger.debug(f"Stream consumed: {self._num_yielded} states processed")

    def replay(self) -> typing.Iterator[ModelState[NDimension]]:
        """
        Load and iterate over previously saved states from store.

        Useful for:
            - Post-processing saved results
            - Resuming from checkpoint
            - Debugging specific timesteps

        Note: Replaying continues to increment `num_yielded` counter. If you replay
        100 states after initially streaming 100 states, `num_yielded` will be 200.
        This tracks total states yielded across all operations.

        :return: Iterator over loaded ModelState instances
        :raises `StreamError`: If no store provided
        :raises `StorageError`: If store file doesn't exist or is corrupted
        """
        if self.store is None:
            raise StreamError("Cannot replay: no store provided")

        logger.debug(f"Replaying states from {self.store}")

        for state in self.store.load(lazy=self.lazy_load, validate=self.validate):
            self._num_yielded += 1
            yield state

        logger.debug(f"Replay complete: {self._num_yielded} states loaded")

    def flush(self) -> None:
        """
        Manually flush accumulated batch to store.

        Writes all states in the current batch to the store and clears the buffer.
        Logs progress and handles any validation errors.

        :raises `StreamError`: If no store provided
        """
        if self.store is None:
            raise StreamError("Cannot flush: no store provided")

        if not self._batch:
            logger.debug("Flush called but batch is empty")
            return

        batch_size = len(self._batch)
        logger.debug(f"Flushing batch of {batch_size} states to {self.store}")

        try:
            self.store.dump(
                states=self._batch,
                exist_ok=True,
                validate=self.validate,
            )
            self._num_saved += batch_size
            logger.debug(
                f"Flushed {batch_size} states (total saved: {self._num_saved})"
            )
        except Exception as exc:
            logger.error(f"Failed to flush batch: {exc}")
            raise
        finally:
            self._batch.clear()

    def get_pending_batch(self) -> typing.List[ModelState[NDimension]]:
        """
        Get a copy of states in the current batch (not yet flushed to store).

        Useful for:
            - Inspecting what will be saved on next flush
            - Recovering states if an error occurs before flush
            - Debugging batch accumulation behavior

        :return: Copy of the current batch buffer (safe to modify without affecting stream)
        """
        return self._batch.copy()

    def _should_save(self, state: ModelState[NDimension]) -> bool:
        """
        Determine if state should be saved based on save_predicate.

        :param state: State to evaluate
        :return: True if state should be saved, False otherwise
        """
        if self.save_predicate is None:
            return True
        return self.save_predicate(state)

    def _should_flush(self) -> bool:
        """
        Determine if batch should be flushed based on batch size and memory limits.

        Flushes if:
            1. Batch size reached, OR
            2. Memory limit exceeded (if configured)

        :return: True if batch should be flushed, False otherwise
        """
        if len(self._batch) >= self.batch_size:
            return True

        if self.max_memory_mb is not None:
            current_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if current_memory_mb > self.max_memory_mb:
                logger.warning(
                    f"Memory limit reached ({current_memory_mb:.1f} MB > "
                    f"{self.max_memory_mb:.1f} MB), flushing batch early"
                )
                return True     
        return False

    def _should_checkpoint(self, state: ModelState[NDimension]) -> bool:
        """
        Determine if a checkpoint should be created for this state.

        :param state: Current state
        :return: True if checkpoint should be created, False otherwise
        """
        if self.checkpoint_interval is None:
            return False
        return state.step > 0 and state.step % self.checkpoint_interval == 0

    def _save_checkpoint(self, state: ModelState[NDimension]) -> None:
        """
        Save a checkpoint for crash recovery.

        Creates a separate checkpoint file that can be used to resume simulation
        from this point.

        :param state: State to checkpoint
        """
        if self._checkpoint_dir is None:
            return

        checkpoint_path = self._checkpoint_dir / f"checkpoint_{state.step:06d}.pkl"
        try:
            checkpoint_store = PickleStore(
                filepath=checkpoint_path, compression="lzma", compression_level=8
            )
            checkpoint_store.dump(states=[state], exist_ok=True, validate=False)

            self._num_checkpoints += 1
            logger.info(
                f"Created checkpoint at step {state.step} ({checkpoint_path.name})"
            )
        except Exception as exc:
            logger.error(f"Failed to create checkpoint at step {state.step}: {exc}")

    def checkpoint(self, step: int) -> ModelState[NDimension]:
        """
        Load a specific checkpoint by step number.

        :param step: Step number of checkpoint to load
        :return: Loaded ModelState from checkpoint
        :raises `StreamError`: If checkpointing not configured
        :raises `FileNotFoundError`: If checkpoint doesn't exist
        """
        if self._checkpoint_dir is None:
            raise StreamError("Checkpointing not configured (no checkpoint_interval)")

        checkpoint_files = self._checkpoint_dir.glob(f"checkpoint_{step:06d}.pkl*")
        checkpoint_path = next(checkpoint_files, None)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_store = PickleStore(filepath=checkpoint_path)
        state = next(checkpoint_store.load(validate=False), None)
        if state is None:
            raise StreamError(f"Checkpoint file is empty: {checkpoint_path}")

        logger.debug(f"Loaded checkpoint from step {step}")
        return state

    def checkpoints(self) -> typing.Generator[ModelState[NDimension], None, None]:
        """
        Load all available checkpoints in order.

        :return: Generator yielding ModelState instances from checkpoints
        :raises `StreamError`: If checkpointing not configured
        """
        if self._checkpoint_dir is None:
            raise StreamError("Checkpointing not configured (no checkpoint_interval)")

        checkpoint_files = sorted(
            self._checkpoint_dir.glob("checkpoint_*.pkl*"),
            key=lambda p: int(p.stem.split("_")[1].split(".")[0]),
        )
        for checkpoint_path in checkpoint_files:
            checkpoint_store = PickleStore(filepath=checkpoint_path)
            state = next(checkpoint_store.load(validate=False), None)
            if state is not None:
                yield state
            else:
                logger.warning(f"Checkpoint file is empty: {checkpoint_path}")

    def list_checkpoints(self) -> typing.List[int]:
        """
        List all available checkpoint step numbers.

        :return: Sorted list of checkpoint step numbers
        :raises `StreamError`: If checkpointing not configured
        """
        if self._checkpoint_dir is None:
            raise StreamError("Checkpointing not configured (no checkpoint_interval)")

        if not self._checkpoint_dir.exists():
            return []

        checkpoints = []
        for path in self._checkpoint_dir.glob("checkpoint_*.pkl*"):
            try:
                step = int(path.stem.split("_")[1].split(".")[0])
                checkpoints.append(step)
            except (ValueError, IndexError):
                logger.warning(f"Invalid checkpoint filename: {path.name}")

        return sorted(checkpoints)

    @property
    def progress(self) -> StreamProgress:
        """
        Get streaming progress statistics.

        :return: Dictionary with progress metrics including:
            - num_yielded: Total states yielded
            - num_saved: Total states saved to store
            - num_checkpoints: Total checkpoints created
            - batch_pending: States in current batch (not yet saved)
            - store_backend: Type of store being used (or None)
            - memory_mb: Current process memory usage in MB
        """
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        return StreamProgress(
            {
                "num_yielded": self._num_yielded,
                "num_saved": self._num_saved,
                "num_checkpoints": self._num_checkpoints,
                "batch_pending": len(self._batch),
                "store_backend": type(self.store).__name__ if self.store else None,
                "memory_mb": memory_mb,
            }
        )

    @property
    def num_yielded(self) -> int:
        """
        Total number of states yielded (including replays).

        :return: Count of yielded states
        """
        return self._num_yielded

    @property
    def is_consumed(self) -> bool:
        """
        Check if the underlying iterable has been exhausted.

        Once consumed, the stream cannot be iterated again unless:
        1. A store was provided and contains data (will auto-replay)
        2. A fresh stream is created with a new iterable

        :return: True if the iterable has been consumed, False otherwise
        """
        return self._consumed

    @property
    def num_saved(self) -> int:
        """
        Number of states saved to store so far.

        :return: Count of saved states
        """
        return self._num_saved

    @property
    def num_checkpoints(self) -> int:
        """
        Number of checkpoints created so far.

        :return: Count of checkpoints
        """
        return self._num_checkpoints

    def __repr__(self) -> str:
        store_info = f"store={self.store}" if self.store else "no store"
        return (
            f"{self.__class__.__name__}({store_info}, batch_size={self.batch_size}, "
            f"yielded={self._num_yielded}, saved={self._num_saved})"
        )
