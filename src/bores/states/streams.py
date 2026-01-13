"""
Stream model state with optional persistence for memory-efficient simulation workflows.
"""

import logging
import queue
import threading
from pathlib import Path
import typing
from os import PathLike

import numpy as np
from typing_extensions import Self

from bores.errors import StreamError
from bores.states import ModelState
from bores.states.stores import StateStore, PickleStore
from bores.types import NDimension


__all__ = ["StateStream", "StreamProgress"]

logger = logging.getLogger(__name__)


class StreamProgress(typing.TypedDict):
    """Progress statistics for state streaming."""

    yield_count: int
    saved_count: int
    checkpoints_count: int
    batch_pending: int
    store_backend: typing.Optional[str]
    memory_usage: float


_stop_io = object()


class StateStream(typing.Generic[NDimension]):
    """
    Memory-efficient stream for model state iteration with optional persistence.

    Wraps a state generator/iterator and optionally persists states to disk as they're yielded,
    immediately freeing memory. Supports batching for I/O efficiency and async I/O for
    non-blocking disk writes.

    Usage Benefits:
        - Low memory overhead (states persisted immediately after yield)
        - Batch persistence for I/O efficiency
        - Optional async I/O (2-3x speedup when I/O slower than simulation)
        - Optional validation before save
        - Progress tracking and logging
        - Auto-save on context exit (no lost data)
        - Replay from store (load previously saved states)
        - Selective persistence (save only states matching predicate)
        - Checkpointing for crash recovery
        - Memory monitoring with automatic flushing

    Async I/O:
        When `async_io=True`, disk writes happen in a background thread, allowing the
        simulation to continue without blocking on I/O. Particularly effective when
        simulation timesteps are faster than disk write times. Includes backpressure
        mechanism to prevent unbounded memory growth.

    Important: The underlying iterable (typically a generator) can only be consumed once.
    After the first iteration, either:
        - Use `replay()` to load states from the store
        - Create a new stream with a fresh iterable
        - Let `__iter__` automatically use `replay()` if a store exists

    Example (Sync I/O):
    ```python
    store = new_store(store="simulation.zarr", backend="zarr")
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

    Example (Async I/O - High Performance):
    ```python
    stream = StateStream(
        states=run_simulation(...),
        store=store,
        async_io=True,           # Enable background I/O
        batch_size=10,
        max_queue_size=50,       # Limit memory usage
    )
    with stream:
        for state in stream:
            # Simulation continues while previous states written in background
            process_state(state)

        # Force all pending I/O to complete before analysis
        stream.flush(block=True)
    # Auto-cleanup ensures all data written
    ```
    """

    def __init__(
        self,
        states: typing.Optional[typing.Iterable[ModelState[NDimension]]] = None,
        store: typing.Optional[StateStore] = None,
        batch_size: int = 10,
        validate: bool = False,
        auto_save: bool = True,
        auto_replay: bool = True,
        lazy_load: bool = True,
        save_predicate: typing.Optional[
            typing.Callable[[ModelState[NDimension]], bool]
        ] = None,
        checkpoint_interval: typing.Optional[int] = None,
        checkpoint_dir: typing.Optional[PathLike] = None,
        max_batch_memory_usage: typing.Optional[float] = None,
        async_io: bool = False,
        max_queue_size: int = 50,
        io_thread_name: str = "state-io-worker",
        queue_timeout: float = 1.0,
    ) -> None:
        """
        Initialize state stream.

        :param states: Generator or iterator of `ModelState` instances
        :param store: Optional `StateStore` for persistence. If None, states only yielded (no persistence)
        :param batch_size: Number of states to accumulate before flushing to disk (default: 10)
        :param validate: Validate states before persisting (default: False)
        :param auto_save: Automatically flush remaining states on context exit (default: True)
        :param auto_replay: If True, automatically replay from store when iterating after consumption.
            If False, raises `StreamError` instead (default: True)
        :param lazy_load: When replaying from store, use lazy loading if supported (default: True)
        :param save_predicate: Optional function to filter which states to save.
            If provided, only states where save_predicate(state) returns True are saved.
            Example: ```lambda s: s.step % 10 == 0``` (save every 10th state)
        :param checkpoint_interval: Optional interval for checkpointing. If provided,
            creates a checkpoint every N states for crash recovery. Example: 100
        :param checkpoint_dir: Directory to save checkpoints. Required if `checkpoint_interval` is set.
        :param max_batch_memory_usage: Maximum batch memory in MB before forcing flush.
            Estimated by sampling first state's memory footprint. Batch flushes when either
            `batch_size` or `max_batch_memory_usage` threshold is reached. Example: 50.0 MB
        :param async_io: Enable asynchronous I/O for non-blocking disk writes (default: False).
            When enabled, disk writes happen in a background thread, allowing simulation to continue.
            Provides 2-3x speedup when I/O is slower than simulation timesteps.
        :param max_queue_size: Maximum states/batches in I/O queue before blocking (default: 50).
            Acts as backpressure to prevent unbounded memory growth when I/O can't keep up.
            Higher values allow more buffering but use more memory.
        :param io_thread_name: Name for I/O worker thread, useful for debugging (default: "state-io-worker")
        :param queue_timeout: Timeout in seconds for queue operations (default: 1.0).
            Used for responsive shutdown and error checking.
        """
        self.states = iter(states) if states is not None else None
        self.store = store
        self.batch_size = batch_size
        self.validate = validate
        self.auto_save = auto_save
        self.auto_replay = auto_replay
        self.lazy_load = lazy_load
        self.save_predicate = save_predicate
        self.checkpoint_interval = checkpoint_interval
        self.max_batch_memory_usage = max_batch_memory_usage

        self.async_io = async_io
        self.max_queue_size = max_queue_size
        self.io_thread_name = io_thread_name
        self.queue_timeout = queue_timeout

        if self.store is None:
            if self.validate:
                logger.warning(
                    "Validation is enabled but no store provided. States will be validated "
                    "but not persisted."
                )
            if self.auto_save:
                logger.debug(
                    "`auto_save=True` but no store provided. This setting has no effect."
                )
            if self.save_predicate is not None:
                logger.warning(
                    "`save_predicate` provided but no store configured. Predicate will be ignored."
                )
            if self.max_batch_memory_usage is not None:
                logger.warning(
                    "`max_batch_memory_usage` provided but no store configured. Memory-based flushing "
                    "will not occur without persistence."
                )
            if self.async_io:
                logger.warning(
                    "`async_io=True` but no store provided. Async I/O disabled."
                )
                self.async_io = False

        self._batch: typing.List[ModelState[NDimension]] = []
        self._yield_count: int = 0
        self._saved_count: int = 0
        self._checkpoints_count: int = 0
        self._started: bool = False
        self._consumed: bool = False
        self._state_size_mb: typing.Optional[float] = None
        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )

        self._io_queue: typing.Optional[queue.Queue] = None
        self._io_thread: typing.Optional[threading.Thread] = None
        self._io_error: typing.Optional[Exception] = None
        self._shutdown_event: typing.Optional[threading.Event] = None
        self._saved_count_lock = threading.Lock()  # Protects _saved_count in async mode

        if self.checkpoint_interval is not None:
            if self._checkpoint_dir is None:
                raise StreamError(
                    "`checkpoint_dir` must be provided when `checkpoint_interval` is set"
                )
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

        if self.states is None and self.store is None:
            raise StreamError(
                "Either `states` or `store` must be provided. "
                "Cannot create a stream with neither a state source nor a store for replay."
            )

        if self.states is None and self.store is not None:
            # Store-only mode is intended for replay
            if not self.auto_replay:
                logger.warning(
                    "Creating stream with `store` but no `states`. "
                    "Setting `auto_replay=True` since replay is the only available operation."
                )
                self.auto_replay = True
            # Mark as already consumed since there's no states to iterate
            self._consumed = True

        if self.async_io:
            self._start_io_worker()

    def _start_io_worker(self) -> None:
        """Start the background I/O worker thread."""
        self._io_queue = queue.Queue(maxsize=self.max_queue_size)
        self._shutdown_event = threading.Event()
        self._io_thread = threading.Thread(
            target=self._io_worker,
            name=self.io_thread_name,
            daemon=False,
        )
        self._io_thread.start()
        logger.info(
            f"Started I/O worker thread '{self.io_thread_name}' "
            f"(max_queue_size={self.max_queue_size})"
        )

    def _io_worker(self) -> None:
        """
        Background thread worker that handles all I/O operations.

        Continuously pulls batches from queue and writes to store.
        Exits when SENTINEL is received or shutdown event is set.
        """
        logger.debug(f"I/O worker thread started (tid={threading.get_ident()})")
        if self._io_queue is None or self.store is None or self._shutdown_event is None:
            logger.error("I/O infrastructure not properly initialized")
            return

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get batch from queue (timeout to check shutdown periodically)
                    item = self._io_queue.get(timeout=self.queue_timeout)
                    if item is _stop_io:
                        logger.debug("I/O worker received shutdown signal")
                        self._io_queue.task_done()
                        break

                    # Process batch
                    batch: typing.List[ModelState[NDimension]] = item
                    logger.debug(f"I/O worker writing batch of {len(batch)} states")

                    try:
                        self.store.dump(
                            states=batch,
                            exist_ok=True,
                            validate=self.validate,
                        )
                        with self._saved_count_lock:
                            self._saved_count += len(batch)
                        del batch  # Free memory immediately
                        logger.debug(
                            f"I/O worker completed batch "
                            f"(total saved: {self._saved_count})"
                        )
                    except Exception as exc:
                        logger.error(f"I/O worker error during write: {exc}")
                        self._io_error = exc
                        raise
                    finally:
                        self._io_queue.task_done()

                except queue.Empty:
                    continue

        except Exception as exc:
            logger.error(f"I/O worker thread crashed: {exc}")
            self._io_error = exc
        finally:
            logger.debug("I/O worker thread exiting")

    def _check_io_error(self) -> None:
        """Check if I/O thread encountered an error and raise it."""
        if self._io_error is not None:
            raise StreamError(
                f"Background I/O thread failed: {self._io_error}"
            ) from self._io_error

    def _wait_for_queue(self) -> None:
        """Wait for all pending I/O operations to complete."""
        if not self.async_io or self._io_queue is None:
            return

        logger.debug("Waiting for I/O queue to drain...")
        self._io_queue.join()  # Block until all tasks done

        # Check for errors that occurred during drain
        self._check_io_error()
        logger.debug("I/O queue drained successfully")

    def _stop_io_thread(self) -> None:
        """Stop the I/O worker thread gracefully."""
        if not self.async_io or self._io_thread is None:
            return

        logger.debug("Stopping I/O worker thread...")

        if self._io_queue is None or self._shutdown_event is None:
            logger.error("I/O infrastructure not properly initialized")
            return

        # Signal shutdown
        self._shutdown_event.set()
        self._io_queue.put(_stop_io)
        # Wait for thread to finish
        self._io_thread.join(timeout=30.0)

        if self._io_thread.is_alive():
            logger.error(
                "I/O worker thread did not exit within 30s timeout. "
                "Some data may not have been written."
            )
        else:
            logger.info("I/O worker thread stopped successfully")

        # Final error check
        self._check_io_error()

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

        # No states provided, this shouldn't happen as `_consumed` should already be set to false
        # but still handle it
        if self.states is None:
            raise StreamError(
                "No states provided and stream not consumed. "
                "This is an internal error - please report this bug."
            )

        if self.store is None:
            logger.info("No store provided, streaming without persistence")
            for state in self.states:
                self._yield_count += 1
                yield state
            self._consumed = True
            return

        io_mode = "async" if self.async_io else "sync"
        logger.debug(
            f"Streaming to {self.store} ({io_mode} I/O, batch_size={self.batch_size})"
        )

        for state in self.states:
            self._yield_count += 1

            # Check for I/O errors before yielding
            if self.async_io:
                self._check_io_error()

            yield state

            if self._should_save(state=state):
                self._batch.append(state)

                if self._should_flush():
                    self.flush(block=False)

                if self._should_checkpoint(state=state):
                    self._save_checkpoint(state=state)

        if self._batch and self.auto_save:
            logger.debug(f"Flushing final batch of {len(self._batch)} states")
            self.flush(block=False)

        # Mark the stream as consumed
        self._consumed = True
        logger.debug(
            f"Completed stream: {self._yield_count} yielded, {self._saved_count} saved"
        )

    def __enter__(self) -> Self:
        """
        Context manager entry. Prepare for streaming.

        :return: Self for context manager usage
        """
        self._started = True
        if self.store is not None:
            logger.info(f"Started stream session to {self.store!r}")
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
        Context manager exit. Flushes any remaining states and ensure async I/O completion.

        Ensures all states saved even if iteration interrupted.
        For async I/O, waits for background thread to finish all pending writes.

        :param exc_type: Exception type if error occurred
        :param exc_val: Exception value if error occurred
        :param exc_tb: Exception traceback if error occurred
        """
        if self._batch and self.auto_save and self.store is not None:
            logger.warning(f"Flushing {len(self._batch)} unsaved states on exit")
            try:
                self.flush(block=False)  # Enqueue for I/O worker
            except Exception as exc:
                logger.error(f"Failed to flush states on exit: {exc}")

        # Wait for background I/O to complete
        if self.async_io:
            try:
                logger.info("Waiting for background I/O to complete...")
                self._wait_for_queue()
                self._stop_io_thread()
            except Exception as exc:
                logger.error(f"Error during I/O worker shutdown: {exc}")
                if exc_type is None:
                    raise

        if exc_type is None:
            logger.info(
                f"Stream complete: {self._saved_count} states saved, "
                f"{self._checkpoints_count} checkpoints created."
            )
        else:
            logger.error(
                f"Stream interrupted after {self._saved_count} states have been saved: {exc_val}"
            )

    def collect(
        self,
        *steps: int,
        key: typing.Optional[typing.Callable[[ModelState[NDimension]], bool]] = None,
    ) -> typing.Iterator[ModelState[NDimension]]:
        """
        Iterate over states from the stream, optionally filtering by step numbers or a predicate.

        This method provides flexible filtering capabilities:
        - Filter by specific step numbers (positional arguments)
        - Filter by a custom predicate function (key argument)
        - Combine both filters (state must match step AND predicate)
        - No filtering (iterate through all states)

        When filtering by steps, iteration stops early once all requested steps have been
        collected, which can significantly improve performance for large streams.

        :param steps: Optional step numbers to filter. If provided, only states with
            matching step numbers are yielded. Supports any number of step values.
            Example: ``collect(0, 10, 20)`` yields only steps 0, 10, and 20.
        :param key: Optional predicate function to filter states. If provided, only states
            for which ``key(state)`` returns ``True`` are yielded. When combined with
            ``steps``, both conditions must be satisfied.
        :return: Iterator of ``ModelState`` instances matching the filter criteria.

        Examples:
        ```python
        # Iterate through entire stream (no filtering)
        for state in stream.collect():
            process(state)

        # Collect specific steps only
        for state in stream.collect(0, 100, 200):
            process(state)

        # Collect states matching a predicate
        for state in stream.collect(key=lambda s: s.step % 50 == 0):
            process(state)

        # Combine step filter with predicate (both must match)
        for state in stream.collect(0, 50, 100, key=lambda s: s.model.pressure_grid.mean() > 3000):
            process(state)

        # Collect first and last states
        for state in stream.collect(0, -1):  # Note: -1 won't work, use key instead
            process(state)
        ```

        Note:
            - When using ``steps``, the method stops early once all requested steps are found.
            - The ``key`` predicate is evaluated for every state, even those not in ``steps``.
            - For replaying from store with filtering, consider using store's native filtering
              if available for better performance.
        """
        if steps:
            logger.debug(f"Collecting states from stream (filtering steps: {steps})")
            remaining_steps = set(steps)
        else:
            logger.debug("Iterating through entire stream")
            remaining_steps = None

        for state in self:
            # Apply key filter first (if provided)
            if key is not None and not key(state):
                continue

            # Apply step filter (if provided)
            if remaining_steps is not None:
                if state.step in remaining_steps:
                    yield state
                    remaining_steps.discard(state.step)
                    if not remaining_steps:
                        logger.debug(
                            f"Collected all {len(steps)} requested steps, stopping early"
                        )
                        break
            else:
                yield state

    def last(self) -> typing.Optional[ModelState[NDimension]]:
        """
        Get the last state from the stream.

        Iterates through the entire stream and returns the final state.
        Useful for quickly accessing the end result of a simulation.

        :return: The last `ModelState` instance, or None if stream is empty
        """
        last_state: typing.Optional[ModelState[NDimension]] = None
        logger.debug("Retrieving last state from stream")

        iterator = iter(self)
        while True:
            try:
                last_state = next(iterator)
            except StopIteration:
                break
        
        if last_state is not None:
            logger.debug(f"Last state retrieved: step {last_state.step}")
        else:
            logger.debug("Stream is empty, no last state available")
        return last_state

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
        logger.debug(f"Stream consumed: {self._yield_count} states processed")

    def replay(self) -> typing.Iterator[ModelState[NDimension]]:
        """
        Load and iterate over previously saved states from store.

        Useful for:
            - Post-processing saved results
            - Resuming from checkpoint
            - Debugging specific timesteps

        Note: Replaying continues to increment `yield_count` counter. If you replay
        100 states after initially streaming 100 states, `yield_count` will be 200.
        This tracks total states yielded across all operations.

        :return: Iterator over loaded ModelState instances
        :raises `StreamError`: If no store provided
        :raises `StorageError`: If store file doesn't exist or is corrupted
        """
        if self.store is None:
            raise StreamError("Cannot replay: no store provided")

        logger.debug(f"Replaying states from {self.store}")

        for state in self.store.load(lazy=self.lazy_load, validate=self.validate):
            self._yield_count += 1
            yield state

        logger.debug(f"Replay complete: {self._yield_count} states loaded")

    def flush(self, block: bool = False) -> None:
        """
        Manually flush accumulated batch to store.

        For sync I/O: Writes batch immediately to disk.
        For async I/O: Enqueues batch for background writing.

        :param block: If True, wait for I/O thread to complete all pending writes.
            If False (default), just enqueue and return immediately (async behavior).
            Only relevant when `async_io=True`.
        :raises StreamError: If no store provided or I/O error occurred
        """
        if self.store is None:
            raise StreamError("Cannot flush: no store provided")

        if not self._batch:
            if block and self.async_io:
                # Even with empty batch, block might want to wait for queue
                logger.debug("Flush called with empty batch, waiting for queue...")
                self._wait_for_queue()
            else:
                logger.debug("Flush called but batch is empty")
            return

        batch_size = len(self._batch)

        if self.async_io:
            logger.debug(
                f"Enqueuing batch of {batch_size} states to I/O thread (block={block})"
            )

            # Check for errors before enqueuing
            self._check_io_error()
            if (
                self._io_queue is None
                or self.store is None
                or self._shutdown_event is None
            ):
                raise StreamError("I/O infrastructure not properly initialized")

            try:
                # Put batch in queue. May block if queue is full (backpressure)
                self._io_queue.put(self._batch.copy(), timeout=10.0)
                logger.debug(f"Batch enqueued (queue size: ~{self._io_queue.qsize()})")

                # Clear batch and reassign to new list to free memory immediately
                self._batch = []

                # If blocking requested, wait for queue to drain
                if block:
                    self._wait_for_queue()

            except queue.Full:
                logger.error(
                    f"I/O queue full ({self.max_queue_size}) for >10s. "
                    f"Consider increasing `max_queue_size` or ensure `max_queue_size` is a certain magnitude larger than "
                    f"`batch_size` ({self.batch_size})."
                )
                raise StreamError("I/O queue full. Backpressure limit reached")
        else:
            logger.debug(f"Flushing batch of {batch_size} states to {self.store}")

            try:
                self.store.dump(
                    states=self._batch,
                    exist_ok=True,
                    validate=self.validate,
                )
                self._saved_count += batch_size
                logger.debug(
                    f"Flushed {batch_size} states (total saved: {self._saved_count})"
                )
            except Exception as exc:
                logger.error(f"Failed to flush batch: {exc}")
                raise
            finally:
                # Reassign to new list to free memory immediately
                self._batch = []

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

    def _estimate_state_size(self, state: ModelState[NDimension]) -> float:
        """
        Estimate memory footprint of a single state in MB.

        :param state: State to measure
        :return: Estimated size in MB
        """
        if self._state_size_mb is not None:
            return self._state_size_mb

        size_bytes = 0

        for attr_name in dir(state):
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(state, attr_name)
                if isinstance(attr, np.ndarray):
                    size_bytes += attr.nbytes
                elif hasattr(attr, "__dict__"):
                    for nested_attr_name in dir(attr):
                        if nested_attr_name.startswith("_"):
                            continue
                        nested_attr = getattr(attr, nested_attr_name, None)
                        if isinstance(nested_attr, np.ndarray):
                            size_bytes += nested_attr.nbytes
            except Exception:
                continue

        self._state_size_mb = size_bytes / 1024 / 1024
        logger.debug(f"Estimated state size: {self._state_size_mb:.2f} MB")
        return self._state_size_mb

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

        :return: True if batch should be flushed, False otherwise
        """
        if len(self._batch) >= self.batch_size:
            return True

        if self.max_batch_memory_usage is not None and self._batch:
            state_size = self._estimate_state_size(self._batch[0])
            batch_memory_usage = state_size * len(self._batch)

            if batch_memory_usage > self.max_batch_memory_usage:
                logger.warning(
                    f"Batch memory limit reached ({batch_memory_usage:.1f} MB > "
                    f"{self.max_batch_memory_usage:.1f} MB) with {len(self._batch)} states, "
                    f"flushing early"
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

            self._checkpoints_count += 1
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
            raise FileNotFoundError(f"Checkpoint not found for step {step}")

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
            - yield_count: Total states yielded
            - saved_count: Total states saved to store
            - checkpoints_count: Total checkpoints created
            - batch_pending: States in current batch (not yet saved)
            - store_backend: Type of store being used (or None)
            - memory_usage: Estimated batch memory in MB
            - io_queue_size: Size of async I/O queue (if async_io enabled)
            - io_thread_alive: Whether I/O thread is running (if async_io enabled)
        """
        if self._batch and self._state_size_mb is not None:
            batch_memory_usage = self._state_size_mb * len(self._batch)
        else:
            batch_memory_usage = 0.0

        progress = StreamProgress(
            yield_count=self._yield_count,
            saved_count=self._saved_count,
            checkpoints_count=self._checkpoints_count,
            batch_pending=len(self._batch),
            store_backend=type(self.store).__name__ if self.store else None,
            memory_usage=batch_memory_usage,
        )

        # Add async I/O specific stats
        if self.async_io and self._io_queue is not None:
            progress["io_queue_size"] = self._io_queue.qsize()  # type: ignore[typeddict-unknown-key]
            progress["io_thread_alive"] = (  # type: ignore[typeddict-unknown-key]
                self._io_thread.is_alive() if self._io_thread else False
            )
        return progress

    @property
    def yield_count(self) -> int:
        """
        Total number of states yielded (including replays).

        :return: Count of yielded states
        """
        return self._yield_count

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
    def saved_count(self) -> int:
        """
        Number of states saved to store so far.

        :return: Count of saved states
        """
        return self._saved_count

    @property
    def checkpoints_count(self) -> int:
        """
        Number of checkpoints created so far.

        :return: Count of checkpoints
        """
        return self._checkpoints_count

    def __repr__(self) -> str:
        store_info = f"store={self.store}" if self.store else "no store"
        io_mode = "async" if self.async_io else "sync"
        return (
            f"{self.__class__.__name__}({store_info}, {io_mode}, "
            f"batch_size={self.batch_size}, yielded={self._yield_count}, "
            f"saved={self._saved_count})"
        )
