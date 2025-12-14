"""
Utilities for bridging async/sync boundaries in generator functions.

This module provides adapters for converting between async and sync
generator functions. For async iterator operations like map, filter,
chain, etc., use the `aioitertools` package instead.

Examples:

```python
from lmm_education.asyncutils import async_to_sync, async_gen_to_sync_iter
import aioitertools as aiter

# Using the decorator/wrapper to convert async function to sync
@async_to_sync
async def my_async_gen(x: int):
    for i in range(x):
        yield str(i)

# Now synchronous
for item in my_async_gen(5):
    print(item)

# Or wrap a generator object
async_gen = some_async_function()
for item in async_gen_to_sync_iter(async_gen):
    print(item)

# For async operations, use aioitertools:
async def example():
    chunks = query("hello")
    texts = aiter.map(lambda c: c.text(), chunks)
    async for text in texts:
        print(text)
```
"""

from typing import TypeVar
from collections.abc import Callable, Iterator, AsyncIterator
from functools import wraps
import asyncio

T = TypeVar('T')


def async_to_sync(
    async_gen_func: Callable[..., AsyncIterator[T]],
) -> Callable[..., Iterator[T]]:
    """
    Transforms an async generator function into a sync generator function.

    This decorator/wrapper takes an async generator function and returns a
    synchronous generator function with the same signature. The returned
    function manages its own event loop to iterate over the async generator.

    Args:
        async_gen_func: An async generator function

    Returns:
        A synchronous generator function with the same signature

    Example:
        @async_to_sync
        async def my_async_gen(x: int) -> AsyncIterator[str]:
            for i in range(x):
                await asyncio.sleep(0.1)
                yield str(i)

        # Now callable synchronously:
        for item in my_async_gen(5):
            print(item)

    Note:
        The wrapper creates a new event loop for each call to the function.
        This is necessary to avoid conflicts with existing running loops.
    """

    @wraps(async_gen_func)
    def sync_gen_func(*args, **kwargs) -> Iterator[T]:  # type: ignore
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Call the async generator function to create the generator object
            async_gen: AsyncIterator[T] = async_gen_func(
                *args, **kwargs
            )

            # Iterate over the async generator synchronously
            while True:
                try:
                    item = loop.run_until_complete(
                        async_gen.__anext__()
                    )
                    yield item
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    return sync_gen_func  # type: ignore


def async_gen_to_sync_iter(
    async_gen: AsyncIterator[T],
) -> Iterator[T]:
    """
    Converts an async generator object into a synchronous iterator.

    Unlike async_to_sync which wraps a function, this takes an already-created
    async generator object and yields its items synchronously.

    Args:
        async_gen: An async generator object (not a function!)

    Yields:
        Items from the async generator

    Example:
        async def my_async_gen():
            for i in range(5):
                yield i

        # Create the async generator object
        gen = my_async_gen()

        # Convert and iterate synchronously
        for item in async_gen_to_sync_iter(gen):
            print(item)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                item: T = loop.run_until_complete(
                    async_gen.__anext__()
                )
                yield item
            except StopAsyncIteration:
                break
    finally:
        loop.close()
