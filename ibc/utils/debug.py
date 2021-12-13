# From anzu stuff.

from contextlib import contextmanager
import pdb
import sys
import traceback


@contextmanager
def launch_pdb_on_exception():
    """
    Provides a context that will launch interactive pdb console automatically
    if an exception is raised.

    Example usage with @iex decorator below:

        @iex
        def my_bad_function():
            x = 1
            assert False

        my_bad_function()
        # Should bring up debugger at `assert` statement.
    """
    # Adapted from:
    # https://github.com/gotcha/ipdb/blob/fc83b4f5f/ipdb/__main__.py#L219-L232

    try:
        yield
    except Exception:
        traceback.print_exc()
        _, _, tb = sys.exc_info()
        pdb.post_mortem(tb)
        # Resume original execution.
        raise


# See docs for `launch_pdb_on_exception()`.
iex = launch_pdb_on_exception()
