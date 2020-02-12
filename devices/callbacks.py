#! usr/bin/env python
"""

"""
import inspect


def handle(func, *args, **kwargs):
    """
    Handles function callbacks, typically on execution of a class method.
    :param func: func, callback function
    :param args: arguments, optional
    :param kwargs: keyword arguments, optional
    """
    if func is not None:
        try:
            return func(*args, **kwargs)
        except TypeError:
            return func()
    return


def validate(func, allow_args=False, allow_return=False, require_self=False):
    """
    Checks whether a function parameter is valid. The selected rules must be
    observed when implementing a function into this framework, however, this
    is implementation dependent.
    :param func: func, function must be callable or None.
    :param allow_args: bool, if False, func may not have input arguments.
    :param allow_return: bool, if False, func may not contain a return statement.
    :param require_self: bool, if False, func requires the first argument `self`.
    """
    assert callable(func) or func is None, \
        'function must be callable or None'
    if callable(func):
        if not allow_args and not require_self:
            signature = inspect.signature(func)
            assert len(signature.parameters) == 0, \
                'func may not have inputs arguments'
        if not allow_return:
            lines, _ = inspect.getsourcelines(func)
            assert not lines[-1].lstrip().startswith('return'), \
                'func may not contain return statement'
        if require_self:
            signature = inspect.signature(func)
            assert next(iter(signature.parameters)) == 'self', \
                'func requires the first argument `self`'


