#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2023 California Institute of Technology.
# SPDX-FileCopyrightText: © 2023 Lee McCuller <mcculler@caltech.edu>
# NOTICE: authors should document their contributions in concisely in NOTICE
# with details inline in source files, comments, and docstrings.
"""
"""
import functools


class ConstructorMethod(object):
    """
    This is a python descriptor class, to be used as a decorator

    @constructormethod
    def __init_method__(self, *args, **kw):
        self.abc = 123
        return

    This decorator allows a method to be used either as a classmethod or as an instance method.
    Both will do the same thing, but if used by a class, it will create and instance with __new__
    and then return it
    """
    def __init__(self, method):
        self.method = method

    def __get__(self, inst, cls):
        if inst is None:
            inst = cls.__new__(cls)

        @functools.wraps(self.method)
        def wrap(*args, **kw):
            ret = self.method(inst, *args, **kw)
            # constructors must always return None
            assert (ret is None)
            return inst

        return wrap


def constructormethod(method):
    """
    This is a python descriptor class, to be used as a decorator

    @constructormethod
    def __init_method__(self, *args, **kw):
        self.abc = 123
        return

    This decorator allows a method to be used either as a classmethod or as an instance method.
    Both will do the same thing, but if used by a class, it will create and instance with __new__
    and then return it
    """
    decorator = ConstructorMethod(method)
    functools.update_wrapper(decorator, method)
    return decorator
