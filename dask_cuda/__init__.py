# Apache License
#
# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modification Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

if sys.platform != "linux":
    raise ImportError("Only Linux is supported by Dask-CUDA at this time")

import dask
import dask.utils
import dask.dataframe.core
import dask.dataframe.shuffle
import dask.dataframe.multi
import dask.bag.core
import os


def is_rocm_available():
    """Utility method to check if AMD GPU is available"""
    import subprocess
    import re

    check_cmd ='rocminfo'
    try:
        proc_complete = subprocess.run(
            check_cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        for line in proc_complete.stdout.decode('utf-8').split():
            if re.search(r"(gfx908|gfx90a|gfx940|gfx941|gfx942|gfx1100)", line):
                return True
        return False
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        print('  Error => ', str(err))
        return False


DASK_USE_ROCM = is_rocm_available()
print("ROCM device found") if DASK_USE_ROCM else print("ROCM device not found")

from ._version import __git_commit__, __version__
from .cuda_worker import CUDAWorker
from .explicit_comms.dataframe.shuffle import (
    get_rearrange_by_column_wrapper,
    get_default_shuffle_method,
)
from .local_cuda_cluster import LocalCUDACluster
from .proxify_device_objects import proxify_decorator, unproxify_decorator


if dask.config.get("dataframe.query-planning", None) is not False and dask.config.get(
    "explicit-comms", False
):
    raise NotImplementedError(
        "The 'explicit-comms' config is not yet supported when "
        "query-planning is enabled in dask. Please use the shuffle "
        "API directly, or use the legacy dask-dataframe API "
        "(set the 'dataframe.query-planning' config to `False`"
        "before importing `dask.dataframe`).",
    )


# Monkey patching Dask to make use of explicit-comms when `DASK_EXPLICIT_COMMS=True`
dask.dataframe.shuffle.rearrange_by_column = get_rearrange_by_column_wrapper(
    dask.dataframe.shuffle.rearrange_by_column
)
# We have to replace all modules that imports Dask's `get_default_shuffle_method()`
# TODO: introduce a shuffle-algorithm dispatcher in Dask so we don't need this hack
dask.dataframe.shuffle.get_default_shuffle_method = get_default_shuffle_method
dask.dataframe.multi.get_default_shuffle_method = get_default_shuffle_method
dask.bag.core.get_default_shuffle_method = get_default_shuffle_method


# Monkey patching Dask to make use of proxify and unproxify in compatibility mode
dask.dataframe.shuffle.shuffle_group = proxify_decorator(
    dask.dataframe.shuffle.shuffle_group
)
dask.dataframe.core._concat = unproxify_decorator(dask.dataframe.core._concat)
