# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
import os
from collections import namedtuple

if sys.platform != "linux":
    raise ImportError("Only Linux is supported by Dask-CUDA at this time")

import dask
import dask.utils
import dask.dataframe.core
import dask.dataframe.shuffle
import dask.dataframe.multi
import dask.bag.core

def __init_dask_cuda_rocm():
    import sys
    from pyrsmi import rocml
    from hip import hip
    # set HIP_VISIBLE_DEVICES as CUDA_VISIBLE_DEVICES if the latter is not set
    if "HIP_VISIBLE_DEVICES" in os.environ and not "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"]=os.environ["HIP_VISIBLE_DEVICES"]

    # pynvml-to-rocml delegation module
    class Pynvml:
        NVMLError_NotSupported = rocml.ROCMLError_NotSupported

        PyrsmiMemInfo = namedtuple("PyrsmiMemInfo", "total free used")

        class NVMLError(Exception):
            def __new__(typ, value):
                obj = Exception.__new__(str(value))
                obj.value = value

        def nvmlInit(self):
            return rocml.smi_initialize()
        def nvmlDeviceGetCount(self):
            return rocml.smi_get_device_count()
        def nvmlDeviceGetHandleByIndex(self, dev):
            """handle <-> dev"""
            return dev
        def nvmlDeviceGetMemoryInfo(self,handle):
            used = rocml.smi_get_device_memory_used(dev=handle)
            total = rocml.smi_get_device_memory_total(dev=handle)
            free = total - used
            return self.PyrsmiMemInfo(
                total=total,
                free=free,
                used=used,
            )
        def nvmlDeviceGetHandleByIndex(self, dev):
            """handle <-> dev"""
            return dev
        def nvmlDeviceGetUUID(self, dev):
            return str(hip.hipDeviceGetUuid(dev)[1].bytes, encoding="utf-8")
        def nvmlDeviceGetHandleByUUID(self, uuid: str):
            for dev in range(self.nvmlDeviceGetCount()):
                if self.nvmlDeviceGetUUID(dev) == uuid:
                    return dev
            raise ValueError("could not associate the given UUID with any device")

        # TODO: no equivalent found: nvmlDeviceGetCpuAffinity
        # TODO: no equivalent found: nvmlDeviceGetDeviceHandleFromMigDeviceHandle
        # TODO: no equivalent found: nvmlDeviceGetMaxMigDeviceCount
        # TODO: no equivalent found: nvmlDeviceGetMigDeviceHandleByIndex
        # TODO: no equivalent found: nvmlDeviceGetMigMode

        def __getattr__ (self, name: str):
            '''"Automatically delegates ``nvml<Name>`` attribute names to `pyrsmi.rocml`
            Note:
                __getattr__ is only called if the attribute is not found the usual way.
            '''
            return getattr(rocml,name)

    # overwrite/set 'pynvml' entry in module registry:
    sys.modules["pynvml"] = Pynvml()

__init_dask_cuda_rocm()

def is_amd_gpu_available():
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


DASK_USE_ROCM = is_amd_gpu_available()
# print("ROCM device found") if DASK_USE_ROCM else print("ROCM device not found")

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
