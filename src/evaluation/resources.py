"""Process peak memory using standard platform interfaces, without another dependency."""
import os
import sys


def peak_memory_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(handle(), ctypes.byref(counters), counters.cb):
            raise OSError("Cannot read process peak memory")
        return counters.PeakWorkingSetSize / 1e6
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1e6 if sys.platform == "darwin" else value / 1e3
