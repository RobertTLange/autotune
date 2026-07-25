export const OUTPUT_HOLDER_HELPERS = String.raw`
def output_pipe_identities(process):
    streams = (process.stdout, process.stderr)
    if sys.platform.startswith("linux"):
        identities = []
        for stream in streams:
            if stream:
                metadata = os.fstat(stream.fileno())
                identities.append((metadata.st_dev, metadata.st_ino))
        return ("proc", tuple(identities))
    lsof = next((path for path in ("/usr/sbin/lsof", "/usr/bin/lsof") if os.access(path, os.X_OK)), None)
    if not lsof:
        return ("none", ())
    try:
        result = subprocess.run(
            [lsof, "-n", "-P", "-F", "pfn", "-a", "-p", str(process.pid), "-d", "1,2"],
            capture_output=True, text=True, timeout=0.5,
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ("none", ())
    return ("lsof", tuple(line[1:] for line in result.stdout.splitlines() if line.startswith("n")))


def terminate_output_holders(pipe_identities, excluded_pid):
    kind, identities = pipe_identities
    if not identities or os.name == "nt":
        return
    deadline = time.monotonic() + 1
    for pid in output_holder_pids(kind, identities, deadline):
        if pid in (os.getpid(), excluded_pid) or time.monotonic() >= deadline:
            continue
        expected = process_identity(pid, deadline)
        if not expected or expected[0] != os.getuid():
            continue
        if not process_holds_output(pid, kind, identities, deadline):
            continue
        if process_identity(pid, deadline) != expected:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def output_holder_pids(kind, identities, deadline):
    if kind == "proc":
        holders = []
        descriptor_budget = [65536]
        try:
            entries = os.scandir("/proc")
        except OSError:
            return holders
        with entries:
            for index, entry in enumerate(entries):
                if index >= 4096 or time.monotonic() >= deadline:
                    break
                if entry.name.isdigit() and process_holds_output(
                    int(entry.name), kind, identities, deadline, descriptor_budget
                ):
                    holders.append(int(entry.name))
        return holders
    lsof = next((path for path in ("/usr/sbin/lsof", "/usr/bin/lsof") if os.access(path, os.X_OK)), None)
    if not lsof:
        return []
    try:
        result = subprocess.run(
            [lsof, "-n", "-P", "-F", "pfn", "-a", "-u", str(os.getuid()), "-d", "0-1023"],
            capture_output=True, text=True, timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    holders = set()
    pid = None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid and line[1:] in identities:
            holders.add(pid)
    return list(holders)


def process_holds_output(pid, kind, identities, deadline, descriptor_budget=None):
    if time.monotonic() >= deadline:
        return False
    if kind == "proc":
        try:
            descriptors = os.scandir(f"/proc/{pid}/fd")
        except OSError:
            return False
        with descriptors:
            for index, descriptor in enumerate(descriptors):
                if index >= 1024 or time.monotonic() >= deadline:
                    break
                if descriptor_budget is not None:
                    if descriptor_budget[0] <= 0:
                        break
                    descriptor_budget[0] -= 1
                try:
                    metadata = os.stat(descriptor.path)
                except OSError:
                    continue
                if (metadata.st_dev, metadata.st_ino) in identities:
                    return True
        return False
    lsof = next((path for path in ("/usr/sbin/lsof", "/usr/bin/lsof") if os.access(path, os.X_OK)), None)
    if not lsof:
        return False
    try:
        result = subprocess.run(
            [lsof, "-n", "-P", "-F", "pfn", "-a", "-p", str(pid), "-d", "0-1023"],
            capture_output=True, text=True, timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(lsof).parent), "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.startswith("n") and line[1:] in identities for line in result.stdout.splitlines())


def process_identity(pid, deadline):
    if time.monotonic() >= deadline:
        return None
    if sys.platform.startswith("linux"):
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text()
            status = Path(f"/proc/{pid}/status").read_text()
            start_time = stat_text[stat_text.rfind(")") + 2:].split()[19]
            uid = int(next(line.split()[1] for line in status.splitlines() if line.startswith("Uid:")))
            return (uid, start_time)
        except (OSError, ValueError, IndexError, StopIteration):
            return None
    ps = next((path for path in ("/bin/ps", "/usr/bin/ps") if os.access(path, os.X_OK)), None)
    if not ps:
        return None
    try:
        result = subprocess.run(
            [ps, "-o", "pid=,uid=,lstart=,pgid=,comm=", "-p", str(pid)],
            capture_output=True, text=True, timeout=max(0.01, deadline - time.monotonic()),
            env={"PATH": str(Path(ps).parent), "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    fields = result.stdout.strip().split(None, 2)
    if len(fields) < 3 or fields[0] != str(pid):
        return None
    try:
        return (int(fields[1]), result.stdout.strip())
    except ValueError:
        return None
`;
