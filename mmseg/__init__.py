from packaging.version import parse


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    versions = parse(version_str)
    assert versions.release, f'failed to parse version {version_str}'
    release = list(versions.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if versions.is_prerelease:
        raise NotImplementedError
    elif versions.is_postrelease:
        release.extend([1, versions.post])
    else:
        release.extend([0, 0])
    return tuple(release)
