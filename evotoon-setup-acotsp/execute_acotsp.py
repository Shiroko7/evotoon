import subprocess

def execute_ACOTSP(
    instance: str,
    seed: int,
    optimum: int,
    executable_path: str,
    alpha: float,
    beta: float,
    rho: float,
    ants: int,
    nnls: int,
    elitistants: int,
    localsearch: int,
    dlb: int,
):
    # print(instance, seed, executable_path, alpha, beta, rho, q0, ants, nnants, localsearch, optimum)
    """
    function to execute ACOTSP program, it returns its output.
    """
    cmd = [
        executable_path,
        "--tsplibfile",
        instance,
        "--eas",
        "--seed",
        str(seed),
        "--tries",
        "1",
        "--time",
        "5",
        "--tours",
        "50",
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
        "--rho",
        str(rho),
        "--nnls",
        str(nnls),
        "--ants",
        str(ants),
        "--elitistants",
        str(elitistants),
        "--localsearch",
        str(localsearch),
        "--dlb",
        str(dlb),
        "--optimum",
        str(optimum)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = float(result.stdout.decode("utf-8"))

    return output