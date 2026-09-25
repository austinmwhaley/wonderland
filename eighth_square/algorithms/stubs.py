class StubAlgorithm:
    status = "stub"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is a planned algorithm that has not been implemented yet. "
            f"See ALGORITHMS.md for details and source papers."
        )


STUBS = {}


def make_stub(name, description, source, action_space, state_space, family, policy):
    return type(
        name,
        (StubAlgorithm,),
        {
            "family": family,
            "policy": policy,
            "action_space": action_space,
            "state_space": state_space,
            "source": source,
            "description": description,
            "name": name,
        },
    )
