import typer


def multiple_of(k: int):
    def multiple_of_k(value: int) -> int:
        if value % k != 0:
            raise typer.BadParameter(f"Value must be a multiple of {k}.")
        return value

    return multiple_of_k
