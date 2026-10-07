import typer


def gazetteer_name(value: str) -> str:
    from geoparser.gazetteer.artifact import validate_gazetteer_name

    try:
        return validate_gazetteer_name(value)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
