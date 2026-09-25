import typing as t

from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from geoparser.paths import geoparser_data_dir

db_location = geoparser_data_dir() / "annotator" / "annotator.db"
db_location.parent.mkdir(parents=True, exist_ok=True)
sqlite_url = f"sqlite:///{db_location}"

engine = create_engine(sqlite_url, echo=False)


@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection: t.Any, connection_record: t.Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_and_tables(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


def get_db() -> t.Iterator[Session]:
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()
