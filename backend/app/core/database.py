from sqlmodel import SQLModel, create_engine, Session
from .config import settings

# اتصال پایدار
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

def init_db():
    from app import models  # noqa
    SQLModel.metadata.create_all(engine)

class DBSession:
    def __enter__(self):
        # نکته‌ی مهم: expire_on_commit=False تا پس از commit آبجکت‌ها از سشن جدا نشوند
        self.session = Session(engine, expire_on_commit=False)
        return self.session

    def __exit__(self, exc_type, exc, tb):
        self.session.close()
