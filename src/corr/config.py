from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="CORR",
    settings_files=["settings.toml", ".secrets.toml"],
    validators=[
        Validator("redis_url", must_exist=True),
    ],
)
