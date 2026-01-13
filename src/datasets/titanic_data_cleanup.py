import polars as pl


def titanic_cleanup(raw_path: str, processed_path: str):
    df = pl.read_csv(raw_path)
    df = df.drop("PassengerId", "Name", "Ticket", "Cabin")
    df = df.drop_nulls()
    df = df.with_columns(
        pl.col("Sex").replace({"female": 0, "male": 1}).cast(pl.Int8).alias("Sex")
    )
    df = df.with_columns(
        pl.col("Embarked")
        .replace({"S": 0, "C": 1, "Q": 2})
        .cast(pl.Int8)
        .alias("Embarked")
    )
    df.write_csv(processed_path)


if __name__ == "__main__":
    titanic_cleanup("data/raw/train.csv", "data/processed/train.csv")
