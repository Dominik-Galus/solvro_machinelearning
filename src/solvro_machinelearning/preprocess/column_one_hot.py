import pandas as pd


def one_hot_encode_column(
    df: pd.DataFrame,
    column: str,
    is_list_column: bool = False,  # noqa: FBT001, FBT002
) -> tuple[pd.DataFrame, list[str]]:
    df_copy = df.copy()
    if is_list_column:
        all_values = set()
        for value_list in df_copy[column]:
            all_values.update(value_list)
        all_values = sorted(all_values)  # type: ignore[assignment]
    else:
        all_values = sorted(df_copy[column].unique())  # type: ignore[assignment]

    value_to_position = {value: i for i, value in enumerate(all_values)}

    def convert_to_one_hot(value: str) -> list[int]:
        one_hot = [0] * len(all_values)

        if is_list_column:
            if isinstance(value, list):
                for item in value:
                    if item in value_to_position:
                        position = value_to_position[item]
                        one_hot[position] = 1
        elif value in value_to_position:
            position = value_to_position[value]
            one_hot[position] = 1

        return one_hot
    df_copy[column] = df_copy[column].apply(convert_to_one_hot)

    mapping_attr = f"{column}_mapping"
    df_copy.attrs[mapping_attr] = dict(enumerate(all_values))

    return df_copy, list(all_values)
