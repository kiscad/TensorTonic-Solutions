import pandas as pd

def data_types_overview(data):
    """
    Returns: dict with 'dtypes', 'type_counts', 'num_columns'
    """
    df = pd.DataFrame(data)
    dtypes = df.dtypes.astype(str).to_dict()
    type_counts = {}
    for typ in dtypes.values():
        type_counts[typ] = type_counts.get(typ, 0) + 1
    return {
        "dtypes": dtypes,
        "type_counts": type_counts,
        "num_columns": len(dtypes),
    }