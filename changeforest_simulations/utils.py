def string_to_kwargs(string):
    if "__" not in string:
        return string, {}

    first_value, list_of_args = string.split("__", 1)

    kwargs = {}

    for args in list_of_args.split("__"):
        k, v = args.split("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass

        kwargs[k] = v

    return first_value, kwargs
